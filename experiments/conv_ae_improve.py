"""Can we catch more attacks? Hybrid score and per-stage models, judged on held-out clients.

Methods (all with online fine-tuning, all 43 sensors):

``psb``           current best: one ConvAE, per-sensor dynamic threshold with
                  per-sensor bars (``bar_strength``).
``hybrid``        one ConvAE, alarm on the larger of the per-sensor score and
                  the shared score (window MSE / its own past-24 h median),
                  jointly calibrated on validation. Adds small changes in many
                  sensors together, which the per-sensor score misses.
``stage``         one small ConvAE per SWaT process stage (stage 6 merged into
                  5), per-sensor threshold as in ``psb``. A change in one stage
                  can't be diluted by, or leak into, the other stages.
``stage_hybrid``  per-stage models + hybrid score.

Protocol (same as ``conv_ae_select_c1``): for each method the bar strength
is chosen on client C1 only (highest mean raw F1 over seeds, ties -> lower
false-alarm rate) and results are reported on held-out clients C2-C5, mean
+/- std over seeds. A secondary table adds a 15-minute late-alarm grace period
(``src.eval.late_alarms``); it must never replace the headline numbers.

Run from the repo root:

    python -m experiments.conv_ae_improve                                  # everything (~2.5 h CPU)
    python -m experiments.conv_ae_improve --methods hybrid --strengths 0 0.25 0.5
    python -m experiments.conv_ae_improve --summary-only                   # re-print from saved runs

Runs are appended to ``experiments/conv_ae_improve/outputs/runs.csv`` as they
finish; trained models are cached per seed.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data_loaders.clients import points_from_clients, split_clients
from src.data_loaders.preprocessing import EVAL_STRIDE, WINDOW, make_windows, prepare_swat
from src.eval import evaluate
from src.eval.adaptive import ratio_to_threshold
from src.eval.federated import pooled_evaluate
from src.eval.late_alarms import GRACE_S, evaluate_with_grace
from src.models import ConvAE, StageConvAE, group_by_stage, sensor_errors, train_ae
from src.models.online import stream_scores

OUT_DIR = Path("experiments/conv_ae_improve/outputs")
METHODS = {                      # name -> (model, threshold mode)
    "psb": ("conv", "per_sensor"),
    "hybrid": ("conv", "hybrid"),
    "stage": ("stage", "per_sensor"),
    "stage_hybrid": ("stage", "hybrid"),
}
DEV = "C1"
COLS = ["precision", "recall", "raw_f1", "pak_auc_f1", "etapr_f1", "auroc", "fpr", "attacks"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--conv-weights", default=None,
                   help="trained ConvAE state_dict, may contain {seed}; trained if omitted")
    p.add_argument("--raw-dir", default="data/raw/swat")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--q", type=float, default=99.0)
    p.add_argument("--grace", type=int, default=GRACE_S)
    p.add_argument("--device", default=None)
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    return p.parse_args()


def metrics(r: dict) -> dict:
    return {"precision": r["raw_precision"], "recall": r["raw_recall"], "raw_f1": r["raw_f1"],
            "pak_auc_f1": r["pak_auc_f1"], "etapr_f1": r["etapr_f1"], "auroc": r["auroc"],
            "fpr": r["fpr"], "attacks": r["n_segments_detected"], "n_attacks": r["n_segments"]}


def get_model(kind: str, seed: int, d: dict, train_w, args) -> torch.nn.Module:
    n = d["train"].shape[1]
    torch.manual_seed(seed)
    if kind == "conv":
        model = ConvAE(n_features=n, window=WINDOW)
        path = Path(args.conv_weights.format(seed=seed)) if args.conv_weights else \
            args.out / f"conv_seed{seed}.pt"
    else:
        model = StageConvAE(group_by_stage(d["features"]), window=WINDOW)
        path = args.out / f"stage_seed{seed}.pt"
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"  {kind} seed {seed}: loaded {path}", flush=True)
    else:
        print(f"  {kind} seed {seed}: training {args.epochs} epochs ...", flush=True)
        train_ae(model, train_w, epochs=args.epochs, device=args.device, seed=seed, verbose=False)
        torch.save(model.state_dict(), path)
    return model


def summarise(table: pd.DataFrame, out: Path) -> dict:
    summary = {}
    rows = []
    for method in table.method.unique():
        t = table[table.method == method]
        dev = t[t.split == "C1 (dev)"].groupby("strength")[["raw_f1", "fpr"]].mean()
        dev = dev.assign(neg_fpr=-dev.fpr).sort_values(["raw_f1", "neg_fpr"], ascending=False)
        chosen = float(dev.index[0])
        test = t[(t.split == "C2-C5 (test)") & (t.strength == chosen)]
        mean, std = test[COLS + ["attacks_grace", "raw_f1_grace", "fpr_grace"]].mean(), \
            test[COLS + ["attacks_grace", "raw_f1_grace", "fpr_grace"]].std()
        summary[method] = {"chosen_strength": chosen, "seeds": sorted(test.seed.tolist()),
                           "test_mean": mean.round(4).to_dict(), "test_std": std.round(4).to_dict()}
        rows.append({"method": method, "strength (chosen on C1)": chosen, "seeds": len(test),
                     **{c: f"{mean[c]:.3f} ± {std[c]:.3f}" for c in COLS}})
    res = pd.DataFrame(rows).set_index("method")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print("\nHeld-out C2-C5, bar strength chosen on C1 (mean ± std over seeds):\n", res)
    res.to_csv(out / "summary_table.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scores").mkdir(exist_ok=True)
    runs = args.out / "runs.csv"
    if args.summary_only:
        summarise(pd.read_csv(runs), args.out)
        return

    print("Loading and preprocessing SWaT (all sensors) ...", flush=True)
    d = prepare_swat(args.raw_dir)
    clients = split_clients(d, n_clients=5)
    train_w = make_windows(d["train"])
    val_w = make_windows(d["val"], stride=EVAL_STRIDE)

    for seed in args.seeds:
        models = {}
        for method in args.methods:
            kind, mode = METHODS[method]
            if kind not in models:
                m = get_model(kind, seed, d, train_w, args)
                v = np.full((len(d["val"]), d["train"].shape[1]), np.nan)
                v[WINDOW - 1:] = sensor_errors(m, val_w, device=args.device)
                models[kind] = (m, v)
            model, val_err = models[kind]
            for strength in args.strengths:
                out = stream_scores(copy.deepcopy(model), [c["attack"] for c in clients], val_err,
                                    threshold=mode, bar_strength=strength, train_windows=train_w,
                                    finetune=True, q=args.q, seed=seed, device=args.device,
                                    verbose=False)
                ratios = [ratio_to_threshold(s, t) for s, t in zip(out["scores"], out["thresholds"])]
                scores, labels, groups = points_from_clients(ratios, clients)
                np.savez_compressed(args.out / "scores" / f"{method}_s{strength}_seed{seed}.npz",
                                    scores=scores.astype(np.float32), labels=labels, groups=groups)
                dev = groups == DEV
                g = evaluate_with_grace(scores[~dev], labels[~dev], 1.0, grace=args.grace,
                                        include_etapr=False)
                base = {"method": method, "seed": seed, "strength": strength}
                rows = [
                    {**base, "split": "C1 (dev)",
                     **metrics(evaluate(scores[dev], labels[dev], 1.0))},
                    {**base, "split": "C2-C5 (test)",
                     **metrics(pooled_evaluate(scores[~dev], labels[~dev], groups[~dev], 1.0)),
                     "attacks_grace": g["attacks_caught_grace"], "raw_f1_grace": g["raw_f1"],
                     "fpr_grace": g["fpr"]},
                ]
                pd.DataFrame(rows).to_csv(runs, mode="a", index=False, header=not runs.exists())
                print(f"  {method:12s} seed {seed} strength {strength:<4} | "
                      f"C1 F1 {rows[0]['raw_f1']:.3f} ({rows[0]['attacks']}/{rows[0]['n_attacks']}) | "
                      f"C2-C5 F1 {rows[1]['raw_f1']:.3f} FPR {rows[1]['fpr']:.4f} "
                      f"attacks {rows[1]['attacks']}/{rows[1]['n_attacks']} "
                      f"(late-alarm: {g['attacks_caught_grace']})", flush=True)
    summarise(pd.read_csv(runs).drop_duplicates(["method", "seed", "strength", "split"],
                                               keep="last"), args.out)


if __name__ == "__main__":
    main()
