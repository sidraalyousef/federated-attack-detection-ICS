"""Choose the bar strength on client C1 only, report on C2-C5.

``psb_online``'s bar strength (0 = one shared bar, 1 = fully per-sensor bars)
was first picked after looking at results on the whole attack file. This
script redoes the choice cleanly:

* **development set:** client C1 (28 Dec 10:00 -> 29 Dec 13:02, 12 attacks);
* **test set:** clients C2-C5 (pooled), never used for the choice;
* **rule, fixed in advance:** the strength with the highest raw F1 on C1,
  averaged over the seeds; ties -> the lower C1 false-alarm rate.

For every seed and strength the full attack file is streamed in time order
(the model fine-tunes through C1 and carries on into C2-C5, as it would live),
then C1 and C2-C5 are evaluated separately. The result for C2-C5 does not
depend on how the strength was chosen; only which row is reported does.

Run from the repo root:

    python -m experiments.conv_ae_select_c1                    # trains a model per seed
    python -m experiments.conv_ae_select_c1 --weights "experiments/conv_ae_regularization/outputs/baseline/conv_ae.pt"

``--weights`` may contain ``{seed}`` to load one trained model per seed.
Results: ``experiments/conv_ae_select_c1/outputs/`` (gitignored).
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
from src.models import ConvAE, sensor_errors, train_ae
from src.models.online import stream_scores

OUT_DIR = Path("experiments/conv_ae_select_c1/outputs")
STRENGTHS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
DEV = "C1"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--strengths", type=float, nargs="+", default=list(STRENGTHS))
    p.add_argument("--weights", default=None,
                   help="trained ConvAE state_dict; may contain {seed}. Trains if omitted.")
    p.add_argument("--raw-dir", default="data/raw/swat")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--q", type=float, default=99.0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    return p.parse_args()


def metrics(r: dict) -> dict:
    return {"precision": r["raw_precision"], "recall": r["raw_recall"], "raw_f1": r["raw_f1"],
            "pak_auc_f1": r["pak_auc_f1"], "etapr_f1": r["etapr_f1"], "auroc": r["auroc"],
            "fpr": r["fpr"], "attacks": r["n_segments_detected"], "n_attacks": r["n_segments"]}


def select(table: pd.DataFrame, metric: str) -> float:
    """Strength with the best mean C1 ``metric`` over seeds; ties -> lower mean C1 FPR."""
    dev = table[table.split == "C1 (dev)"].groupby("strength")[[metric, "fpr"]].mean()
    dev = dev.assign(neg_fpr=-dev["fpr"]).sort_values([metric, "neg_fpr"], ascending=False)
    return float(dev.index[0])


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print("Loading and preprocessing SWaT (all sensors) ...", flush=True)
    d = prepare_swat(args.raw_dir)
    clients = split_clients(d, n_clients=5)
    train_w = make_windows(d["train"])
    val_w = make_windows(d["val"], stride=EVAL_STRIDE)
    results_path = args.out / "runs.csv"

    for seed in args.seeds:
        torch.manual_seed(seed)
        model = ConvAE(n_features=d["train"].shape[1], window=WINDOW)
        if args.weights:
            path = args.weights.format(seed=seed)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            print(f"\nseed {seed}: loaded {path}", flush=True)
        else:
            print(f"\nseed {seed}: training ...", flush=True)
            train_ae(model, train_w, epochs=args.epochs, device=args.device, seed=seed,
                     verbose=False)
            torch.save(model.state_dict(), args.out / f"conv_ae_seed{seed}.pt")
        val_err = np.full((len(d["val"]), d["train"].shape[1]), np.nan)
        val_err[WINDOW - 1:] = sensor_errors(model, val_w, device=args.device)

        for strength in args.strengths:
            out = stream_scores(copy.deepcopy(model), [c["attack"] for c in clients], val_err,
                                threshold="per_sensor", bar_strength=strength,
                                train_windows=train_w, finetune=True, q=args.q, seed=seed,
                                device=args.device, verbose=False)
            ratios = [ratio_to_threshold(s, t) for s, t in zip(out["scores"], out["thresholds"])]
            scores, labels, groups = points_from_clients(ratios, clients)
            dev = groups == DEV
            rows = [
                {"seed": seed, "strength": strength, "split": "C1 (dev)",
                 **metrics(evaluate(scores[dev], labels[dev], 1.0))},
                {"seed": seed, "strength": strength, "split": "C2-C5 (test)",
                 **metrics(pooled_evaluate(scores[~dev], labels[~dev], groups[~dev], 1.0))},
            ]
            pd.DataFrame(rows).to_csv(results_path, mode="a", index=False,
                                      header=not results_path.exists())
            print(f"  seed {seed} strength {strength:<4}  C1 F1 {rows[0]['raw_f1']:.3f} "
                  f"(FPR {rows[0]['fpr']:.3f}, {rows[0]['attacks']}/{rows[0]['n_attacks']})  |  "
                  f"C2-C5 F1 {rows[1]['raw_f1']:.3f} (FPR {rows[1]['fpr']:.3f}, "
                  f"{rows[1]['attacks']}/{rows[1]['n_attacks']})", flush=True)

    table = pd.read_csv(results_path)
    table = table.drop_duplicates(["seed", "strength", "split"], keep="last")
    cols = ["precision", "recall", "raw_f1", "pak_auc_f1", "etapr_f1", "auroc", "fpr", "attacks"]
    mean = table.groupby(["split", "strength"])[cols].mean()
    std = table.groupby(["split", "strength"])[cols].std()
    chosen = select(table, "raw_f1")
    chosen_etapr = select(table, "etapr_f1")
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print("\nMean over seeds:\n", mean)
    test = table[(table.split == "C2-C5 (test)") & (table.strength == chosen)]
    summary = {
        "rule": "highest mean raw F1 on C1, ties -> lower C1 FPR",
        "chosen_strength": chosen,
        "chosen_strength_if_selected_on_etapr": chosen_etapr,
        "seeds": sorted(table.seed.unique().tolist()),
        "test_C2_C5_mean": test[cols].mean().round(4).to_dict(),
        "test_C2_C5_std": test[cols].std().round(4).to_dict(),
    }
    print("\nChosen on C1:", chosen, "| on eTaPR it would be:", chosen_etapr)
    print("C2-C5 (test) for the chosen strength:", json.dumps(summary["test_C2_C5_mean"]))
    mean.to_csv(args.out / "mean_over_seeds.csv")
    std.to_csv(args.out / "std_over_seeds.csv")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
