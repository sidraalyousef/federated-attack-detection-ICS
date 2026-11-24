"""Centralized Conv-AE on SWaT with drift-robust scoring, keeping ALL sensors.

Trains (or loads) the same ConvAE as ``conv_ae_centralized`` on all 43
sensors, then compares eight ways of turning reconstruction errors into alarms:

``fixed``      current setup: window MSE, fixed threshold = 99th percentile of
               validation scores.
``robust``     per-sensor errors normalised by their validation median / IQR
               (GDN), score = 3rd-largest sensor (3 sensors must be unusual
               together), fixed threshold = 99th percentile of validation.
``dynamic``    window MSE, threshold = factor x median of the past 24 h of
               scores (factor = 99th percentile / median on validation).
``online``     ``dynamic`` + every 6 h, fine-tune the model on that block's
               windows that stayed below the threshold (label-free), with
               replay of training windows.
``ps_dynamic`` per-sensor dynamic threshold: each sensor's error divided by
               its own past-24 h median, score = largest ratio, fixed
               threshold = 99th percentile of the same score on validation.
               Stops drifting sensors from masking attacks on other sensors.
``ps_online``  ``ps_dynamic`` + the same online fine-tuning as ``online``.
``psb_dynamic`` / ``psb_online``  like ``ps_*``, but each sensor's ratio is
               first divided by its own validation 99th percentile raised to
               ``--bar-strength`` (0 = shared bar, 1 = fully per-sensor), so
               quiet sensors get tighter bars than noisy ones.

Plus the usual ``random`` sanity-check row. All rules are fixed in advance on
validation data; no test labels are used to pick anything.

Run from the repo root:

    python -m experiments.conv_ae_adaptive                           # train + all configs
    python -m experiments.conv_ae_adaptive --weights <path/conv_ae.pt>  # reuse a trained model
    python -m experiments.conv_ae_adaptive --configs fixed dynamic

Outputs (gitignored): ``experiments/conv_ae_adaptive/outputs/``.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data_loaders.clients import client_windows, points_from_clients, split_clients, summary
from src.data_loaders.preprocessing import EVAL_STRIDE, WINDOW, make_windows, prepare_swat
from src.eval import evaluate_groups, random_scores, threshold_from_normal
from src.eval.adaptive import aggregate, fit_robust, ratio_to_threshold, robust_z
from src.eval.federated import client_table, pooled_evaluate
from src.models import ConvAE, sensor_errors, train_ae
from src.models.online import stream_scores

OUT_DIR = Path("experiments/conv_ae_adaptive/outputs")
CONFIGS = ("fixed", "robust", "dynamic", "online", "ps_dynamic", "ps_online",
           "psb_dynamic", "psb_online")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    p.add_argument("--raw-dir", default="data/raw/swat")
    p.add_argument("--n-clients", type=int, default=5, choices=(3, 5))
    p.add_argument("--weights", type=Path, default=None,
                   help="load a trained ConvAE state_dict instead of training")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--q", type=float, default=99.0, help="validation percentile for all threshold rules")
    p.add_argument("--k", type=int, default=3, help="robust: score = k-th most unusual sensor")
    p.add_argument("--threshold-hours", type=float, default=24, help="dynamic: rolling median window")
    p.add_argument("--bar-strength", type=float, default=0.25,
                   help="psb_*: 0 = one shared bar, 1 = fully per-sensor bars")
    p.add_argument("--update-hours", type=float, default=6, help="online: fine-tune every N hours")
    p.add_argument("--finetune-epochs", type=int, default=3)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    return p.parse_args()


def to_points(window_values: np.ndarray, n_points: int) -> np.ndarray:
    """Window values (stride 1) -> per second, each window's value on its last second.

    Works for 1-D scores and 2-D per-sensor errors; the first WINDOW-1 rows are NaN.
    """
    shape = (n_points,) + window_values.shape[1:]
    out = np.full(shape, np.nan)
    out[WINDOW - 1:] = window_values
    return out


def summarise(name, pooled) -> dict:
    return {
        "config": name,
        "precision": pooled["raw_precision"], "recall": pooled["raw_recall"],
        "raw_f1": pooled["raw_f1"], "pak_auc_f1": pooled["pak_auc_f1"],
        "etapr_f1": pooled["etapr_f1"], "auroc": pooled["auroc"], "fpr": pooled["fpr"],
        "attacks": f"{pooled['n_segments_detected']}/{pooled['n_segments']}",
    }


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    torch.manual_seed(args.seed)

    print("Loading and preprocessing SWaT (all sensors) ...", flush=True)
    d = prepare_swat(args.raw_dir)
    clients = split_clients(d, n_clients=args.n_clients)
    n_feat = d["train"].shape[1]
    print(summary(clients), "\n", flush=True)
    train_w = make_windows(d["train"])
    val_w = make_windows(d["val"], stride=EVAL_STRIDE)

    model = ConvAE(n_features=n_feat, window=WINDOW)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        print(f"loaded {args.weights}")
    else:
        train_ae(model, train_w, epochs=args.epochs, val_windows=val_w[::10],
                 device=args.device, seed=args.seed)
        torch.save(model.state_dict(), args.out / "conv_ae.pt")

    # Errors of the trained (frozen) model: per sensor, per second.
    val_err = to_points(sensor_errors(model, val_w, device=args.device), len(d["val"]))
    test_err = [to_points(sensor_errors(model, client_windows(c)["attack"], device=args.device),
                          len(c["attack"])) for c in clients]
    val_mse = val_err.mean(axis=1)   # = window MSE per second (NaN in warm-up)

    results, tables, saved = {}, {}, {}

    def record(name, score_lists_or_points, threshold):
        if isinstance(score_lists_or_points, list):          # per-client window values
            scores, labels, groups = points_from_clients(score_lists_or_points, clients)
        else:
            scores, labels, groups = score_lists_or_points
        per_client = evaluate_groups(scores, labels, groups, threshold)
        pooled = pooled_evaluate(scores, labels, groups, threshold)
        results[name] = summarise(name, pooled)
        tables[name] = client_table(per_client)
        saved[name] = scores
        print(f"{name:11s} precision {pooled['raw_precision']:.3f}  recall {pooled['raw_recall']:.3f}  "
              f"raw F1 {pooled['raw_f1']:.3f}  eTaPR F1 {pooled['etapr_f1']:.3f}  "
              f"FPR {pooled['fpr']:.3f}  attacks {pooled['n_segments_detected']}/{pooled['n_segments']}",
              flush=True)
        return labels, groups

    labels = groups = None
    if "fixed" in args.configs:
        thr = threshold_from_normal(val_mse, q=args.q)
        labels, groups = record("fixed", [e[WINDOW - 1:].mean(axis=1) for e in test_err], thr)

    if "robust" in args.configs:
        med, iqr = fit_robust(val_err)
        val_s = aggregate(robust_z(val_err, med, iqr), "kth", args.k)
        thr = threshold_from_normal(val_s, q=args.q)
        labels, groups = record(
            "robust", [aggregate(robust_z(e[WINDOW - 1:], med, iqr), "kth", args.k) for e in test_err], thr)

    log = {}
    for name in ("dynamic", "online", "ps_dynamic", "ps_online", "psb_dynamic", "psb_online"):
        if name not in args.configs:
            continue
        per_sensor = name.startswith("ps")
        finetune = name.endswith("online")
        print(f"\n{name}: streaming the attack file in time order ...", flush=True)
        out = stream_scores(copy.deepcopy(model), [c["attack"] for c in clients],
                            val_err if per_sensor else val_mse,
                            threshold="per_sensor" if per_sensor else "shared",
                            bar_strength=args.bar_strength if name.startswith("psb") else 0.0,
                            train_windows=train_w, finetune=finetune,
                            threshold_window=int(args.threshold_hours * 3600), q=args.q,
                            update_every=int(args.update_hours * 3600),
                            finetune_epochs=args.finetune_epochs, finetune_lr=args.finetune_lr,
                            seed=args.seed, device=args.device, verbose=finetune)
        ratios = [ratio_to_threshold(s, t) for s, t in zip(out["scores"], out["thresholds"])]
        labels, groups = record(name, ratios, 1.0)
        log[name] = pd.DataFrame(out["updates"])
        log[name].to_csv(args.out / f"{name}_blocks.csv", index=False)

    # Random sanity check on the same seconds, same fixed rule.
    if labels is not None:
        ref = next(iter(saved.values()))
        rnd = random_scores(len(ref), seed=args.seed)
        rnd[np.isnan(ref)] = np.nan
        thr_rnd = threshold_from_normal(random_scores(len(val_mse), seed=args.seed + 1), q=args.q)
        record("random", (rnd, labels, groups), thr_rnd)

    comparison = pd.DataFrame(list(results.values())).set_index("config")
    fpr = pd.DataFrame({k: v["fpr"] for k, v in tables.items()})
    comparison.to_csv(args.out / "comparison.csv")
    fpr.to_csv(args.out / "per_client_fpr.csv")
    for k, v in tables.items():
        v.to_csv(args.out / f"per_client_{k}.csv")
    np.savez_compressed(args.out / "scores.npz", labels=labels, groups=groups,
                        **{k: v for k, v in saved.items()})
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\nPooled results (all 43 sensors):\n", comparison)
        print("\nFalse-alarm rate per client:\n", fpr)
    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "n_features": n_feat, "minutes": round((time.perf_counter() - t0) / 60, 1)}
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"\nSaved to {args.out}  ({config['minutes']} min)")


if __name__ == "__main__":
    main()
