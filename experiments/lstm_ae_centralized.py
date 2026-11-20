"""Centralized LSTM-AE baseline on SWaT, scored on the federated client slices.

Same pipeline as ``conv_ae_centralized`` with the model swapped, so the two
baselines can be compared row for row:

1. ``prepare_swat`` -> ``split_clients`` (5 clients by default).
2. Train on ``make_windows(d["train"])`` (stride 10).
3. Threshold: score the validation windows (stride 1), map them to seconds,
   ``threshold_from_normal`` (99th percentile).
4. Score each client's attack windows (stride 1) ->
   ``points_from_clients`` -> ``evaluate_groups``.
5. Sanity check: a ``random_scores`` row, thresholded with the same rule on
   random validation scores and evaluated on the same seconds.

Run from the repo root:

    python -m experiments.lstm_ae_centralized                 # defaults
    python -m experiments.lstm_ae_centralized --epochs 2      # quick check
    python -m experiments.lstm_ae_centralized --drop-drifting --out <dir>

Outputs go to ``experiments/lstm_ae_centralized/outputs/`` (gitignored):
model weights, training history, per-second scores (so the tables can be
rebuilt without retraining), per-client tables and the comparison table.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data_loaders.clients import client_windows, points_from_clients, split_clients, summary
from src.data_loaders.preprocessing import (
    EVAL_STRIDE,
    EXCLUDE,
    WINDOW,
    make_windows,
    prepare_swat,
)
from src.eval import (
    evaluate_groups,
    random_scores,
    summary_row,
    threshold_from_normal,
    window_scores_to_points,
)
from src.eval.federated import client_summary, client_table, compare_settings, pooled_evaluate
from src.models import LSTMAE, score_windows, train_ae

OUT_DIR = Path("experiments/lstm_ae_centralized/outputs")
# Analyser sensors that drift in the attack file (see PREPROCESSING.md).
DRIFTING = ("AIT201", "AIT203", "AIT402", "AIT502")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw-dir", default="data/raw/swat")
    p.add_argument("--n-clients", type=int, default=5, choices=(3, 5))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64, help="LSTM hidden size")
    p.add_argument("--latent", type=int, default=16, help="bottleneck size")
    p.add_argument("--layers", type=int, default=1, help="stacked LSTM layers")
    p.add_argument("--q", type=float, default=99.0, help="threshold percentile on validation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
    p.add_argument("--drop-drifting", action="store_true",
                   help=f"also drop the drifting analysers {', '.join(DRIFTING)}")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="extra sensors to drop (on top of P201/P403)")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # 1. Data
    print("Loading and preprocessing SWaT ...", flush=True)
    exclude = tuple(dict.fromkeys(
        [*EXCLUDE, *(DRIFTING if args.drop_drifting else ()), *args.exclude]))
    d = prepare_swat(args.raw_dir, exclude=exclude)
    print(f"{len(d['features'])} sensors kept; excluded: {', '.join(d['excluded'])}")
    clients = split_clients(d, n_clients=args.n_clients)
    print(summary(clients), "\n", flush=True)

    train_w = make_windows(d["train"])                      # stride 10
    val_w = make_windows(d["val"], stride=EVAL_STRIDE)      # stride 1
    print(f"train windows {train_w.shape}, val windows {val_w.shape}", flush=True)

    # 2. Train
    torch.manual_seed(args.seed)
    model = LSTMAE(n_features=d["train"].shape[1], window=WINDOW, hidden=args.hidden,
                   latent=args.latent, num_layers=args.layers)
    print(f"LSTMAE: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    history = train_ae(model, train_w, epochs=args.epochs, batch_size=args.batch_size,
                       lr=args.lr, val_windows=val_w[::10], device=args.device,
                       seed=args.seed)
    torch.save(model.state_dict(), args.out / "lstm_ae.pt")

    # 3. Threshold from normal validation data (no test labels)
    val_pts = window_scores_to_points(score_windows(model, val_w, device=args.device),
                                      len(d["val"]), WINDOW)
    thr = threshold_from_normal(val_pts, method="percentile", q=args.q)
    print(f"\nthreshold (q={args.q}) = {thr:.6g}", flush=True)

    # 4. Score each client's attack windows
    test_ws = [score_windows(model, client_windows(c)["attack"], device=args.device)
               for c in clients]
    scores, labels, groups = points_from_clients(test_ws, clients)
    res_model = evaluate_groups(scores, labels, groups, thr)

    # 5. Random baseline on the same seconds, same threshold rule
    rnd = random_scores(len(scores), seed=args.seed)
    rnd[np.isnan(scores)] = np.nan
    thr_rnd = threshold_from_normal(random_scores(len(val_pts), seed=args.seed + 1), q=args.q)
    res_rnd = evaluate_groups(rnd, labels, groups, thr_rnd)

    # Tables
    name = "LSTMAE centralized"
    table = client_table(res_model)
    comparison = compare_settings({name: res_model, "random": res_rnd})
    pooled = {name: pooled_evaluate(scores, labels, groups, thr),
              "random": pooled_evaluate(rnd, labels, groups, thr_rnd)}

    with pd.option_context("display.width", 160, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        print("\nPer client (LSTMAE centralized):\n", table)
        print("\nAcross clients:\n", client_summary(table))
        print("\nMean over clients with attacks:\n", comparison)
    print("\nPooled (all clients as one test set):")
    for n, r in pooled.items():
        print(summary_row(r, n))

    # Save
    np.savez_compressed(args.out / "scores.npz", val_points=val_pts, test_points=scores,
                        labels=labels, groups=groups, random_points=rnd)
    table.to_csv(args.out / "per_client.csv")
    client_table(res_rnd).to_csv(args.out / "per_client_random.csv")
    comparison.to_csv(args.out / "comparison.csv")
    pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)
    config = {**vars(args), "out": str(args.out), "excluded": d["excluded"],
              "threshold": thr,
              "threshold_random": thr_rnd, "minutes": round((time.perf_counter() - t0) / 60, 1)}
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"\nSaved to {args.out}  ({config['minutes']} min)")


if __name__ == "__main__":
    main()
