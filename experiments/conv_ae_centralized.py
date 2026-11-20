"""Centralized Conv-AE baseline on SWaT, scored on the federated client slices.

Pipeline:

1. ``prepare_swat`` -> ``split_clients`` (5 clients by default).
2. Train on ``make_windows(d["train"])`` (stride 10).
3. Threshold: score the validation windows (stride 1), map them to seconds,
   ``threshold_from_normal`` (99th percentile).
4. Score each client's attack windows (stride 1) ->
   ``points_from_clients`` -> ``evaluate_groups``.
5. Sanity check: a ``random_scores`` row, thresholded with the same rule on
   random validation scores and evaluated on the same seconds.

Run from the repo root:

    python -m experiments.conv_ae_centralized                 # defaults
    python -m experiments.conv_ae_centralized --epochs 2      # quick check
    python -m experiments.conv_ae_centralized --drop-drifting --out <dir>

Outputs go to ``experiments/conv_ae_centralized/outputs/`` (gitignored):
model weights, training history, per-second scores (so the tables can be
rebuilt without retraining), per-client tables and the comparison table.

Other experiments can call :func:`run` with their own (e.g. transformed)
version of the ``prepare_swat`` output.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
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
    best_f1_threshold,
    evaluate_groups,
    random_scores,
    summary_row,
    threshold_from_normal,
    window_scores_to_points,
)
from src.eval.federated import client_summary, client_table, compare_settings, pooled_evaluate
from src.models import ConvAE, score_windows, train_ae

OUT_DIR = Path("experiments/conv_ae_centralized/outputs")
# Analyser sensors that drift in the attack file (see PREPROCESSING.md).
DRIFTING = ("AIT201", "AIT203", "AIT402", "AIT502")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw-dir", default="data/raw/swat")
    p.add_argument("--n-clients", type=int, default=5, choices=(3, 5))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--q", type=float, default=99.0, help="threshold percentile on validation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
    p.add_argument("--drop-drifting", action="store_true",
                   help=f"also drop the drifting analysers {', '.join(DRIFTING)}")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="extra sensors to drop (on top of P201/P403)")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    return p.parse_args()


def oracle_raw_f1(scores, labels) -> float:
    """Raw F1 at the best threshold chosen WITH test labels. An upper bound, not a result."""
    return best_f1_threshold(scores, labels, protocol="raw")[1]


def run(d: Mapping, out: Path, *, n_clients: int = 5, epochs: int = 20,
        batch_size: int = 256, lr: float = 1e-3, q: float = 99.0, seed: int = 0,
        device=None, name: str = "ConvAE centralized", verbose: bool = True) -> dict:
    """Train, threshold, score and evaluate on ``d`` (a ``prepare_swat``-like dict).

    ``d`` needs ``train``, ``val``, ``attack`` (2-D arrays), ``attack_labels``
    and the three timestamp arrays. Saves everything to ``out`` and returns
    ``{"pooled", "pooled_random", "per_client", "comparison", "threshold",
    "oracle_raw_f1", "history"}``.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    clients = split_clients(d, n_clients=n_clients)
    if verbose:
        print(summary(clients), "\n", flush=True)

    train_w = make_windows(d["train"])                      # stride 10
    val_w = make_windows(d["val"], stride=EVAL_STRIDE)      # stride 1
    n_features = d["train"].shape[1]
    if verbose:
        print(f"train windows {train_w.shape}, val windows {val_w.shape}", flush=True)

    # Train
    torch.manual_seed(seed)
    model = ConvAE(n_features=n_features, window=WINDOW)
    if verbose:
        print(f"ConvAE: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    history = train_ae(model, train_w, epochs=epochs, batch_size=batch_size, lr=lr,
                       val_windows=val_w[::10], device=device, seed=seed, verbose=verbose)
    torch.save(model.state_dict(), out / "conv_ae.pt")

    # Threshold from normal validation data (no test labels)
    val_pts = window_scores_to_points(score_windows(model, val_w, device=device),
                                      len(d["val"]), WINDOW)
    thr = threshold_from_normal(val_pts, method="percentile", q=q)

    # Score each client's attack windows
    test_ws = [score_windows(model, client_windows(c)["attack"], device=device)
               for c in clients]
    scores, labels, groups = points_from_clients(test_ws, clients)
    res_model = evaluate_groups(scores, labels, groups, thr)

    # Random baseline on the same seconds, same threshold rule
    rnd = random_scores(len(scores), seed=seed)
    rnd[np.isnan(scores)] = np.nan
    thr_rnd = threshold_from_normal(random_scores(len(val_pts), seed=seed + 1), q=q)
    res_rnd = evaluate_groups(rnd, labels, groups, thr_rnd)

    table = client_table(res_model)
    comparison = compare_settings({name: res_model, "random": res_rnd})
    pooled = pooled_evaluate(scores, labels, groups, thr)
    pooled_rnd = pooled_evaluate(rnd, labels, groups, thr_rnd)
    oracle = oracle_raw_f1(scores, labels)

    if verbose:
        print(f"\nthreshold (q={q}) = {thr:.6g}")
        with pd.option_context("display.width", 160, "display.max_columns", 30,
                               "display.float_format", "{:.3f}".format):
            print(f"\nPer client ({name}):\n", table)
            print("\nAcross clients:\n", client_summary(table))
            print("\nMean over clients with attacks:\n", comparison)
        print("\nPooled (all clients as one test set):")
        print(summary_row(pooled, name), f" FPR={pooled['fpr']:.3f}")
        print(summary_row(pooled_rnd, "random"), f" FPR={pooled_rnd['fpr']:.3f}")
        print(f"oracle raw F1 (threshold picked on test labels, upper bound): {oracle:.3f}")

    np.savez_compressed(out / "scores.npz", val_points=val_pts, test_points=scores,
                        labels=labels, groups=groups, random_points=rnd)
    table.to_csv(out / "per_client.csv")
    client_table(res_rnd).to_csv(out / "per_client_random.csv")
    comparison.to_csv(out / "comparison.csv")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    config = {"name": name, "n_clients": n_clients, "epochs": epochs,
              "batch_size": batch_size, "lr": lr, "q": q, "seed": seed,
              "features": list(d.get("features", [])), "threshold": thr,
              "threshold_random": thr_rnd, "oracle_raw_f1": oracle,
              "minutes": round((time.perf_counter() - t0) / 60, 1)}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    if verbose:
        print(f"\nSaved to {out}  ({config['minutes']} min)")
    return {"pooled": pooled, "pooled_random": pooled_rnd, "per_client": table,
            "comparison": comparison, "threshold": thr, "oracle_raw_f1": oracle,
            "history": history}


def main():
    args = parse_args()
    exclude = tuple(dict.fromkeys(
        [*EXCLUDE, *(DRIFTING if args.drop_drifting else ()), *args.exclude]))
    print("Loading and preprocessing SWaT ...", flush=True)
    d = prepare_swat(args.raw_dir, exclude=exclude)
    print(f"{len(d['features'])} sensors kept; excluded: {', '.join(d['excluded'])}")
    run(d, args.out, n_clients=args.n_clients, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, q=args.q, seed=args.seed,
        device=args.device)


if __name__ == "__main__":
    main()
