"""Try two fixes for sensor drift on the centralized Conv-AE, without touching
the shared preprocessing.

``src/data_loaders/preprocessing.py`` and ``data/processed/swat_scaled.npz``
are NOT changed. Each variant is built in memory from the normal
``prepare_swat`` output and only used here:

* ``base``  - the shared preprocessing as it is (43 sensors).
* ``drop``  - ``base`` minus the analysers that drift most in the attack file
              (``WORST``, picked by how far their normal values move from the
              training mean, in training standard deviations).
* ``diff``  - ``base`` turned into first differences, ``x[t] - x[t-1]``, each
              sensor divided by the std of its differences on the training
              part. A slow drift becomes ~0; jumps caused by attacks stay.

Everything else (model, windows, threshold rule, clients, metrics) is the same
as ``experiments/conv_ae_centralized.py``.

Run from the repo root:

    python -m experiments.drift_variants                      # all three
    python -m experiments.drift_variants --variants base diff
    python -m experiments.drift_variants --epochs 3           # quick check

Outputs: ``experiments/drift_variants/outputs/<variant>/`` (gitignored) plus
``comparison.csv`` and ``per_client_fpr.csv`` next to them.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.conv_ae_centralized import run
from src.data_loaders.preprocessing import prepare_swat

OUT_DIR = Path("experiments/drift_variants/outputs")
VARIANTS = ("base", "drop", "diff")
# Largest shift of the normal-period mean in the attack file, in training SDs:
# AIT201 17.7, AIT504 5.3, AIT501 4.0, AIT202 2.9, AIT402 2.7, AIT502 2.0.
WORST = ("AIT201", "AIT504", "AIT501", "AIT202", "AIT402", "AIT502")


def drop_sensors(d: Mapping, sensors) -> dict:
    """Copy of ``d`` without the given sensor columns (scaling is per sensor,
    so this equals running ``prepare_swat`` with them excluded)."""
    features = list(d["features"])
    missing = [s for s in sensors if s not in features]
    if missing:
        raise ValueError(f"not in the data: {missing}")
    keep = [i for i, f in enumerate(features) if f not in sensors]
    out = dict(d)
    for k in ("train", "val", "attack"):
        out[k] = np.ascontiguousarray(d[k][:, keep])
    out["features"] = [features[i] for i in keep]
    return out


def first_differences(d: Mapping) -> dict:
    """Copy of ``d`` with ``x[t] - x[t-1]`` per sensor, scaled by the std of the
    training differences. The first row of each array is 0 (no previous second).

    Train, val and attack are differenced separately (they are separate
    recordings / splits); the client split happens afterwards, as usual.
    """
    out = dict(d)
    diffs = {k: np.diff(d[k], axis=0, prepend=d[k][:1]) for k in ("train", "val", "attack")}
    sd = diffs["train"].std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    for k, x in diffs.items():
        out[k] = (x / sd).astype(np.float32)
    out["diff_scale"] = sd
    return out


def build(variant: str, base: Mapping) -> dict:
    if variant == "base":
        return dict(base)
    if variant == "drop":
        return drop_sensors(base, WORST)
    if variant == "diff":
        return first_differences(base)
    raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--raw-dir", default="data/raw/swat")
    p.add_argument("--n-clients", type=int, default=5, choices=(3, 5))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--q", type=float, default=99.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    print("Loading and preprocessing SWaT (shared preprocessing, unchanged) ...", flush=True)
    base = prepare_swat(args.raw_dir)

    rows, fpr_cols = {}, {}
    for v in args.variants:
        d = build(v, base)
        print(f"\n===== {v}: {d['train'].shape[1]} sensors =====", flush=True)
        r = run(d, args.out / v, n_clients=args.n_clients, epochs=args.epochs,
                q=args.q, seed=args.seed, device=args.device, name=f"ConvAE {v}",
                verbose=False)
        p = r["pooled"]
        rows[v] = {
            "sensors": d["train"].shape[1],
            "raw_f1": p["raw_f1"],
            "pak_auc_f1": p["pak_auc_f1"],
            "etapr_f1": p["etapr_f1"],
            "auroc": p["auroc"],
            "auprc": p["auprc"],
            "fpr": p["fpr"],
            "attacks_detected": f"{p['n_segments_detected']}/{p['n_segments']}",
            "oracle_raw_f1": r["oracle_raw_f1"],
            "final_train_loss": r["history"][-1]["train_loss"],
        }
        fpr_cols[v] = r["per_client"]["fpr"]
        print(f"pooled raw F1 {p['raw_f1']:.3f}  FPR {p['fpr']:.3f}  "
              f"AUROC {p['auroc']:.3f}  oracle raw F1 {r['oracle_raw_f1']:.3f}", flush=True)
    rows["random"] = {"raw_f1": r["pooled_random"]["raw_f1"],
                      "pak_auc_f1": r["pooled_random"]["pak_auc_f1"],
                      "etapr_f1": r["pooled_random"]["etapr_f1"],
                      "auroc": r["pooled_random"]["auroc"],
                      "auprc": r["pooled_random"]["auprc"],
                      "fpr": r["pooled_random"]["fpr"]}

    comparison = pd.DataFrame.from_dict(rows, orient="index")
    per_client_fpr = pd.DataFrame(fpr_cols)
    comparison.to_csv(args.out / "comparison.csv")
    per_client_fpr.to_csv(args.out / "per_client_fpr.csv")
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print("\nPooled results (all clients as one test set; oracle = upper bound, "
              "threshold picked on test labels):\n", comparison)
        print("\nFalse-alarm rate per client:\n", per_client_fpr)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
