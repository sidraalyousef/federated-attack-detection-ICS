"""Does input standardization or regularization help the Conv-AE? All 43 sensors.

Each variant trains a ConvAE from scratch (same seed, 20 epochs) and is then
evaluated with the current best pipeline, ``psb_online`` (per-sensor dynamic
threshold, bar strength 0.25, online fine-tuning every 6 h). Fine-tuning uses
the same regularization as the initial training. Variants:

``baseline``       min-max inputs (shared preprocessing), no regularization.
``zscore``         inputs standardized per sensor with the training mean / std
                   (built in memory; the shared preprocessing is unchanged).
``weight_decay``   AdamW with weight decay 1e-3.
``dropout``        Dropout(0.1) after every ReLU.
``noise_*``        denoising AE: Gaussian noise with std 0.01 / 0.05 / 0.1 / 0.2
                   added to the input (inputs are min-max scaled, so 0.05 = 5%
                   of a sensor's training range).
``mask_*``         denoising AE: each value set to 0 with prob. 0.1 / 0.25.
``sensormask_*``   denoising AE: a sensor's whole window set to 0 with prob.
                   0.1 / 0.25, so the model learns to rebuild a sensor from the
                   others.

In every denoising variant the loss compares the reconstruction with the
CLEAN window; at test time inputs are never corrupted.

Run from the repo root:

    python -m experiments.conv_ae_regularization                       # all (~75 min on CPU)
    python -m experiments.conv_ae_regularization --variants baseline noise_0.05
    python -m experiments.conv_ae_regularization --epochs 3 --variants dropout   # quick check

Results are appended to ``experiments/conv_ae_regularization/outputs/results.csv``
after every variant, so a stopped run keeps what it finished.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data_loaders.clients import points_from_clients, split_clients
from src.data_loaders.preprocessing import EVAL_STRIDE, WINDOW, make_windows, prepare_swat
from src.eval import evaluate_groups
from src.eval.adaptive import ratio_to_threshold
from src.eval.federated import client_table, pooled_evaluate
from src.models import ConvAE, sensor_errors, train_ae
from src.models.online import stream_scores

OUT_DIR = Path("experiments/conv_ae_regularization/outputs")

# name -> (zscore inputs, ConvAE kwargs, train_ae regularization kwargs)
VARIANTS: dict[str, tuple[bool, dict, dict]] = {
    "baseline": (False, {}, {}),
    "zscore": (True, {}, {}),
    "weight_decay": (False, {}, {"weight_decay": 1e-3}),
    "dropout": (False, {"dropout": 0.1}, {}),
    "noise_0.01": (False, {}, {"noise_std": 0.01}),
    "noise_0.05": (False, {}, {"noise_std": 0.05}),
    "noise_0.1": (False, {}, {"noise_std": 0.1}),
    "noise_0.2": (False, {}, {"noise_std": 0.2}),
    "mask_0.1": (False, {}, {"mask_prob": 0.1}),
    "mask_0.25": (False, {}, {"mask_prob": 0.25}),
    "sensormask_0.1": (False, {}, {"sensor_mask_prob": 0.1}),
    "sensormask_0.25": (False, {}, {"sensor_mask_prob": 0.25}),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--raw-dir", default="data/raw/swat")
    p.add_argument("--n-clients", type=int, default=5, choices=(3, 5))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bar-strength", type=float, default=0.25)
    p.add_argument("--q", type=float, default=99.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    return p.parse_args()


def zscore(d: dict) -> dict:
    """Copy of ``d`` with every sensor standardized by its training mean / std."""
    mu = d["train"].mean(axis=0)
    sd = d["train"].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    out = dict(d)
    for k in ("train", "val", "attack"):
        out[k] = ((d[k] - mu) / sd).astype(np.float32)
    return out


def run_variant(name: str, d: dict, clients: list, args) -> dict:
    use_z, model_kw, reg_kw = VARIANTS[name]
    if use_z:
        d = zscore(d)
        clients = split_clients(d, n_clients=args.n_clients)
    t0 = time.perf_counter()
    train_w = make_windows(d["train"])
    val_w = make_windows(d["val"], stride=EVAL_STRIDE)

    torch.manual_seed(args.seed)
    model = ConvAE(n_features=d["train"].shape[1], window=WINDOW, **model_kw)
    hist = train_ae(model, train_w, epochs=args.epochs, val_windows=val_w[::10],
                    device=args.device, seed=args.seed, verbose=False, **reg_kw)
    (args.out / name).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / name / "conv_ae.pt")

    val_err = np.full((len(d["val"]), d["train"].shape[1]), np.nan)
    val_err[WINDOW - 1:] = sensor_errors(model, val_w, device=args.device)
    out = stream_scores(model, [c["attack"] for c in clients], val_err,
                        threshold="per_sensor", bar_strength=args.bar_strength,
                        train_windows=train_w, finetune=True, q=args.q, seed=args.seed,
                        device=args.device, verbose=False, finetune_kwargs=reg_kw)
    ratios = [ratio_to_threshold(s, t) for s, t in zip(out["scores"], out["thresholds"])]
    scores, labels, groups = points_from_clients(ratios, clients)
    per_client = evaluate_groups(scores, labels, groups, 1.0)
    r = pooled_evaluate(scores, labels, groups, 1.0)
    client_table(per_client).to_csv(args.out / name / "per_client.csv")
    np.savez_compressed(args.out / name / "scores.npz", scores=scores, labels=labels, groups=groups)
    fpr = client_table(per_client)["fpr"]
    return {
        "variant": name,
        "final_train_loss": hist[-1]["train_loss"], "final_val_loss": hist[-1]["val_loss"],
        "precision": r["raw_precision"], "recall": r["raw_recall"], "raw_f1": r["raw_f1"],
        "pak_auc_f1": r["pak_auc_f1"], "etapr_f1": r["etapr_f1"], "auroc": r["auroc"],
        "fpr": r["fpr"], "attacks": f"{r['n_segments_detected']}/{r['n_segments']}",
        "max_client_fpr": float(fpr.max()),
        "minutes": round((time.perf_counter() - t0) / 60, 1),
    }


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print("Loading and preprocessing SWaT (all sensors) ...", flush=True)
    d = prepare_swat(args.raw_dir)
    clients = split_clients(d, n_clients=args.n_clients)
    results_path = args.out / "results.csv"
    for name in args.variants:
        print(f"\n=== {name} ===", flush=True)
        row = run_variant(name, d, clients, args)
        print(f"{name}: precision {row['precision']:.3f}  recall {row['recall']:.3f}  "
              f"raw F1 {row['raw_f1']:.3f}  eTaPR {row['etapr_f1']:.3f}  FPR {row['fpr']:.3f}  "
              f"attacks {row['attacks']}  ({row['minutes']} min)", flush=True)
        pd.DataFrame([row]).to_csv(results_path, mode="a", index=False,
                                   header=not results_path.exists())
    res = pd.read_csv(results_path).drop_duplicates("variant", keep="last").set_index("variant")
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print(f"\nAll results (psb_online, bar strength {args.bar_strength:.2f}):\n", res)
    (args.out / "config.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, indent=2))


if __name__ == "__main__":
    main()
