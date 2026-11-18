"""SWaT 2015 preprocessing: load, clean, split, scale.

Steps (see PREPROCESSING.md for the reasons):

1. Drop the first 6 hours of the normal file (the plant is still settling).
2. Drop sensors that never change in either file, plus P201 and P403, which
   change during normal operation in the attack file and only cause false alarms.
3. Split the remaining normal data in time: first 85% train, last 15% validation.
4. Min-max scale every sensor with ranges fitted on the training part only.

The output is 2-D, one row per second. Windows are cut afterwards with
:func:`make_windows`: train with stride ``TRAIN_STRIDE`` (10), validation and
attack with stride ``EVAL_STRIDE`` (1) so every second gets a score.

    d = prepare_swat("data/raw/swat")
    d["train"], d["val"], d["attack"]   # (seconds, 43 sensors), float32, scaled
    d["attack_labels"]                  # 0/1 per second of the attack file
"""
from pathlib import Path

import numpy as np
import pandas as pd

NORMAL_FILE = "SWaT_Dataset_Normal_v1.csv"
ATTACK_FILE = "SWaT_Dataset_Attack_v0.csv"
TIME_FORMAT = "%d/%m/%Y %I:%M:%S %p"

WARMUP_S = 21_600            # first 6 h of the normal file, dropped
VAL_FRAC = 0.15              # last 15% of the (post-warm-up) normal file
EXCLUDE = ("P201", "P403")   # change outside attacks in the attack file
WINDOW = 60                  # seconds per window
TRAIN_STRIDE = 10            # stride for training windows
EVAL_STRIDE = 1              # stride for validation / test windows: never change


def load_swat(path):
    """Read one SWaT CSV: Timestamp, sensor columns, 0/1 label."""
    df = pd.read_csv(path, low_memory=False)
    if str(df.columns[0]).startswith("Unnamed"):  # banner row above the header
        df = pd.read_csv(path, header=1, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"].str.strip(), format=TIME_FORMAT)
    label = df.pop("Normal/Attack").str.replace(" ", "").str.lower()  # "A ttack" typos
    df["label"] = (label == "attack").astype(np.int8)
    return df


def constant_in_all(frames, columns):
    """Columns that never change in any of the frames."""
    return [c for c in columns if all(f[c].nunique() == 1 for f in frames)]


def split_normal(normal, warmup=WARMUP_S, val_frac=VAL_FRAC):
    """Drop the warm-up rows, then split in time into (train, val)."""
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be in (0, 1)")
    normal = normal.iloc[warmup:].reset_index(drop=True)
    n_val = round(len(normal) * val_frac)
    return normal.iloc[:-n_val].reset_index(drop=True), normal.iloc[-n_val:].reset_index(drop=True)


def fit_minmax(x):
    """Per-column min and range; a zero range becomes 1."""
    lo = x.min(axis=0)
    rng = x.max(axis=0) - lo
    return lo, np.where(rng == 0, 1.0, rng)


def apply_minmax(x, lo, rng):
    return ((x - lo) / rng).astype(np.float32)


def make_windows(x, window=WINDOW, stride=TRAIN_STRIDE):
    """(n, window, features) read-only view; n = (len(x) - window) // stride + 1.

    Use ``stride=EVAL_STRIDE`` (1) for anything you score: with a larger stride,
    ``src.eval.window_scores_to_points`` leaves most seconds without a score.
    The view shares memory with ``x``: copy a batch before handing it to torch.
    """
    view = np.lib.stride_tricks.sliding_window_view(x, window, axis=0)
    return view[::stride].transpose(0, 2, 1)


def prepare_swat(raw_dir, warmup=WARMUP_S, val_frac=VAL_FRAC, exclude=EXCLUDE):
    """Load both files, drop warm-up and unusable sensors, split, scale."""
    raw_dir = Path(raw_dir)
    normal = load_swat(raw_dir / NORMAL_FILE)
    attack = load_swat(raw_dir / ATTACK_FILE)
    train, val = split_normal(normal, warmup, val_frac)

    sensors = [c for c in normal.columns if c not in ("Timestamp", "label")]
    # Constant in the (post-warm-up) normal data and in the attack file.
    dropped_constant = constant_in_all([pd.concat([train, val]), attack], sensors)
    excluded = [c for c in exclude if c in sensors and c not in dropped_constant]
    drop = dropped_constant + excluded
    features = [c for c in sensors if c not in drop]

    lo, rng = fit_minmax(train[features].to_numpy(np.float64))

    def scale(df):
        return apply_minmax(df[features].to_numpy(np.float64), lo, rng)

    return {
        "features": features,
        "dropped": drop,
        "dropped_constant": dropped_constant,
        "excluded": excluded,
        "scaler": (lo, rng),
        "train": scale(train),
        "val": scale(val),
        "attack": scale(attack),
        "train_timestamps": train["Timestamp"].to_numpy(),
        "val_timestamps": val["Timestamp"].to_numpy(),
        "attack_timestamps": attack["Timestamp"].to_numpy(),
        "attack_labels": attack["label"].to_numpy(),
        "warmup": warmup,
        "val_frac": val_frac,
    }
