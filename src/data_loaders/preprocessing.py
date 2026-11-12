"""SWaT 2015 preprocessing: load, drop constant sensors, scale, window.

Settings: window 60 s, stride 10 s, min-max scaling fitted on normal data only.
Windows are not labelled here; labels stay per second (see src.eval).

    data = prepare_swat("data/raw/swat")
    data["normal_windows"]   # (49495, 60, 45), train on this
    data["attack_windows"]   # (44986, 60, 45), test on this
    data["attack_labels"]    # 0/1 per second
"""
from pathlib import Path

import numpy as np
import pandas as pd

NORMAL_FILE = "SWaT_Dataset_Normal_v1.csv"
ATTACK_FILE = "SWaT_Dataset_Attack_v0.csv"
TIME_FORMAT = "%d/%m/%Y %I:%M:%S %p"


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


def fit_minmax(x):
    """Per-column min and range; a zero range becomes 1."""
    lo = x.min(axis=0)
    rng = x.max(axis=0) - lo
    return lo, np.where(rng == 0, 1.0, rng)


def apply_minmax(x, lo, rng):
    return ((x - lo) / rng).astype(np.float32)


def make_windows(x, window=60, stride=10):
    """(n, window, features) read-only view; n = (len(x) - window) // stride + 1."""
    # The attack file has an 82 s gap. Its windows are kept so n matches src.eval.
    view = np.lib.stride_tricks.sliding_window_view(x, window, axis=0)
    return view[::stride].transpose(0, 2, 1)


def prepare_swat(raw_dir, window=60, stride=10):
    """Load both files, drop sensors constant in both, scale on normal, window."""
    raw_dir = Path(raw_dir)
    normal = load_swat(raw_dir / NORMAL_FILE)
    attack = load_swat(raw_dir / ATTACK_FILE)
    sensors = [c for c in normal.columns if c not in ("Timestamp", "label")]
    # Sensors constant in normal but moving in the attack file are kept.
    drop = constant_in_all([normal, attack], sensors)
    features = [c for c in sensors if c not in drop]

    lo, rng = fit_minmax(normal[features].to_numpy(np.float64))
    xn = apply_minmax(normal[features].to_numpy(np.float64), lo, rng)
    xa = apply_minmax(attack[features].to_numpy(np.float64), lo, rng)
    return {
        "features": features,
        "dropped": drop,
        "scaler": (lo, rng),
        "normal_windows": make_windows(xn, window, stride),
        "attack_windows": make_windows(xa, window, stride),
        "normal_timestamps": normal["Timestamp"].to_numpy(),
        "attack_timestamps": attack["Timestamp"].to_numpy(),
        "attack_labels": attack["label"].to_numpy(),
        "window": window,
        "stride": stride,
    }
