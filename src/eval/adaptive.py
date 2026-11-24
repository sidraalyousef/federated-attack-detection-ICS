"""Scoring and thresholding that cope with sensor drift, without dropping sensors.

Two ideas from the literature, both label-free:

* **Robust per-sensor normalisation** (GDN, Deng & Hooi 2021): divide each
  sensor's error by how large that sensor's error normally is (median / IQR on
  normal validation data), so one sensor cannot dominate the score, then
  combine the sensors. ``aggregate(..., how="kth", k=3)`` only rises when at
  least 3 sensors are unusual at the same time.
* **Dynamic threshold** (Telemanom, Hundman et al. 2018; DAICS, Abdelaty et al.
  2021): instead of one fixed threshold, use ``factor x median of the scores
  of the past `window` seconds``. Slow drift raises the median and the
  threshold with it; sudden jumps still cross it. ``factor`` is fixed on
  validation data with :func:`factor_from_normal` (same 99th-percentile rule
  as the fixed threshold), so no test labels are used.

**Per-sensor dynamic threshold**: with one shared score, drift in a few
sensors (e.g. the analysers) inflates the typical error of the whole window,
and an attack on other sensors then adds too little on top to cross a shared
dynamic threshold ("masking"). :func:`per_sensor_dynamic_scores` instead
divides each sensor's error by *its own* past median and takes the largest
ratio, so a drifting sensor only raises its own baseline. Threshold that score
with the usual fixed rule on validation (99th percentile).

With a dynamic threshold, evaluate ``score / threshold`` against a threshold
of 1.0 (:func:`ratio_to_threshold`), so ``evaluate`` / ``evaluate_groups``
work unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY_S = 24 * 3_600


def fit_robust(normal_errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sensor median and IQR of ``(n, sensors)`` errors on normal data.

    NaN rows (e.g. the warm-up seconds) are ignored. A zero IQR becomes a
    tiny number so constant-error sensors don't divide by zero.
    """
    e = np.asarray(normal_errors, dtype=float)
    med = np.nanmedian(e, axis=0)
    iqr = np.nanpercentile(e, 75, axis=0) - np.nanpercentile(e, 25, axis=0)
    return med, np.where(iqr > 0, iqr, 1e-12)


def robust_z(errors: np.ndarray, med: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    """``(errors - median) / IQR`` per sensor (GDN's normalisation)."""
    return (np.asarray(errors, dtype=float) - med) / iqr


def aggregate(z: np.ndarray, how: str = "kth", k: int = 3) -> np.ndarray:
    """Combine per-sensor scores ``(n, sensors)`` into one score per row.

    how="max":  the most unusual sensor (GDN).
    how="kth":  the k-th most unusual sensor, so at least k sensors must be
                unusual together. Robust to one drifting sensor.
    how="mean": the average over sensors.
    Rows that are all NaN stay NaN.
    """
    z = np.asarray(z, dtype=float)
    nan_rows = np.isnan(z).all(axis=1)
    filled = np.where(np.isnan(z), -np.inf, z)
    if how == "max":
        out = filled.max(axis=1)
    elif how == "kth":
        if not 1 <= k <= z.shape[1]:
            raise ValueError(f"k must be between 1 and {z.shape[1]}")
        out = -np.partition(-filled, k - 1, axis=1)[:, k - 1]
    elif how == "mean":
        out = np.nanmean(np.where(nan_rows[:, None], 0.0, z), axis=1)
    else:
        raise ValueError("how must be 'max', 'kth' or 'mean'")
    out = out.astype(float)
    out[nan_rows] = np.nan
    return out


def factor_from_normal(normal_scores, q: float = 99.0) -> float:
    """How far above the typical score an alarm is: ``percentile_q / median``,
    both on normal (validation) scores. Use as ``factor`` in
    :func:`rolling_threshold`."""
    s = np.asarray(normal_scores, dtype=float)
    s = s[~np.isnan(s)]
    if s.size == 0:
        raise ValueError("normal_scores is empty")
    med = np.median(s)
    if med <= 0:
        raise ValueError("median of normal scores must be positive")
    return float(np.percentile(s, q) / med)


def rolling_median(values, history=None, window: int = DAY_S,
                   min_periods: int = 600) -> np.ndarray:
    """Past-only rolling median of the last ``window`` rows, per column.

    ``values`` is 1-D (one score per second) or 2-D (seconds x sensors). Row t
    uses rows before t only (``history`` first, then earlier ``values``).
    Rows of ``history`` that are all NaN are dropped; NaNs are ignored in the
    median. Rows with fewer than ``min_periods`` past values are NaN.
    """
    v = np.asarray(values, dtype=float)
    two_d = v.ndim == 2
    v2 = v if two_d else v[:, None]
    if history is None:
        h = np.zeros((0, v2.shape[1]))
    else:
        h = np.asarray(history, dtype=float)
        h = h if h.ndim == 2 else h[:, None]
        h = h[~np.isnan(h).all(axis=1)][-window:]
    frame = pd.DataFrame(np.vstack([h, v2]))
    med = (frame.shift(1).rolling(window, min_periods=min(min_periods, window))
           .median().to_numpy()[len(h):])
    return med if two_d else med[:, 0]


def per_sensor_dynamic_scores(errors, history=None, window: int = DAY_S,
                              min_periods: int = 600, how: str = "max",
                              k: int = 1, bars: np.ndarray | None = None) -> np.ndarray:
    """Per-sensor dynamic scoring: ``aggregate(error_i / past median of error_i)``.

    ``errors`` and ``history`` are ``(seconds, sensors)``. Each sensor is
    compared with its own median over the past ``window`` seconds, then the
    sensors are combined with :func:`aggregate` (default: the largest ratio).
    ``bars`` (from :func:`fit_sensor_bars`) divides each sensor's ratio by how
    high that ratio normally goes, so quiet sensors get a tighter bar than
    noisy ones. Threshold the result with ``threshold_from_normal`` on validation scores
    computed the same way (validation compared with its own past).
    """
    e = np.asarray(errors, dtype=float)
    med = rolling_median(e, history, window, min_periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = e / np.maximum(med, 1e-12)
    ratio[np.isnan(med) | np.isnan(e)] = np.nan
    if bars is not None:
        ratio = ratio / bars
    return aggregate(ratio, how, k)


def fit_sensor_bars(normal_errors, window: int = DAY_S, q: float = 99.0,
                    min_periods: int = 600, strength: float = 1.0) -> np.ndarray:
    """Per-sensor bar: the q-th percentile of each sensor's ratio (error / own
    past median) on normal data, e.g. validation compared with its own past,
    raised to ``strength``.

    ``strength`` blends between one shared bar (0: all bars 1) and fully
    per-sensor bars (1). Validation is calmer than the attack days, so fully
    per-sensor bars are too tight for some sensors (more false alarms);
    in-between values trade attacks caught against false alarms.
    Pass the result as ``bars`` to :func:`per_sensor_dynamic_scores`."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    e = np.asarray(normal_errors, dtype=float)
    e = e[~np.isnan(e).all(axis=1)]
    med = rolling_median(e, None, window, min_periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = e / np.maximum(med, 1e-12)
    ratio[np.isnan(med)] = np.nan
    bars = np.nanpercentile(ratio, q, axis=0)
    return np.where(np.isfinite(bars) & (bars > 0), bars, 1.0) ** strength


def rolling_threshold(scores, history=None, window: int = DAY_S, factor: float = 1.0,
                      min_periods: int = 600) -> np.ndarray:
    """Dynamic threshold: ``factor x median`` of the scores of the past ``window`` seconds.

    Past-only: the threshold at second t uses seconds before t, never t itself
    or anything after it. ``history`` (e.g. validation scores, or earlier
    blocks of the stream) is prepended so the first seconds have a baseline.
    NaN scores are ignored in the median. Seconds with fewer than
    ``min_periods`` past scores get NaN.
    """
    return factor * rolling_median(scores, history, window, min_periods)


def ratio_to_threshold(scores, thresholds) -> np.ndarray:
    """``scores / thresholds``: evaluate this against a threshold of 1.0.

    NaN where either is NaN.
    """
    s = np.asarray(scores, dtype=float)
    t = np.asarray(thresholds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = s / t
    out[np.isnan(s) | np.isnan(t) | (t <= 0)] = np.nan
    return out
