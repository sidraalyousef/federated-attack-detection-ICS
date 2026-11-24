"""Stream the test data in time order with a dynamic threshold, optionally
fine-tuning the model on recent data it judged normal.

This is the label-free version of DAICS's idea (Abdelaty et al. 2021: update
the weights when behaviour changes). DAICS fine-tunes on alarms a technician
confirms as false. Here nobody confirms anything: every ``update_every``
seconds the model is fine-tuned on that block's windows that stayed **below**
the dynamic threshold ("pseudo-normal"), mixed with an equal number of the
original training windows (replay, so it doesn't forget normal behaviour).

No look-ahead: every window is scored by the model *before* it is fine-tuned
on that window's block, and its threshold only uses earlier scores.

Two thresholds (``threshold=``):

* ``"shared"``: one score per window (MSE), alarm when it exceeds
  ``factor x`` the past median score (``rolling_threshold``).
* ``"per_sensor"``: each sensor's error divided by its own past median, the
  largest ratio is the score, alarm above the 99th percentile of the same
  score on validation (``per_sensor_dynamic_scores``). Avoids drift in a few
  sensors masking attacks on the others. ``val_points`` must then be
  per-sensor errors ``(seconds, sensors)``.

Known risk (say so in the report): if an attack is slow or long enough to
stay under the dynamic threshold, the model can learn it as normal.

    out = stream_scores(model, [c["attack"] for c in clients], val_points,
                        train_windows=make_windows(d["train"]), finetune=True)
    ratios = [ratio_to_threshold(s, t) for s, t in zip(out["scores"], out["thresholds"])]
    scores, labels, groups = points_from_clients(ratios, clients)
    evaluate_groups(scores, labels, groups, 1.0)
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from torch import nn

from src.data_loaders.preprocessing import TRAIN_STRIDE, WINDOW, make_windows
from src.eval.adaptive import (
    DAY_S,
    factor_from_normal,
    fit_sensor_bars,
    per_sensor_dynamic_scores,
    rolling_threshold,
)
from src.eval.metrics import threshold_from_normal

from .conv_ae import score_windows, sensor_errors, train_ae


def stream_scores(model: nn.Module, series: Sequence[np.ndarray], val_points: np.ndarray, *,
                  train_windows: np.ndarray | None = None, finetune: bool = False,
                  threshold: str = "shared", bar_strength: float = 0.0,
                  window: int = WINDOW, threshold_window: int = DAY_S, q: float = 99.0,
                  update_every: int = 6 * 3_600, finetune_epochs: int = 3,
                  finetune_lr: float = 1e-4, finetune_stride: int = TRAIN_STRIDE,
                  min_update: int = 200, batch_size: int = 256, seed: int = 0,
                  finetune_kwargs: dict | None = None,
                  device=None, verbose: bool = True) -> dict:
    """Score consecutive 2-D series (e.g. each client's attack data, in time order).

    Args:
        series: 2-D arrays ``(seconds, sensors)``, consecutive in time. The model
            and the threshold history carry over from one to the next.
        val_points: validation errors of the starting model, per second. For
            ``threshold="shared"``: 1-D scores; they set the factor
            (``factor_from_normal(val_points, q)``) and warm-start the rolling
            median. For ``"per_sensor"``: 2-D ``(seconds, sensors)`` errors; they
            set the fixed threshold on the ratio score and warm-start each
            sensor's rolling median.
        threshold: ``"shared"`` or ``"per_sensor"`` (see module docstring).
        finetune_kwargs: extra arguments for ``train_ae`` during fine-tuning,
            e.g. the same ``noise_std`` / ``weight_decay`` the model was trained with.
        bar_strength: per-sensor only. 0 = one shared bar for all sensors;
            > 0 divides each sensor's ratio by its own validation 99th
            percentile raised to this power (``fit_sensor_bars``), up to 1 =
            fully per-sensor bars. Higher catches more attacks but raises more
            false alarms.
        train_windows: training windows for replay. Required if ``finetune``.
        finetune: if False, only the dynamic threshold is applied (no weight updates).

    Returns ``{"scores", "thresholds"}`` (one array of window scores and one of
    window thresholds per series, aligned with ``make_windows(x, window, 1)``),
    plus ``"factor"`` and ``"updates"`` (one log row per block).
    """
    if finetune and train_windows is None:
        raise ValueError("finetune=True needs train_windows for replay")
    if threshold not in ("shared", "per_sensor"):
        raise ValueError("threshold must be 'shared' or 'per_sensor'")
    per_sensor = threshold == "per_sensor"
    rng = np.random.default_rng(seed)
    history = np.asarray(val_points, dtype=float)
    if per_sensor:
        if history.ndim != 2:
            raise ValueError("threshold='per_sensor' needs 2-D val_points (seconds, sensors)")
        history = history[~np.isnan(history).all(axis=1)]
        bars = (fit_sensor_bars(history, threshold_window, q, strength=bar_strength)
                if bar_strength > 0 else None)
        val_scores = per_sensor_dynamic_scores(history, None, threshold_window, bars=bars)
        factor = threshold_from_normal(val_scores, q=q)        # fixed threshold on the ratio score
    else:
        history = history[~np.isnan(history)]
        factor = factor_from_normal(history, q)
    scores_out, thr_out, updates = [], [], []

    for k, x in enumerate(series):
        w = make_windows(x, window, 1)
        s_all = np.empty(len(w))
        t_all = np.empty(len(w))
        for a in range(0, len(w), update_every):
            b = min(a + update_every, len(w))
            if per_sensor:                                          # model before this block's update
                e = sensor_errors(model, w[a:b], device=device)
                s = per_sensor_dynamic_scores(e, history, threshold_window, bars=bars)
                t = np.full(len(s), factor)
                history = np.vstack([history, e])[-threshold_window:]
            else:
                s = score_windows(model, w[a:b], device=device)
                t = rolling_threshold(s, history, threshold_window, factor)
                history = np.concatenate([history, s])[-threshold_window:]
            s_all[a:b], t_all[a:b] = s, t

            row = {"series": k, "start": a, "end": b, "alarm_rate": float(np.nanmean(s > t))}
            if finetune:
                calm = a + np.flatnonzero(s < t)                    # pseudo-normal windows (NaN t -> False)
                calm = calm[::finetune_stride]
                row["n_finetune"] = len(calm)
                if len(calm) >= min_update:
                    n_replay = min(len(calm), len(train_windows))
                    replay = np.sort(rng.choice(len(train_windows), size=n_replay, replace=False))
                    batch = np.concatenate([w[calm], train_windows[replay]])
                    train_ae(model, batch, epochs=finetune_epochs, batch_size=batch_size,
                             lr=finetune_lr, device=device, seed=int(rng.integers(1 << 31)),
                             verbose=False, **(finetune_kwargs or {}))
                    row["updated"] = True
                else:
                    row["updated"] = False
            updates.append(row)
            if verbose:
                extra = (f", fine-tuned on {row['n_finetune']} windows"
                         if finetune and row.get("updated") else "")
                print(f"  series {k} seconds {a:>6}-{b:<6} alarms {row['alarm_rate']:.1%}{extra}",
                      flush=True)
        scores_out.append(s_all)
        thr_out.append(t_all)
    return {"scores": scores_out, "thresholds": thr_out, "factor": factor, "updates": updates}
