"""Evaluation metrics for time-series anomaly / attack detection.

Every experiment in the project (centralized, local-only, FedAvg, FedProx,
WADI, HAI) should score its models through :func:`evaluate`, so all result
tables use the same protocols and threshold rules.

Three F1 protocols are reported (Kim et al., "Towards a Rigorous Evaluation of
Time-Series Anomaly Detection", AAAI-22):

* raw   - plain point-wise F1, no adjustment.
* PA    - point adjustment: if any point of a true attack segment is flagged,
          the whole segment counts as detected. Common in the literature and
          known to inflate scores; reported only for comparability to literature.
* PA%K  - a segment gets the PA credit only if more than K% of its points are
          flagged. K=0 equals PA, K=100 equals raw. We sweep K and report the
          area under the F1-vs-K curve (normalised to [0, 1]).

On top of these, eTaPR (enhanced time-series aware precision/recall, Hwang et
al. 2022; the metric used by the HAI dataset authors) is reported via the
``faster-etapr`` package. It scores whole attacks and whole alarms rather than
single timesteps. See :func:`etapr_scores`.

Conventions
-----------
* ``labels``: 1-D array of 0/1 per timestep (1 = attack).
* ``scores``: 1-D array of anomaly scores per timestep, higher = more anomalous.
* ``preds``:  1-D array of 0/1 predictions, ``scores > threshold``.
* Window-level scores from the autoencoders must first be mapped to timesteps
  with :func:`window_scores_to_points` (score assigned to the window's last
  point by default).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# for PA%K:Ks between 0 and 100, incremented by 10 
DEFAULT_KS: tuple[int, ...] = tuple(range(0, 101, 10))
# eTaPR defaults (Hwang et al.): alarm must be >=50% correct, attack >=10% covered
DEFAULT_THETA_P: float = 0.5
DEFAULT_THETA_R: float = 0.1


# --------------------------------------------------------------------------- 
# Basic helpers
# --------------------------------------------------------------------------- 
def _as_binary(x: Sequence, name: str) -> np.ndarray:
    """Convert ``x`` to a flat 0/1 integer array, or raise ValueError if it isn't binary."""
    arr = np.asarray(x).astype(int).ravel()
    if arr.size and not np.isin(arr, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0/1 values")
    return arr


def get_segments(labels: Sequence[int]) -> list[tuple[int, int]]:
    """Return attack segments as ``(start, end)`` pairs, ``end`` exclusive.

    A segment is a run of consecutive 1s in ``labels``.
    """
    y = _as_binary(labels, "labels")
    padded = np.concatenate(([0], y, [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def as_dict(self) -> dict:
        return asdict(self)


def prf(preds: Sequence[int], labels: Sequence[int]) -> PRF:
    """Point-wise precision / recall / F1 (0.0 where undefined)."""
    p = _as_binary(preds, "preds")
    y = _as_binary(labels, "labels")
    if p.shape != y.shape:
        raise ValueError(f"preds {p.shape} and labels {y.shape} differ in length")
    tp = int(np.sum((p == 1) & (y == 1)))
    fp = int(np.sum((p == 1) & (y == 0)))
    fn = int(np.sum((p == 0) & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(precision, recall, f1, tp, fp, fn)


# --------------------------------------------------------------------------- 
# Point adjustment (PA and PA%K)
# --------------------------------------------------------------------------- 
def point_adjust(preds: Sequence[int], labels: Sequence[int], k: float = 0.0) -> np.ndarray:
    """Apply PA%K adjustment and return the adjusted predictions.

    For each true attack segment, if the fraction of its points flagged is
    strictly greater than ``k / 100``, every point in the segment is set to 1.
    Predictions outside attack segments are never changed, so false positives
    are untouched.

    ``k=0``   -> classic point adjustment (one hit is enough).
    ``k=100`` -> no adjustment (identical to raw predictions).
    """
    if not 0 <= k <= 100:
        raise ValueError("k must be in [0, 100]")
    p = _as_binary(preds, "preds").copy()
    y = _as_binary(labels, "labels")
    if p.shape != y.shape:
        raise ValueError(f"preds {p.shape} and labels {y.shape} differ in length")
    for start, end in get_segments(y):
        hit_ratio = p[start:end].mean()
        if hit_ratio > k / 100.0:
            p[start:end] = 1
    return p


def pak_curve(
    preds: Sequence[int], labels: Sequence[int], ks: Iterable[float] = DEFAULT_KS
) -> dict[float, PRF]:
    """PA%K precision/recall/F1 for each K in ``ks``."""
    return {k: prf(point_adjust(preds, labels, k), labels) for k in ks}


def pak_auc(curve: dict[float, PRF], metric: str = "f1") -> float:
    """Area under the metric-vs-K curve, K rescaled to [0, 1] (trapezoidal)."""
    ks = np.array(sorted(curve))
    if ks.size < 2:
        raise ValueError("need at least two K values for an AUC")
    vals = np.array([getattr(curve[k], metric) for k in ks])
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    return float(np.trapezoid(vals, x)) if hasattr(np, "trapezoid") else float(np.trapz(vals, x))


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
def threshold_from_normal(
    normal_scores: Sequence[float], method: str = "percentile", q: float = 99.0, n_std: float = 3.0
) -> float:
    """Threshold fitted on scores of *normal* data only (no test labels used).

    This is the default, label-free rule for the project. Use scores from a
    held-out slice of the normal training data (validation), not the data the
    model was trained on, if possible.

    method="percentile": the q-th percentile of normal scores.
    method="std":        mean + n_std * std of normal scores.
    """
    s = np.asarray(normal_scores, dtype=float)
    s = s[~np.isnan(s)]
    if s.size == 0:
        raise ValueError("normal_scores is empty")
    if method == "percentile":
        return float(np.percentile(s, q))
    if method == "std":
        return float(s.mean() + n_std * s.std())
    raise ValueError("method must be 'percentile' or 'std'")


def best_f1_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    protocol: str = "raw",
    k: float = 0.0,
    n_candidates: int = 200,
) -> tuple[float, float]:
    """ORACLE threshold that maximises F1 on the given labels.

    Uses the test labels, so results are optimistic. Report it only as an
    upper bound, clearly labelled, never as the headline number.

    protocol: "raw" or "pa" (PA%K with the given ``k``).
    Returns ``(threshold, best_f1)``.
    """
    s = np.asarray(scores, dtype=float)
    y = _as_binary(labels, "labels")
    mask = ~np.isnan(s)
    s, y = s[mask], y[mask]
    candidates = np.unique(np.quantile(s, np.linspace(0.0, 1.0, n_candidates)))
    best_t, best_f1 = float(candidates[0]), -1.0
    for t in candidates:
        p = (s > t).astype(int)
        if protocol == "pa":
            p = point_adjust(p, y, k)
        elif protocol != "raw":
            raise ValueError("protocol must be 'raw' or 'pa'")
        f1 = prf(p, y).f1
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Window -> timestep alignment
# ---------------------------------------------------------------------------
def window_scores_to_points(
    window_scores: Sequence[float],
    n_points: int,
    window_size: int,
    stride: int = 1,
    mode: str = "last",
) -> np.ndarray:
    """Map one score per window to one score per timestep.

    Window ``i`` covers timesteps ``[i*stride, i*stride + window_size)``.

    mode="last": each window's score goes to its last timestep (the point a
                 forecaster/reconstructor "explains" in real time). Timesteps
                 with no window ending on them are NaN - drop them with
                 :func:`drop_nan` before evaluating, on scores *and* labels.
                 With stride=1 only the first ``window_size - 1`` are NaN.
    mode="max":  each timestep gets the max score of all windows covering it.
    mode="mean": each timestep gets the mean score of all windows covering it.
    """
    w = np.asarray(window_scores, dtype=float).ravel()
    expected = (n_points - window_size) // stride + 1
    if w.size != expected:
        raise ValueError(
            f"got {w.size} window scores, expected {expected} for "
            f"n_points={n_points}, window_size={window_size}, stride={stride}"
        )
    out = np.full(n_points, np.nan)
    starts = np.arange(w.size) * stride
    if mode == "last":
        out[starts + window_size - 1] = w
        return out
    if mode not in ("max", "mean"):
        raise ValueError("mode must be 'last', 'max' or 'mean'")
    acc = np.full(n_points, -np.inf) if mode == "max" else np.zeros(n_points)
    cnt = np.zeros(n_points)
    for st, val in zip(starts, w):
        sl = slice(st, st + window_size)
        if mode == "max":
            acc[sl] = np.maximum(acc[sl], val)
        else:
            acc[sl] += val
        cnt[sl] += 1
    covered = cnt > 0
    out[covered] = acc[covered] if mode == "max" else acc[covered] / cnt[covered]
    return out


def drop_nan(scores: Sequence[float], *arrays: Sequence) -> tuple[np.ndarray, ...]:
    """Drop timesteps where ``scores`` is NaN from scores and every other array."""
    s = np.asarray(scores, dtype=float)
    keep = ~np.isnan(s)
    return (s[keep],) + tuple(np.asarray(a)[keep] for a in arrays)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def random_scores(n: int, seed: int = 0) -> np.ndarray:
    """Uniform random anomaly scores - the sanity-check baseline of Kim et al.

    Any real model should clearly beat this under raw F1 and PA%K AUC. Under
    plain PA it often looks deceptively strong, which is the point.
    """
    return np.random.default_rng(seed).random(n)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def etapr_scores(
    preds: Sequence[int],
    labels: Sequence[int],
    theta_p: float = DEFAULT_THETA_P,
    theta_r: float = DEFAULT_THETA_R,
) -> dict[str, float]:
    """Enhanced time-series aware precision/recall (eTaPR), Hwang et al. 2022.

    The metric used by the HAI dataset authors. Unlike point-wise F1 it scores
    whole attacks and whole alarms:

    * eTaR (recall): was each attack detected (enough of it covered,
      ``theta_r``), and what share of it was covered?
    * eTaP (precision): does each predicted alarm overlap a real attack
      (``theta_p``), and what share of the alarm is correct?

    Thin wrapper around the ``faster-etapr`` package (a tested, faster
    re-implementation of the authors' original code). The import is done here,
    not at module level, so the rest of this module works without it and its
    deprecation warnings stay out of the way.

    Parameters
    ----------
    preds, labels : 1-D 0/1 arrays of equal length.
    theta_p : minimum share of an alarm that must overlap an attack for the
        alarm to count as correct. Default 0.5 (authors' default).
    theta_r : minimum share of an attack that must be covered for the attack
        to count as detected. Default 0.1 (authors' default).

    Returns
    -------
    dict with ``etapr_precision``, ``etapr_recall``, ``etapr_f1`` and
    ``etapr_detected_attacks`` (number of attacks counted as detected).
    """
    p = _as_binary(preds, "preds")
    y = _as_binary(labels, "labels")
    if p.shape != y.shape:
        raise ValueError(f"preds {p.shape} and labels {y.shape} differ in length")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from faster_etapr import evaluate_from_preds

        res = evaluate_from_preds(p, y, theta_p=theta_p, theta_r=theta_r)
    return {
        "etapr_precision": float(res["eta/precision"]),
        "etapr_recall": float(res["eta/recall"]),
        "etapr_f1": float(res["eta/f1"]),
        "etapr_detected_attacks": int(res["eta/detected_anomalies"]),
    }


def evaluate(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
    ks: Iterable[float] = DEFAULT_KS,
    include_etapr: bool = True,
    theta_p: float = DEFAULT_THETA_P,
    theta_r: float = DEFAULT_THETA_R,
) -> dict:
    """Score one model on one test set with all protocols.

    Parameters
    ----------
    scores : per-timestep anomaly scores (NaNs are dropped together with labels).
    labels : per-timestep 0/1 ground truth.
    threshold : decision threshold; ``preds = scores > threshold``. Fit it with
        :func:`threshold_from_normal` unless you are explicitly reporting an
        oracle upper bound.
    ks : K values (in %) for the PA%K sweep.
    include_etapr : also compute eTaPR (see :func:`etapr_scores`) on the same
        predictions. Needs the ``faster-etapr`` package.
    theta_p, theta_r : eTaPR parameters; keep the defaults for all reported
        results so numbers stay comparable.

    Returns a flat dict, convenient for building result tables
    (plus etapr_precision, etapr_recall, etapr_f1, etapr_detected_attacks
    when ``include_etapr`` is True):
        raw_precision, raw_recall, raw_f1,
        pa_precision,  pa_recall,  pa_f1,
        pak_auc_f1, pak_f1 (dict K -> F1),
        auroc, auprc (threshold-free; NaN if only one class present),
        threshold, n_points, n_attack_points, n_segments, n_segments_detected
    """
    s, y = drop_nan(scores, labels)
    y = _as_binary(y, "labels")
    preds = (s > threshold).astype(int)

    raw = prf(preds, y)
    pa = prf(point_adjust(preds, y, 0), y)
    curve = pak_curve(preds, y, ks)
    segments = get_segments(y)
    detected = sum(int(preds[a:b].any()) for a, b in segments)

    both_classes = 0 < y.sum() < y.size
    result = {
        "raw_precision": raw.precision,
        "raw_recall": raw.recall,
        "raw_f1": raw.f1,
        "pa_precision": pa.precision,
        "pa_recall": pa.recall,
        "pa_f1": pa.f1,
        "pak_auc_f1": pak_auc(curve, "f1"),
        "pak_f1": {k: v.f1 for k, v in curve.items()},
        "auroc": float(roc_auc_score(y, s)) if both_classes else float("nan"),
        "auprc": float(average_precision_score(y, s)) if both_classes else float("nan"),
        "threshold": float(threshold),
        "n_points": int(y.size),
        "n_attack_points": int(y.sum()),
        "n_segments": len(segments),
        "n_segments_detected": detected,
    }
    if include_etapr:
        result.update(etapr_scores(preds, y, theta_p, theta_r))
    return result


def evaluate_groups(
    scores: Sequence[float],
    labels: Sequence[int],
    groups: Sequence,
    threshold: float | dict,
    ks: Iterable[float] = DEFAULT_KS,
    **evaluate_kwargs,
) -> dict:
    """Run :func:`evaluate` separately for each group (e.g. each FL client).

    Extra keyword arguments (``include_etapr``, ``theta_p``, ``theta_r``) are
    passed on to :func:`evaluate`.

    ``threshold`` is either one global float or a dict mapping group -> threshold
    (per-client thresholds). Note: segments are recomputed inside each group,
    so pass groups whose timesteps are contiguous in time.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels)
    g = np.asarray(groups)
    ks = tuple(ks)
    out = {}
    for grp in np.unique(g):
        m = g == grp
        t = threshold[grp] if isinstance(threshold, dict) else threshold
        out[grp.item() if hasattr(grp, "item") else grp] = evaluate(s[m], y[m], t, ks, **evaluate_kwargs)
    return out


def summary_row(result: dict, name: str = "") -> str:
    """One-line human-readable summary of an :func:`evaluate` result."""
    etapr = f"  eTaPR F1={result['etapr_f1']:.3f}" if "etapr_f1" in result else ""
    return (
        f"{name:<20} raw F1={result['raw_f1']:.3f}  PA F1={result['pa_f1']:.3f}  "
        f"PA%K-AUC={result['pak_auc_f1']:.3f}{etapr}  AUROC={result['auroc']:.3f}  "
        f"segments {result['n_segments_detected']}/{result['n_segments']}"
    )
