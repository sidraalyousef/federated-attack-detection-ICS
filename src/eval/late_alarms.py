"""Secondary analysis: give credit for alarms shortly after an attack ends.

The labels end when the attacker stops, but the plant can take minutes to
show the effect, and alarms in that period are not really false. Some papers
(e.g. Kim et al. 2019, seq2seq on SWaT) count an alert within 15 minutes after
an attack as a detection. This is an evaluation-protocol choice, so report it
only as a secondary table next to the headline numbers, with the grace period
stated.

Rules (``grace`` seconds, default 900 = 15 min):

* an attack counts as caught if any alarm falls inside it or within ``grace``
  seconds after it ends, but never past the start of the next attack (so the
  next attack's alarm is not credited to this one);
* normal seconds inside those grace periods are left out of every other metric
  (they count neither as false alarms nor as correct silence);
* all other metrics come from :func:`src.eval.evaluate` on the remaining seconds.
"""
from __future__ import annotations

import numpy as np

from .metrics import evaluate, get_segments

GRACE_S = 900


def grace_periods(labels, grace: int = GRACE_S) -> list[tuple[int, int]]:
    """``(start, end)`` of the grace period after each attack, end exclusive.

    Each period starts where the attack ends and stops after ``grace`` seconds,
    at the next attack's start, or at the end of the data, whichever is first.
    """
    y = np.asarray(labels).astype(int)
    segs = get_segments(y)
    out = []
    for k, (_, e) in enumerate(segs):
        nxt = segs[k + 1][0] if k + 1 < len(segs) else len(y)
        out.append((e, min(e + grace, nxt)))
    return out


def evaluate_with_grace(scores, labels, threshold: float, grace: int = GRACE_S,
                        **evaluate_kwargs) -> dict:
    """:func:`evaluate` with a late-alarm grace period (see module docstring).

    Adds ``attacks_caught_grace`` (attacks with an alarm inside them or in
    their grace period), ``n_attacks`` and ``grace_s`` to the usual result.
    NaN scores never count as alarms.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels).astype(int)
    if s.shape != y.shape:
        raise ValueError("scores and labels must have the same shape")
    periods = grace_periods(y, grace)
    in_grace = np.zeros(len(y), dtype=bool)
    for a, b in periods:
        in_grace[a:b] = True
    keep = ~in_grace
    result = evaluate(s[keep], y[keep], threshold, **evaluate_kwargs)
    alarm = np.where(np.isnan(s), False, s > threshold)
    segs = get_segments(y)
    caught = sum(bool(alarm[a:pe].any()) for (a, _), (_, pe) in zip(segs, periods))
    result.update({"attacks_caught_grace": int(caught), "n_attacks": len(segs),
                   "grace_s": int(grace)})
    return result
