"""Evaluation helpers for the by-stage (vertical) client split.

In the by-stage split (``split_clients_by_stage``) every client covers the
*same* seconds and only its own sensors, so the time-split helpers
(``points_from_clients``, ``evaluate_groups``, ``pooled_evaluate``) do not fit:
they assume each client is a separate stretch of time. Pooling stage clients
would count every attack second once per client.

Nothing here computes a new metric. Every number comes from
:func:`src.eval.metrics.evaluate`. Three views of the same stage scores:

* :func:`evaluate_stages`      - each stage alone vs the plant-wide labels.
  Shows how blind an isolated stage owner is to attacks elsewhere in the
  plant; low recall is expected.
* :func:`evaluate_own_attacks` - each stage only on the attacks aimed at its
  own stage. Shows whether a stage model does its own job well.
* :func:`evaluate_any_alarm`   - the plant raises an alarm when *any* stage is
  above its own threshold. Only alarms are shared, never data or models, so
  this is still local-only. This is the plant-wide number to put next to the
  centralized and federated models.

Typical use::

    clients = split_clients_by_stage(d)                    # S1 .. S4, S5+6
    test = points_from_stages(test_window_scores, clients)             # {name: per-second}
    val  = points_from_stages(val_window_scores,  clients, part="val")
    thr  = {k: threshold_from_normal(v) for k, v in val.items()}
    labels = clients[0]["attack_labels"]

    per_stage = evaluate_stages(test, labels, thr)          # -> client_table(per_stage)
    plant     = evaluate_any_alarm(test, labels, thr)

    # Centralized model: score it once on the full attack file.
    central = evaluate(central_points, labels, central_thr)
    client_table({"centralized": central, "local any-alarm": plant})  # one row each

All stage score arrays must be aligned second by second (same length, same
windows). That holds when every stage is windowed with ``client_windows``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np

from .metrics import evaluate, get_segments, window_scores_to_points

_STAGE_NAME = re.compile(r"^S(\d+(?:\+\d+)*)$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def points_from_stages(
    window_scores: Sequence | Mapping,
    clients: Sequence[Mapping],
    window: int = 60,
    stride: int = 1,
    part: str = "attack",
) -> dict[str, np.ndarray]:
    """Per-client window scores -> ``{client name: per-second scores}``.

    ``window_scores`` is a list in the order of ``clients`` or a dict keyed by
    client name, holding the scores of ``client_windows(c)[part]``. The first
    ``window - 1`` seconds are NaN (no window ends there) and are dropped later
    by :func:`evaluate`.
    """
    if isinstance(window_scores, Mapping):
        window_scores = [window_scores[c["name"]] for c in clients]
    if len(window_scores) != len(clients):
        raise ValueError("need one array of window scores per client")
    lengths = {len(c[part]) for c in clients}
    if len(lengths) != 1:
        raise ValueError(
            f"stage clients must cover the same seconds, got lengths {sorted(lengths)}"
        )
    n = lengths.pop()
    return {
        c["name"]: window_scores_to_points(ws, n, window, stride)
        for ws, c in zip(window_scores, clients)
    }


def _aligned(stage_scores: Mapping, labels: Sequence | None = None) -> dict[str, np.ndarray]:
    """Check that all stage score arrays (and the labels) have the same length."""
    if not stage_scores:
        raise ValueError("stage_scores is empty")
    out = {k: np.asarray(v, dtype=float).ravel() for k, v in stage_scores.items()}
    lengths = {v.size for v in out.values()}
    if labels is not None:
        lengths.add(np.asarray(labels).size)
    if len(lengths) != 1:
        raise ValueError(
            "stage scores (and labels) must be aligned second by second, "
            f"got lengths {sorted(lengths)}"
        )
    return out


def _threshold(thresholds: float | Mapping, name) -> float:
    if isinstance(thresholds, Mapping):
        if name not in thresholds:
            raise KeyError(f"no threshold for stage client {name!r}")
        return float(thresholds[name])
    return float(thresholds)


def stages_of(name: str) -> tuple[int, ...]:
    """Stages of a client from its name: ``"S1"`` -> (1,), ``"S5+6"`` -> (5, 6)."""
    m = _STAGE_NAME.match(str(name))
    if m is None:
        raise ValueError(f"cannot read stages from client name {name!r}; pass client_stages")
    return tuple(int(s) for s in m.group(1).split("+"))


# ---------------------------------------------------------------------------
# View 1: each stage vs the plant-wide labels
# ---------------------------------------------------------------------------
def evaluate_stages(
    stage_scores: Mapping,
    labels: Sequence[int],
    thresholds: float | Mapping,
    **evaluate_kwargs,
) -> dict:
    """Run :func:`evaluate` for each stage client against the plant-wide labels.

    Returns ``{client: result}``, ready for ``client_table`` and
    ``client_summary`` in :mod:`src.eval.federated`. Do not use
    ``compare_settings`` on it to compare with the time split: a mean over
    stages is not a plant-wide score. Use :func:`evaluate_any_alarm` for that.
    """
    scores = _aligned(stage_scores, labels)
    return {
        name: evaluate(s, labels, _threshold(thresholds, name), **evaluate_kwargs)
        for name, s in scores.items()
    }


# ---------------------------------------------------------------------------
# View 2: each stage only on the attacks aimed at it
# ---------------------------------------------------------------------------
def evaluate_own_attacks(
    stage_scores: Mapping,
    labels: Sequence[int],
    thresholds: float | Mapping,
    attack_stages: Sequence[Sequence[int]],
    client_stages: Mapping | None = None,
    **evaluate_kwargs,
) -> dict:
    """Score each stage client only on the attacks that target its own stages.

    ``attack_stages[i]`` lists the stage(s) targeted by the i-th attack segment
    *in the labels* (the order of ``get_segments(labels)``), e.g.
    ``[(1,), (1,), (3,), (1, 3), ...]``. Build it once by matching each labelled
    segment's start time to the official SWaT attack list. Note that the number
    of labelled segments need not equal the number of attacks in that list.
    Use an empty tuple for a segment whose target stage is unknown: it is then
    left out for every client.

    For each client, the seconds of attacks on *other* stages are removed
    before scoring, so they count neither as misses nor as false alarms. Its own
    attacks keep label 1 and every normal second stays in. A client with no
    attacks of its own gets NaN F1 (judge it by ``n_false_alarms`` / ``fpr``),
    as in the time split.

    ``client_stages`` maps client name -> stages; by default it is read from
    the names (``"S5+6"`` -> (5, 6)). Each result also has ``n_own_attacks``.
    """
    scores = _aligned(stage_scores, labels)
    y = np.asarray(labels).astype(int).ravel()
    segments = get_segments(y)
    if len(attack_stages) != len(segments):
        raise ValueError(
            f"attack_stages has {len(attack_stages)} entries but the labels have "
            f"{len(segments)} attack segments; give one entry per labelled segment"
        )
    targets = [{int(s) for s in t} for t in attack_stages]

    out = {}
    for name, s in scores.items():
        own = set(client_stages[name] if client_stages is not None else stages_of(name))
        s = s.copy()
        n_own = 0
        for (a, b), tgt in zip(segments, targets):
            if tgt & own:
                n_own += 1
            else:
                s[a:b] = np.nan  # someone else's attack: leave it out
        res = evaluate(s, y, _threshold(thresholds, name), **evaluate_kwargs)
        res["n_own_attacks"] = n_own
        out[name] = res
    return out


# ---------------------------------------------------------------------------
# View 3: plant-wide alarm = any stage above its own threshold
# ---------------------------------------------------------------------------
def any_alarm_scores(
    stage_scores: Mapping,
    thresholds: float | Mapping,
    mode: str = "ratio",
) -> np.ndarray:
    """Combine stage scores into one plant-wide score, with the decision at 0.

    Each stage's score is first turned into a margin over its own threshold,
    then the plant-wide score is the largest margin. So ``combined > 0``
    exactly when at least one stage is above its threshold.

    mode="ratio": margin = score / threshold - 1 (default). Puts stages on a
                  comparable scale, which matters for AUROC/AUPRC: stages
                  with more sensors or noisier sensors have larger raw errors.
                  Needs thresholds > 0 (true for reconstruction errors).
    mode="diff":  margin = score - threshold. Same alarms, but AUROC/AUPRC are
                  dominated by the stage with the largest error scale.

    A second is NaN if any stage is NaN there (the window warm-up).
    """
    scores = _aligned(stage_scores)
    margins = []
    for name, s in scores.items():
        t = _threshold(thresholds, name)
        if mode == "ratio":
            if t <= 0:
                raise ValueError(f"mode='ratio' needs a threshold > 0, got {t} for {name!r}")
            margins.append(s / t - 1.0)
        elif mode == "diff":
            margins.append(s - t)
        else:
            raise ValueError("mode must be 'ratio' or 'diff'")
    return np.max(np.vstack(margins), axis=0)  # NaN propagates


def evaluate_any_alarm(
    stage_scores: Mapping,
    labels: Sequence[int],
    thresholds: float | Mapping,
    mode: str = "ratio",
    **evaluate_kwargs,
) -> dict:
    """Plant-wide :func:`evaluate` of the any-stage alarm (see :func:`any_alarm_scores`).

    The reported ``threshold`` is 0 (the decision is on the combined margin).
    Two extra keys show which stages drive the result:

    * ``stage_alarms``:       ``{client: seconds where that stage alarmed}``
    * ``stage_false_alarms``: ``{client: of those, seconds with no attack}``

    A stage with many false alarms (for example one with a drifting analyser
    sensor) pulls down the plant-wide precision on its own; these counts show
    which one it is.
    """
    scores = _aligned(stage_scores, labels)
    y = np.asarray(labels).astype(int).ravel()
    combined = any_alarm_scores(scores, thresholds, mode)
    res = evaluate(combined, y, 0.0, **evaluate_kwargs)

    valid = ~np.isnan(combined)
    res["stage_alarms"] = {}
    res["stage_false_alarms"] = {}
    for name, s in scores.items():
        alarm = valid & (np.nan_to_num(s, nan=-np.inf) > _threshold(thresholds, name))
        res["stage_alarms"][name] = int(alarm.sum())
        res["stage_false_alarms"][name] = int((alarm & (y == 0)).sum())
    return res
