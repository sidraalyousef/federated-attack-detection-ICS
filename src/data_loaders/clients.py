"""Split preprocessed SWaT data into federated clients, then cut windows.

Clients are split by time. Each client gets:

* a consecutive piece of the attack file (its test data), cut only in quiet
  periods so every attack stays whole and is followed by at least
  ``MIN_QUIET_S`` (1 h) of recovery inside the same client;
* the matching piece of the normal training data and of the validation data:
  both are cut into ``n_clients`` equal consecutive pieces, and client k gets
  piece k of each. Joined back together, the pieces are exactly the
  centralized train / validation sets, so centralized, local-only and FedAvg
  all see the same data.

Two settings are defined (``CLIENT_STARTS``):

* 5 clients (main): C4 has no attacks (judge it by false alarms).
* 3 clients (check): C1+C2, C3+C4 and C5 of the 5-client split. Every client
  has attacks; the no-attack rule applies to the 5-client setting only.

Typical use::

    d = prepare_swat("data/raw/swat")
    clients = split_clients(d, n_clients=5)
    for c in clients:
        w = client_windows(c)          # train stride 10, val/test stride 1
        ...train / score...
    scores, labels, groups = points_from_clients(test_window_scores, clients)
    evaluate_groups(scores, labels, groups, thresholds)
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise

import numpy as np
import pandas as pd

from .preprocessing import EVAL_STRIDE, TRAIN_STRIDE, WINDOW, make_windows

MIN_QUIET_S = 3_600  # recovery time kept after the last attack of a client

# First second of each client after C1, in the attack file.
_C2, _C3, _C4, _C5 = (
    "2015-12-29 13:02:01",
    "2015-12-30 18:29:01",
    "2015-12-31 23:11:41",
    "2016-01-01 10:26:00",
)
CLIENT_STARTS: dict[int, tuple[str, ...]] = {
    5: (_C2, _C3, _C4, _C5),
    3: (_C3, _C5),  # C1+C2 | C3+C4 | C5
}


def attack_bounds(attack_timestamps, n_clients: int) -> list[tuple[int, int]]:
    """``(start, end)`` row ranges of each client in the attack file, end exclusive."""
    if n_clients not in CLIENT_STARTS:
        raise ValueError(f"n_clients must be one of {sorted(CLIENT_STARTS)}")
    ts = pd.to_datetime(np.asarray(attack_timestamps))
    starts = []
    for s in CLIENT_STARTS[n_clients]:
        i = int(ts.searchsorted(pd.Timestamp(s)))
        if i >= len(ts) or ts[i] != pd.Timestamp(s):
            raise ValueError(f"client start {s} not found in the attack timestamps")
        starts.append(i)
    edges = [0, *starts, len(ts)]
    return list(pairwise(edges))


def check_bounds(labels, bounds, min_quiet: int = MIN_QUIET_S) -> None:
    """Raise if a cut splits an attack or leaves less than ``min_quiet`` after one."""
    y = np.asarray(labels).astype(int)
    for k, (s, e) in enumerate(bounds, start=1):
        part = y[s:e]
        if part[0] == 1 or part[-1] == 1:
            raise ValueError(f"client {k} starts or ends inside an attack")
        attack_rows = np.flatnonzero(part)
        if attack_rows.size and (len(part) - 1 - attack_rows[-1]) < min_quiet:
            raise ValueError(f"client {k} has less than {min_quiet} s after its last attack")


def split_clients(d: Mapping, n_clients: int = 5, min_quiet: int = MIN_QUIET_S) -> list[dict]:
    """Split the output of ``prepare_swat`` into clients.

    Returns one dict per client, in time order, with keys ``name`` ("C1", ...),
    ``train``, ``val``, ``attack`` (2-D scaled arrays), ``attack_labels``,
    ``train_timestamps``, ``val_timestamps``, ``attack_timestamps`` and
    ``attack_rows`` (its ``(start, end)`` rows in the full attack file).
    """
    bounds = attack_bounds(d["attack_timestamps"], n_clients)
    check_bounds(d["attack_labels"], bounds, min_quiet)
    train_parts = np.array_split(np.arange(len(d["train"])), n_clients)
    val_parts = np.array_split(np.arange(len(d["val"])), n_clients)

    clients = []
    for k, ((s, e), tr, va) in enumerate(zip(bounds, train_parts, val_parts), start=1):
        clients.append({
            "name": f"C{k}",
            "train": d["train"][tr[0]:tr[-1] + 1],
            "val": d["val"][va[0]:va[-1] + 1],
            "attack": d["attack"][s:e],
            "attack_labels": d["attack_labels"][s:e],
            "train_timestamps": d["train_timestamps"][tr[0]:tr[-1] + 1],
            "val_timestamps": d["val_timestamps"][va[0]:va[-1] + 1],
            "attack_timestamps": d["attack_timestamps"][s:e],
            "attack_rows": (s, e),
        })
    return clients


def client_windows(client: Mapping, window: int = WINDOW,
                   train_stride: int = TRAIN_STRIDE, eval_stride: int = EVAL_STRIDE) -> dict:
    """Windows cut inside one client, so no window crosses a client boundary.

    ``train`` uses ``train_stride`` (10); ``val`` and ``attack`` use
    ``eval_stride`` (1) so every second can be scored. All are read-only views.
    """
    return {
        "train": make_windows(client["train"], window, train_stride),
        "val": make_windows(client["val"], window, eval_stride),
        "attack": make_windows(client["attack"], window, eval_stride),
    }


def summary(clients: Sequence[Mapping]) -> pd.DataFrame:
    """One row per client: sizes in hours, number of attacks, attack seconds."""
    rows = []
    for c in clients:
        y = np.asarray(c["attack_labels"]).astype(int)
        rows.append({
            "client": c["name"],
            "train_h": len(c["train"]) / 3600,
            "val_h": len(c["val"]) / 3600,
            "test_h": len(y) / 3600,
            "test_start": pd.Timestamp(c["attack_timestamps"][0]),
            "test_end": pd.Timestamp(c["attack_timestamps"][-1]),
            "attacks": int((np.diff(np.r_[0, y]) == 1).sum()),
            "attack_s": int(y.sum()),
        })
    return pd.DataFrame(rows).set_index("client").round(1)


def points_from_clients(window_scores: Sequence, clients: Sequence[Mapping],
                        window: int = WINDOW, eval_stride: int = EVAL_STRIDE,
                        part: str = "attack"):
    """Turn per-client window scores into ``(scores, labels, groups)`` per second.

    ``window_scores[k]`` holds the scores of ``client_windows(clients[k])[part]``.
    The result goes straight into ``src.eval.evaluate_groups`` (labels are None
    for ``part="val"``). Each client's first ``window - 1`` seconds are NaN and
    are dropped by ``evaluate``.
    """
    from src.eval import window_scores_to_points

    if len(window_scores) != len(clients):
        raise ValueError("need one array of window scores per client")
    scores, labels, groups = [], [], []
    for ws, c in zip(window_scores, clients):
        n = len(c[part])
        scores.append(window_scores_to_points(ws, n, window, eval_stride))
        groups.append(np.full(n, c["name"]))
        if part == "attack":
            labels.append(np.asarray(c["attack_labels"]))
    return (np.concatenate(scores),
            np.concatenate(labels) if labels else None,
            np.concatenate(groups))
