"""Federated evaluation helpers, layered on top of :mod:`src.eval.metrics`.

Nothing here computes a new detection metric. Every number comes from
:func:`src.eval.metrics.evaluate`; this module only organises the results of
several clients (or several training settings) into tables:

* :func:`client_table`     - one row per client.
* :func:`client_summary`   - mean / std / min across clients.
* :func:`compare_settings` - centralized vs local-only vs FedAvg ... in one table.
* :func:`pooled_evaluate`  - one score over all clients' data together.

Typical use::

    per_client = evaluate_groups(scores, labels, client_ids, thresholds)
    table = client_table(per_client)
    summary = client_summary(table)

    compare_settings({
        "centralized": evaluate_groups(s_central, y, client_ids, thr_central),
        "local-only":  evaluate_groups(s_local,   y, client_ids, thr_local),
        "FedAvg":      evaluate_groups(s_fedavg,  y, client_ids, thr_fedavg),
    })

Score the centralized model on the same client slices as the federated ones,
so all settings are compared on identical data.

Clients without attacks
-----------------------
On a test slice with no attacks every F1 is 0 and AUROC/AUPRC are NaN,
whatever the model does, so these numbers say nothing. Here they are set to
NaN (and therefore skipped by the mean/std), while ``n_false_alarms`` and
``fpr`` are kept for every client. Judge a no-attack client by its false
alarms.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import evaluate

DEFAULT_TABLE_METRICS: tuple[str, ...] = (
    "raw_f1",
    "pa_f1",
    "pak_auc_f1",
    "etapr_f1",
    "auroc",
    "auprc",
)
# Columns that are valid for every client, attacks or not.
COUNT_COLUMNS: tuple[str, ...] = (
    "n_points",
    "n_attack_points",
    "n_segments",
    "n_segments_detected",
    "n_false_alarms",
    "fpr",
)


def client_table(
    results: Mapping,
    metrics: Sequence[str] = DEFAULT_TABLE_METRICS,
) -> pd.DataFrame:
    """One row per client from ``{client: evaluate(...) result}``.

    Columns: the requested detection ``metrics`` (NaN for clients with no
    attacks, or when a metric such as ``etapr_f1`` was not computed), followed
    by :data:`COUNT_COLUMNS` (always filled).
    """
    metrics = tuple(metrics)
    columns = list(metrics) + list(COUNT_COLUMNS)
    clients, rows = [], []
    for client, res in results.items():
        has_attacks = res["n_attack_points"] > 0
        row = [res[m] if has_attacks and m in res else np.nan for m in metrics]
        row += [res[c] for c in COUNT_COLUMNS]
        clients.append(client)
        rows.append(row)
    table = pd.DataFrame(rows, index=pd.Index(clients, name="client"), columns=columns)
    return table


def client_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean, std and min across clients, for a :func:`client_table` result.

    Detection metrics are summarised over clients that have attacks only (NaNs
    are skipped). ``fpr`` is summarised over all clients. The ``min`` row shows
    the worst client, which is what matters when asking whether FedAvg helps
    everyone or only some clients. ``std`` uses one degree of freedom, so it is
    NaN with a single client.
    """
    metric_cols = [c for c in table.columns if c not in COUNT_COLUMNS]
    return table[metric_cols + ["fpr"]].agg(["mean", "std", "min"])


def compare_settings(
    named_results: Mapping[str, Mapping],
    metrics: Sequence[str] = DEFAULT_TABLE_METRICS,
    with_std: bool = True,
) -> pd.DataFrame:
    """One row per setting (centralized / local-only / FedAvg ...).

    ``named_results`` maps a setting name to its ``{client: result}`` dict.
    Each metric column is the mean over clients with attacks; ``<metric>_std``
    is the std over those clients. ``fpr`` is the mean false-alarm rate over
    all clients. ``n_clients`` and ``n_clients_with_attacks`` show how many
    clients each average rests on.
    """
    rows = {}
    for name, results in named_results.items():
        table = client_table(results, metrics)
        summary = client_summary(table)
        row = {}
        for m in list(metrics) + ["fpr"]:
            row[m] = summary.loc["mean", m]
            if with_std:
                row[f"{m}_std"] = summary.loc["std", m]
        row["n_clients"] = len(table)
        row["n_clients_with_attacks"] = int((table["n_attack_points"] > 0).sum())
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def pooled_evaluate(
    scores: Sequence[float],
    labels: Sequence[int],
    groups: Sequence,
    threshold: float | Mapping,
    **evaluate_kwargs,
) -> dict:
    """Evaluate all clients' data together, as one test set.

    Clients with more attack points weigh more here, unlike the per-client
    mean. Clients are concatenated in order of first appearance in ``groups``;
    NaN warm-up points are dropped per client first. A warning is raised if an
    attack runs to the end of one client and continues at the start of the next,
    because the two would then be counted as a single attack.

    ``threshold`` is one global float, or a ``{client: threshold}`` mapping. With
    a mapping, each client's scores are shifted by its own threshold (score minus
    threshold, decision at 0), so AUROC/AUPRC are then computed on those shifted
    margins and the reported ``threshold`` is 0.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels).astype(int)
    g = np.asarray(groups)
    per_client = isinstance(threshold, Mapping)

    parts_s: list[np.ndarray] = []
    parts_y: list[np.ndarray] = []
    for grp in dict.fromkeys(g.tolist()):
        m = g == grp
        sc, yy = s[m], y[m]
        keep = ~np.isnan(sc)
        sc, yy = sc[keep], yy[keep]
        if sc.size == 0:
            continue
        if per_client:
            sc = sc - threshold[grp]
        if parts_y and parts_y[-1][-1] == 1 and yy[0] == 1:
            warnings.warn(
                f"an attack touches the boundary before client {grp!r}; the two "
                "pieces count as one attack in the pooled result",
                stacklevel=2,
            )
        parts_s.append(sc)
        parts_y.append(yy)

    if not parts_s:
        raise ValueError("no non-NaN scores to evaluate")
    return evaluate(
        np.concatenate(parts_s),
        np.concatenate(parts_y),
        0.0 if per_client else threshold,
        **evaluate_kwargs,
    )
