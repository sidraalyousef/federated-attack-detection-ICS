import numpy as np
import pandas as pd
import pytest

from src.eval import evaluate, evaluate_groups
from src.eval.federated import (
    client_summary,
    client_table,
    compare_settings,
    pooled_evaluate,
)

# two attacks: [3,7) and [10,12); 6 attack points, 8 normal points
LABELS = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0])
NO_ATTACKS = np.zeros(14, dtype=int)


def _results(scale_b=1.0):
    """Client a: perfect. Client b: perfect but scores scaled. Client c: no attacks,
    2 of its 14 points are false alarms."""
    c_scores = np.zeros(14)
    c_scores[[1, 8]] = 1.0
    scores = np.concatenate([LABELS * 1.0, LABELS * scale_b, c_scores])
    labels = np.concatenate([LABELS, LABELS, NO_ATTACKS])
    groups = np.array(["a"] * 14 + ["b"] * 14 + ["c"] * 14)
    return scores, labels, groups


def test_no_attack_client_metrics_are_nan_but_fpr_kept():
    scores, labels, groups = _results()
    table = client_table(evaluate_groups(scores, labels, groups, 0.5))
    assert list(table.index) == ["a", "b", "c"]
    assert table.loc["c", ["raw_f1", "pa_f1", "etapr_f1", "auroc"]].isna().all()
    assert table.loc["c", "n_false_alarms"] == 2
    assert table.loc["c", "fpr"] == pytest.approx(2 / 14)
    assert table.loc["a", "raw_f1"] == pytest.approx(1.0)


def test_summary_skips_clients_without_attacks():
    scores, labels, groups = _results()
    res = evaluate_groups(scores, labels, groups, 0.5)
    summary = client_summary(client_table(res))
    assert summary.loc["mean", "raw_f1"] == pytest.approx(1.0)  # a and b only
    assert summary.loc["min", "raw_f1"] == pytest.approx(1.0)
    # fpr is averaged over ALL clients: (0 + 0 + 2/14) / 3
    assert summary.loc["mean", "fpr"] == pytest.approx((2 / 14) / 3)


def test_summary_min_shows_worst_client():
    scores, labels, groups = _results()
    scores[14:28] = 0.0  # client b detects nothing
    summary = client_summary(client_table(evaluate_groups(scores, labels, groups, 0.5)))
    assert summary.loc["min", "raw_f1"] == 0.0
    assert summary.loc["mean", "raw_f1"] == pytest.approx(0.5)


def test_table_without_etapr_gives_nan_column():
    scores, labels, groups = _results()
    res = evaluate_groups(scores, labels, groups, 0.5, include_etapr=False)
    table = client_table(res)
    assert table["etapr_f1"].isna().all()
    assert table.loc["a", "raw_f1"] == pytest.approx(1.0)


def test_compare_settings_layout():
    scores, labels, groups = _results()
    good = evaluate_groups(scores, labels, groups, 0.5)
    bad_scores = scores.copy()
    bad_scores[:28] = 0.0
    bad = evaluate_groups(bad_scores, labels, groups, 0.5)
    out = compare_settings({"centralized": good, "local-only": bad})
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["centralized", "local-only"]
    assert out.loc["centralized", "raw_f1"] == pytest.approx(1.0)
    assert out.loc["local-only", "raw_f1"] == 0.0
    assert out.loc["centralized", "n_clients"] == 3
    assert out.loc["centralized", "n_clients_with_attacks"] == 2
    assert "raw_f1_std" in out.columns
    assert "raw_f1_std" not in compare_settings({"x": good}, with_std=False).columns


def test_pooled_matches_evaluate_on_concatenation():
    scores, labels, groups = _results()
    pooled = pooled_evaluate(scores, labels, groups, 0.5)
    direct = evaluate(scores, labels, 0.5)
    for key in ("raw_f1", "pa_f1", "pak_auc_f1", "auroc", "n_false_alarms"):
        assert pooled[key] == pytest.approx(direct[key])


def test_pooled_per_client_thresholds():
    # client b's scores are 10x larger; one global threshold cannot serve both
    scores, labels, groups = _results(scale_b=10.0)
    thr = {"a": 0.5, "b": 5.0, "c": 0.5}
    per_client = pooled_evaluate(scores, labels, groups, thr)
    assert per_client["raw_recall"] == pytest.approx(1.0)
    assert per_client["threshold"] == 0.0
    global_thr = pooled_evaluate(scores, labels, groups, 5.0)
    assert global_thr["raw_recall"] < 1.0


def test_pooled_drops_nan_warmup_per_client():
    scores, labels, groups = _results()
    scores[0] = np.nan
    scores[14] = np.nan
    pooled = pooled_evaluate(scores, labels, groups, 0.5)
    assert pooled["n_points"] == 40


def test_pooled_warns_when_attack_touches_client_boundary():
    labels = np.array([0, 0, 1, 1, 1, 1, 0, 0])  # attack ends client a, resumes in b
    scores = labels * 1.0
    groups = np.array(["a"] * 4 + ["b"] * 4)
    labels_touch = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    with pytest.warns(UserWarning, match="boundary"):
        pooled_evaluate(scores, labels_touch, groups, 0.5)


def test_pooled_no_warning_when_attacks_are_inside_clients(recwarn):
    scores, labels, groups = _results()
    pooled_evaluate(scores, labels, groups, 0.5)
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_pooled_all_nan_raises():
    with pytest.raises(ValueError):
        pooled_evaluate([np.nan, np.nan], [0, 1], ["a", "a"], 0.5)
