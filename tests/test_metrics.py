import numpy as np
import pytest

from src.eval import (
    best_f1_threshold,
    evaluate,
    evaluate_groups,
    get_segments,
    point_adjust,
    prf,
    random_scores,
    threshold_from_normal,
    window_scores_to_points,
)

# labels: two segments, [3,7) length 4 and [10,12) length 2
LABELS = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0])


def test_get_segments():
    assert get_segments(LABELS) == [(3, 7), (10, 12)]
    assert get_segments([1, 1, 0, 1]) == [(0, 2), (3, 4)]
    assert get_segments([0, 0, 0]) == []


def test_prf_basic_and_edge_cases():
    r = prf(LABELS, LABELS)
    assert (r.precision, r.recall, r.f1) == (1.0, 1.0, 1.0)
    r = prf(np.zeros_like(LABELS), LABELS)
    assert (r.precision, r.recall, r.f1) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        prf([0, 1], [0, 1, 0])


def test_pa_single_hit_fills_segment():
    preds = np.zeros_like(LABELS)
    preds[4] = 1  # one hit in the first segment
    adj = point_adjust(preds, LABELS, k=0)
    assert adj[3:7].tolist() == [1, 1, 1, 1]
    assert adj[10:12].tolist() == [0, 0]  # untouched segment stays undetected


def test_pa_never_changes_false_positives():
    preds = np.zeros_like(LABELS)
    preds[0] = 1
    adj = point_adjust(preds, LABELS, k=0)
    assert adj[0] == 1 and adj.sum() == 1


def test_pak_threshold_rule():
    preds = np.zeros_like(LABELS)
    preds[3:5] = 1  # 50% of first segment
    assert point_adjust(preds, LABELS, k=40)[3:7].sum() == 4  # 50% > 40%
    assert point_adjust(preds, LABELS, k=50)[3:7].sum() == 2  # 50% not > 50%
    # K=100 never adjusts -> identical to raw
    assert np.array_equal(point_adjust(preds, LABELS, k=100), preds)


def test_evaluate_ordering_pa_ge_pak_ge_raw():
    scores = np.zeros(len(LABELS))
    scores[4] = 1.0
    res = evaluate(scores, LABELS, threshold=0.5)
    assert res["pa_f1"] >= res["pak_auc_f1"] >= res["raw_f1"]
    assert res["pak_f1"][0] == pytest.approx(res["pa_f1"])
    assert res["pak_f1"][100] == pytest.approx(res["raw_f1"])
    assert res["n_segments"] == 2 and res["n_segments_detected"] == 1


def test_perfect_scores():
    res = evaluate(LABELS.astype(float), LABELS, threshold=0.5)
    for key in ("raw_f1", "pa_f1", "pak_auc_f1", "auroc", "auprc"):
        assert res[key] == pytest.approx(1.0)


def test_random_scores_inflated_by_pa():
    # Long segments make PA reward random scores, the effect Kim et al. describe.
    labels = np.zeros(20_000, dtype=int)
    for start in range(1_000, 19_000, 2_000):
        labels[start : start + 500] = 1
    s = random_scores(labels.size, seed=1)
    res = evaluate(s, labels, threshold=0.99)
    assert res["pa_f1"] > 0.5
    assert res["raw_f1"] < 0.1
    assert res["pak_auc_f1"] < res["pa_f1"]
    assert 0.45 < res["auroc"] < 0.55


def test_nan_scores_dropped_with_labels():
    scores = np.array([np.nan, np.nan, 0.1, 0.9, 0.9])
    labels = np.array([1, 1, 0, 1, 1])
    res = evaluate(scores, labels, threshold=0.5)
    assert res["n_points"] == 3 and res["raw_f1"] == 1.0


def test_window_alignment_last():
    # 10 points, window 4, stride 1 -> 7 windows; first 3 points have no score
    w = np.arange(7, dtype=float)
    pts = window_scores_to_points(w, n_points=10, window_size=4)
    assert np.isnan(pts[:3]).all()
    assert pts[3:].tolist() == list(range(7))


def test_window_alignment_stride_and_modes():
    w = np.array([1.0, 5.0])  # windows [0,4) and [2,6)
    last = window_scores_to_points(w, 6, 4, stride=2, mode="last")
    assert last[3] == 1.0 and last[5] == 5.0 and np.isnan(last[4])
    mx = window_scores_to_points(w, 6, 4, stride=2, mode="max")
    assert mx.tolist() == [1, 1, 5, 5, 5, 5]
    mean = window_scores_to_points(w, 6, 4, stride=2, mode="mean")
    assert mean.tolist() == [1, 1, 3, 3, 5, 5]
    with pytest.raises(ValueError):
        window_scores_to_points(np.ones(5), 6, 4, stride=2)


def test_threshold_from_normal():
    normal = np.arange(101, dtype=float)
    assert threshold_from_normal(normal, "percentile", q=99) == pytest.approx(99.0)
    assert threshold_from_normal(normal, "std", n_std=0) == pytest.approx(50.0)


def test_best_f1_threshold_finds_separator():
    scores = LABELS * 0.8 + 0.1
    t, f1 = best_f1_threshold(scores, LABELS)
    assert f1 == pytest.approx(1.0)
    assert 0.1 <= t < 0.9


def test_evaluate_groups_per_client_thresholds():
    scores = np.concatenate([LABELS * 1.0, LABELS * 10.0])
    labels = np.concatenate([LABELS, LABELS])
    groups = np.array(["c1"] * len(LABELS) + ["c2"] * len(LABELS))
    res = evaluate_groups(scores, labels, groups, {"c1": 0.5, "c2": 5.0})
    assert res["c1"]["raw_f1"] == 1.0 and res["c2"]["raw_f1"] == 1.0
    res_global = evaluate_groups(scores, labels, groups, 5.0)
    assert res_global["c1"]["raw_f1"] == 0.0
