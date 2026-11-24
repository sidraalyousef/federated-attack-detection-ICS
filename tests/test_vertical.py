import numpy as np
import pytest

from src.eval import evaluate
from src.eval.federated import client_table
from src.eval.vertical import (
    any_alarm_scores,
    evaluate_any_alarm,
    evaluate_own_attacks,
    evaluate_stages,
    points_from_stages,
    stages_of,
)

# Two attacks: [3, 6) aimed at stage 1 and [9, 12) aimed at stage 2.
LABELS = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0])
ATTACK_1 = np.zeros(14)
ATTACK_1[3:6] = 1.0
ATTACK_2 = np.zeros(14)
ATTACK_2[9:12] = 1.0


def _scores():
    """S1 sees only attack 1, S2 sees only attack 2 (on a 10x larger error scale),
    S5+6 sees nothing but has one false alarm at second 7."""
    s56 = np.zeros(14)
    s56[7] = 1.0
    scores = {"S1": ATTACK_1.copy(), "S2": ATTACK_2 * 10, "S5+6": s56}
    thresholds = {"S1": 0.5, "S2": 5.0, "S5+6": 0.5}
    return scores, thresholds


def test_stages_of_reads_client_names():
    assert stages_of("S1") == (1,)
    assert stages_of("S5+6") == (5, 6)
    with pytest.raises(ValueError):
        stages_of("C1")


def test_each_stage_alone_misses_the_other_attack():
    scores, thr = _scores()
    res = evaluate_stages(scores, LABELS, thr, include_etapr=False)
    assert res["S1"]["raw_recall"] == pytest.approx(0.5)
    assert res["S1"]["raw_precision"] == pytest.approx(1.0)
    assert res["S1"]["n_segments_detected"] == 1
    assert res["S5+6"]["raw_recall"] == 0.0
    assert res["S5+6"]["n_false_alarms"] == 1
    assert list(client_table(res).index) == ["S1", "S2", "S5+6"]


def test_own_attacks_drop_the_other_stages_attacks():
    scores, thr = _scores()
    res = evaluate_own_attacks(scores, LABELS, thr, [(1,), (2,)], include_etapr=False)
    for name in ("S1", "S2"):
        assert res[name]["raw_f1"] == pytest.approx(1.0)
        assert res[name]["n_own_attacks"] == 1
        assert res[name]["n_segments"] == 1
        assert res[name]["n_points"] == 11  # 14 seconds minus the other attack's 3
    # No attacks of its own: both attacks removed, judge it by false alarms only.
    assert res["S5+6"]["n_own_attacks"] == 0
    assert res["S5+6"]["n_attack_points"] == 0
    assert res["S5+6"]["n_false_alarms"] == 1


def test_own_attacks_shared_and_unknown_targets():
    scores, thr = _scores()
    # Attack 1 hits stages 1 and 2; attack 2's target is unknown (left out for all).
    res = evaluate_own_attacks(scores, LABELS, thr, [(1, 2), ()], include_etapr=False)
    assert res["S1"]["n_own_attacks"] == 1 and res["S2"]["n_own_attacks"] == 1
    assert res["S1"]["raw_f1"] == pytest.approx(1.0)
    assert res["S2"]["raw_recall"] == 0.0  # S2 only reacts to attack 2, which is left out


def test_own_attacks_needs_one_entry_per_labelled_segment():
    scores, thr = _scores()
    with pytest.raises(ValueError, match="2 attack segments"):
        evaluate_own_attacks(scores, LABELS, thr, [(1,)])


def test_own_attacks_with_explicit_client_stages():
    scores, thr = _scores()
    scores = {"a": scores["S1"], "b": scores["S2"]}
    thr = {"a": 0.5, "b": 5.0}
    res = evaluate_own_attacks(
        scores, LABELS, thr, [(1,), (2,)], client_stages={"a": (1,), "b": (2,)},
        include_etapr=False,
    )
    assert res["a"]["raw_f1"] == pytest.approx(1.0)


def test_any_alarm_fires_when_any_stage_is_above_its_threshold():
    scores, thr = _scores()
    combined = any_alarm_scores(scores, thr)
    expected = (LABELS == 1) | (np.arange(14) == 7)
    np.testing.assert_array_equal(combined > 0, expected)
    np.testing.assert_array_equal(any_alarm_scores(scores, thr, mode="diff") > 0, expected)


def test_any_alarm_plant_result_and_stage_counts():
    scores, thr = _scores()
    res = evaluate_any_alarm(scores, LABELS, thr, include_etapr=False)
    assert res["raw_recall"] == pytest.approx(1.0)
    assert res["raw_precision"] == pytest.approx(6 / 7)
    assert res["n_segments_detected"] == 2
    assert res["threshold"] == 0.0
    assert res["stage_alarms"] == {"S1": 3, "S2": 3, "S5+6": 1}
    assert res["stage_false_alarms"] == {"S1": 0, "S2": 0, "S5+6": 1}


def test_ratio_mode_puts_stages_on_one_scale():
    # S2's errors are 100x larger. Its small false alarm at second 6 (10% over its
    # threshold) outranks S1's clear attack (100% over) with "diff", not with "ratio".
    s1 = np.array([0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1])
    s2 = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 55.0, 10.0])
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    scores, thr = {"S1": s1, "S2": s2}, {"S1": 0.5, "S2": 50.0}
    ratio = evaluate_any_alarm(scores, y, thr, mode="ratio", include_etapr=False)
    diff = evaluate_any_alarm(scores, y, thr, mode="diff", include_etapr=False)
    assert ratio["auroc"] == pytest.approx(1.0)
    assert diff["auroc"] < 1.0
    assert ratio["raw_f1"] == diff["raw_f1"]  # same alarms either way


def test_any_alarm_with_one_global_threshold_and_nan_warmup():
    s1 = np.array([np.nan, np.nan, 0.0, 1.0, 0.0])
    s2 = np.array([np.nan, np.nan, 0.0, 0.0, 0.0])
    y = np.array([0, 0, 0, 1, 0])
    res = evaluate_any_alarm({"S1": s1, "S2": s2}, y, 0.5, include_etapr=False)
    assert res["n_points"] == 3  # warm-up seconds dropped
    assert res["raw_f1"] == pytest.approx(1.0)


def test_misaligned_scores_rejected():
    with pytest.raises(ValueError, match="aligned"):
        evaluate_any_alarm({"S1": np.zeros(5), "S2": np.zeros(6)}, np.zeros(5), 0.5)
    with pytest.raises(ValueError, match="aligned"):
        evaluate_stages({"S1": np.zeros(5)}, np.zeros(6), 0.5)


def test_ratio_needs_positive_thresholds_and_known_mode():
    scores, _ = _scores()
    with pytest.raises(ValueError, match="> 0"):
        any_alarm_scores(scores, 0.0)
    with pytest.raises(ValueError, match="mode"):
        any_alarm_scores(scores, 0.5, mode="sum")
    with pytest.raises(KeyError):
        any_alarm_scores(scores, {"S1": 0.5})


def test_points_from_stages_maps_windows_to_seconds():
    clients = [{"name": "S1", "attack": np.zeros((10, 3))},
               {"name": "S2", "attack": np.zeros((10, 5))}]
    ws = [np.arange(7.0), np.arange(7.0) * 2]
    pts = points_from_stages(ws, clients, window=4)
    assert set(pts) == {"S1", "S2"}
    assert np.isnan(pts["S1"][:3]).all()
    np.testing.assert_array_equal(pts["S2"][3:], np.arange(7.0) * 2)
    # dict input gives the same result
    pts2 = points_from_stages({"S1": ws[0], "S2": ws[1]}, clients, window=4)
    np.testing.assert_array_equal(pts2["S1"][3:], pts["S1"][3:])


def test_points_from_stages_rejects_different_lengths():
    clients = [{"name": "S1", "attack": np.zeros((10, 3))},
               {"name": "S2", "attack": np.zeros((9, 5))}]
    with pytest.raises(ValueError, match="same seconds"):
        points_from_stages([np.zeros(7), np.zeros(6)], clients, window=4)


def test_centralized_and_any_alarm_in_one_table():
    scores, thr = _scores()
    plant = evaluate_any_alarm(scores, LABELS, thr, include_etapr=False)
    central = evaluate(LABELS * 1.0, LABELS, 0.5, include_etapr=False)
    table = client_table({"centralized": central, "local any-alarm": plant})
    assert list(table.index) == ["centralized", "local any-alarm"]
    assert table.loc["centralized", "raw_f1"] == pytest.approx(1.0)
