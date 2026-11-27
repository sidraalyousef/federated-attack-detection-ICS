import numpy as np
import pytest

from src.eval.late_alarms import evaluate_with_grace, grace_periods


def _labels():
    y = np.zeros(100, dtype=int)
    y[10:20] = 1          # attack A
    y[40:50] = 1          # attack B, 20 s after A
    return y


def test_grace_periods_stop_at_next_attack_and_end():
    assert grace_periods(_labels(), grace=30) == [(20, 40), (50, 80)]
    assert grace_periods(_labels(), grace=100) == [(20, 40), (50, 100)]


def test_late_alarm_counts_only_for_its_own_attack():
    y = _labels()
    s = np.zeros(100)
    s[25] = 1.0           # 5 s after A ends -> A caught with grace
    r = evaluate_with_grace(s, y, 0.5, grace=10, include_etapr=False)
    assert r["attacks_caught_grace"] == 1 and r["n_attacks"] == 2
    assert r["n_false_alarms"] == 0    # the late alarm is not a false alarm
    s2 = np.zeros(100)
    s2[42] = 1.0          # inside B: must not count for A
    r2 = evaluate_with_grace(s2, y, 0.5, grace=100, include_etapr=False)
    assert r2["attacks_caught_grace"] == 1


def test_alarm_after_grace_is_false_and_nan_is_silent():
    y = _labels()
    s = np.zeros(100)
    s[90] = 1.0
    s[:5] = np.nan
    r = evaluate_with_grace(s, y, 0.5, grace=10, include_etapr=False)
    assert r["attacks_caught_grace"] == 0
    assert r["n_false_alarms"] == 1
    with pytest.raises(ValueError):
        evaluate_with_grace(s[:50], y, 0.5)
