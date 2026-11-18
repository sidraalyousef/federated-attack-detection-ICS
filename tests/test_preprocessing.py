import numpy as np
import pandas as pd
import pytest

from src.data_loaders.preprocessing import (
    apply_minmax,
    fit_minmax,
    make_windows,
    split_normal,
)


def _normal(n):
    return pd.DataFrame({"Timestamp": pd.date_range("2015-12-22", periods=n, freq="s"),
                         "A": np.arange(n, dtype=float), "label": 0})


def test_split_drops_warmup_and_keeps_time_order():
    train, val = split_normal(_normal(1000), warmup=200, val_frac=0.25)
    assert len(train) == 600 and len(val) == 200
    assert train["A"].iloc[0] == 200            # warm-up rows gone
    assert train["A"].iloc[-1] < val["A"].iloc[0]  # val comes after train


def test_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        split_normal(_normal(100), warmup=0, val_frac=1.0)


def test_scaler_fitted_on_train_only():
    train = np.array([[0.0], [10.0]])
    val = np.array([[20.0]])
    lo, rng = fit_minmax(train)
    assert apply_minmax(val, lo, rng)[0, 0] == pytest.approx(2.0)  # outside 0-1 is allowed


def test_constant_column_gets_unit_range():
    _, rng = fit_minmax(np.array([[5.0], [5.0]]))
    assert rng[0] == 1.0


def test_windows_count_and_eval_stride():
    x = np.arange(100, dtype=np.float32).reshape(100, 1)
    assert make_windows(x, window=10, stride=10).shape == (10, 10, 1)
    w = make_windows(x, window=10, stride=1)
    assert w.shape == (91, 10, 1)
    assert w[5, -1, 0] == 14  # window i ends at timestep i + window - 1
