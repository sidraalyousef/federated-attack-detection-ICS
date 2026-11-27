import numpy as np
import pytest

from src.data_loaders.preprocessing import make_windows
from src.eval.adaptive import (
    aggregate,
    factor_from_normal,
    fit_robust,
    fit_sensor_bars,
    per_sensor_dynamic_scores,
    ratio_to_threshold,
    robust_z,
    rolling_median,
    rolling_threshold,
)
from src.models import ConvAE, score_windows, sensor_errors
from src.models.online import stream_scores

W, F = 60, 43


def test_robust_z_uses_median_and_iqr():
    e = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [np.nan, np.nan]])
    med, iqr = fit_robust(e)
    np.testing.assert_allclose(med, [2.0, 20.0])
    np.testing.assert_allclose(iqr, [1.0, 10.0])
    np.testing.assert_allclose(robust_z(e[:3], med, iqr), [[-1, -1], [0, 0], [1, 1]])


def test_kth_needs_k_sensors():
    z = np.array([[100.0, 0, 0, 0], [5.0, 5.0, 5.0, 0], [np.nan] * 4])
    np.testing.assert_allclose(aggregate(z, "max")[:2], [100, 5])
    np.testing.assert_allclose(aggregate(z, "kth", k=3)[:2], [0, 5])   # one huge sensor alone -> 0
    assert np.isnan(aggregate(z, "kth", k=3)[2])
    with pytest.raises(ValueError):
        aggregate(z, "kth", k=5)


def test_rolling_threshold_is_past_only():
    s = np.ones(2000)
    s[1500] = 1000.0                                  # a spike must not raise its own threshold
    t = rolling_threshold(s, history=np.ones(1000), window=500, factor=2.0, min_periods=10)
    assert t[1500] == pytest.approx(2.0)
    assert t[0] == pytest.approx(2.0)                 # history gives a baseline from the start


def test_rolling_threshold_follows_slow_drift():
    s = np.linspace(1, 10, 5000)                      # slow drift
    t = rolling_threshold(s, history=np.ones(100), window=200, factor=1.5, min_periods=10)
    assert np.nanmean(s > t) == 0                     # drift alone never alarms


def test_factor_and_ratio():
    assert factor_from_normal(np.r_[np.ones(99), 3.0], q=100) == pytest.approx(3.0)
    r = ratio_to_threshold([1.0, 4.0, np.nan], [2.0, 2.0, 2.0])
    np.testing.assert_allclose(r[:2], [0.5, 2.0])
    assert np.isnan(r[2])


def test_sensor_errors_average_to_window_mse():
    rng = np.random.default_rng(0)
    w = make_windows(rng.random((300, F)).astype(np.float32), W, 1)
    m = ConvAE(F, W)
    e = sensor_errors(m, w, batch_size=50, device="cpu")
    assert e.shape == (len(w), F)
    np.testing.assert_allclose(e.mean(1), score_windows(m, w, device="cpu"), rtol=1e-4)


@pytest.mark.parametrize("finetune", [False, True])
def test_stream_scores_shapes_and_no_side_effects(finetune):
    rng = np.random.default_rng(1)
    series = [rng.random((900, F)).astype(np.float32), rng.random((700, F)).astype(np.float32)]
    before = [s.copy() for s in series]
    train_w = make_windows(rng.random((2000, F)).astype(np.float32), W, 10)
    out = stream_scores(ConvAE(F, W), series, val_points=rng.random(500) + 0.5,
                        train_windows=train_w, finetune=finetune, threshold_window=400,
                        update_every=300, min_update=5, finetune_stride=1, device="cpu",
                        verbose=False)
    assert [len(s) for s in out["scores"]] == [900 - W + 1, 700 - W + 1]
    assert [len(t) for t in out["thresholds"]] == [900 - W + 1, 700 - W + 1]
    assert len(out["updates"]) == 3 + 3          # 841 and 641 windows, blocks of 300
    if finetune:
        assert any(u["updated"] for u in out["updates"])
    for s, b in zip(series, before):
        np.testing.assert_array_equal(s, b)


def test_stream_finetune_needs_replay():
    with pytest.raises(ValueError):
        stream_scores(ConvAE(F, W), [np.zeros((100, F), np.float32)], np.ones(10), finetune=True)


def test_rolling_median_2d_matches_columns():
    rng = np.random.default_rng(2)
    x, h = rng.random((300, 3)), rng.random((50, 3))
    m2 = rolling_median(x, h, window=40, min_periods=5)
    for j in range(3):
        np.testing.assert_allclose(m2[:, j], rolling_median(x[:, j], h[:, j], 40, 5))


def test_per_sensor_drift_does_not_mask_other_sensors():
    n = 3000
    e = np.ones((n, 3))
    e[:1000, 0] = np.linspace(1, 100, 1000)   # sensor 0 drifts to 100x, then stays there
    e[1000:, 0] = 100.0
    e[2500:2510, 1] = 5.0                     # later, sensor 1 jumps 5x for 10 s
    s = per_sensor_dynamic_scores(e, np.ones((300, 3)), window=300, min_periods=10)
    assert s[1500:2500].max() < 1.1           # once the drift has settled: no alarm
    assert s[2500:2510].min() > 4.5           # the jump stands out clearly
    shared = e.mean(1)                        # one shared score: the jump is masked
    assert shared[2500] / np.median(shared[2200:2500]) < 1.1


def test_stream_per_sensor_threshold():
    rng = np.random.default_rng(3)
    series = [rng.random((700, F)).astype(np.float32)]
    train_w = make_windows(rng.random((2000, F)).astype(np.float32), W, 10)
    out = stream_scores(ConvAE(F, W), series, val_points=rng.random((800, F)) + 0.5,
                        train_windows=train_w, finetune=True, threshold="per_sensor",
                        threshold_window=400, update_every=300, min_update=5,
                        finetune_stride=1, device="cpu", verbose=False)
    assert len(out["scores"][0]) == 700 - W + 1
    assert np.all(out["thresholds"][0] == out["factor"])
    with pytest.raises(ValueError):
        stream_scores(ConvAE(F, W), series, val_points=np.ones(100), threshold="per_sensor")


def test_sensor_bars_tighten_quiet_sensors():
    rng = np.random.default_rng(4)
    e = np.ones((3000, 2))
    e[:, 0] += rng.normal(0, 0.5, 3000).clip(-0.9)       # noisy sensor
    e[:, 1] += rng.normal(0, 0.01, 3000)                  # quiet sensor
    bars = fit_sensor_bars(e, window=500, min_periods=10)
    assert bars[0] > bars[1] > 1.0
    np.testing.assert_allclose(fit_sensor_bars(e, window=500, min_periods=10, strength=0.0), 1.0)
    np.testing.assert_allclose(fit_sensor_bars(e, window=500, min_periods=10, strength=0.5),
                               np.sqrt(bars))
    with pytest.raises(ValueError):
        fit_sensor_bars(e, strength=2.0)
    test = np.ones((50, 2)); test[:, 1] = 1.5             # +50% on the quiet sensor
    plain = per_sensor_dynamic_scores(test, e, window=500, min_periods=10)
    barred = per_sensor_dynamic_scores(test, e, window=500, min_periods=10, bars=bars)
    assert barred[-1] > plain[-1] / bars[1] - 1e-9         # the quiet sensor's change is scaled up


def test_hybrid_scores_shapes_and_threshold():
    rng = np.random.default_rng(5)
    series = [rng.random((700, F)).astype(np.float32)]
    train_w = make_windows(rng.random((2000, F)).astype(np.float32), W, 10)
    out = stream_scores(ConvAE(F, W), series, val_points=rng.random((800, F)) + 0.5,
                        train_windows=train_w, finetune=True, threshold="hybrid",
                        bar_strength=0.25, threshold_window=400, update_every=300,
                        min_update=5, finetune_stride=1, device="cpu", verbose=False)
    assert len(out["scores"][0]) == 700 - W + 1
    assert np.all(out["thresholds"][0] == out["factor"])
    assert out["factor"] >= 1.0       # max of two scores each scaled to their own 99th pct
    with pytest.raises(ValueError):
        stream_scores(ConvAE(F, W), series, val_points=np.ones(100), threshold="hybrid")


def test_shared_score_is_mse_over_past_median():
    from src.eval.adaptive import shared_dynamic_scores
    rng = np.random.default_rng(6)
    hist, e = rng.random((400, 5)) + 0.5, rng.random((300, 5)) + 0.5
    got = shared_dynamic_scores(e, hist, window=200, min_periods=10)
    mse = np.r_[hist.mean(1), e.mean(1)]
    for t in (0, 150, 299):
        past = mse[400 + t - 200:400 + t]
        assert got[t] == pytest.approx(mse[400 + t] / np.median(past))
