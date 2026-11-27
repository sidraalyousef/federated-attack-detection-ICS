import numpy as np
import pytest
import torch

from src.data_loaders.preprocessing import make_windows
from src.models import ConvAE, corrupt, score_windows, train_ae

W, F = 60, 43


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, F)).astype(np.float32)


def test_output_has_input_shape():
    model = ConvAE(F, W)
    x = torch.rand(8, W, F)
    assert model(x).shape == (8, W, F)


def test_rejects_conv_layout():
    # (B, sensors, time) must not be passed: forward does the permute itself
    with pytest.raises(ValueError):
        ConvAE(F, W)(torch.rand(2, F, W))


def test_window_must_divide_by_4():
    with pytest.raises(ValueError):
        ConvAE(F, 62)


def test_scores_read_only_windows_one_per_window():
    x = _data()
    windows = make_windows(x, W, 1)
    assert not windows.flags.writeable
    s = score_windows(ConvAE(F, W), windows, batch_size=100, device="cpu")
    assert s.shape == (len(x) - W + 1,)
    assert np.isfinite(s).all() and (s >= 0).all()


def test_score_matches_manual_mse():
    model = ConvAE(F, W).eval()
    windows = make_windows(_data(200), W, 1)
    s = score_windows(model, windows, batch_size=7, device="cpu")
    with torch.no_grad():
        x = torch.tensor(windows[:5])
        manual = ((model(x) - x) ** 2).mean(dim=(1, 2)).numpy()
    np.testing.assert_allclose(s[:5], manual, rtol=1e-5)


def test_training_lowers_loss_and_leaves_data_untouched():
    x = _data(1_000)
    before = x.copy()
    windows = make_windows(x, W, 10)
    model = ConvAE(F, W)
    hist = train_ae(model, windows, epochs=5, batch_size=32, lr=1e-3,
                    val_windows=windows, device="cpu", verbose=False)
    assert [h["epoch"] for h in hist] == [1, 2, 3, 4, 5]
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert "val_loss" in hist[0]
    np.testing.assert_array_equal(x, before)


def test_training_is_reproducible():
    windows = make_windows(_data(400), W, 10)
    runs = []
    for _ in range(2):
        torch.manual_seed(0)
        m = ConvAE(F, W)
        runs.append(train_ae(m, windows, epochs=2, batch_size=8, device="cpu",
                             seed=1, verbose=False)[-1]["train_loss"])
    assert runs[0] == pytest.approx(runs[1])


def test_default_model_unchanged_by_dropout_option():
    keys = set(ConvAE(F, W).state_dict())
    assert keys == set(ConvAE(F, W, dropout=0.0).state_dict())
    with pytest.raises(ValueError):
        ConvAE(F, W, dropout=1.0)


def test_dropout_only_in_training_mode():
    m = ConvAE(F, W, dropout=0.5)
    x = torch.rand(4, W, F)
    m.eval()
    with torch.no_grad():
        assert torch.equal(m(x), m(x))
    m.train()
    with torch.no_grad():
        assert not torch.equal(m(x), m(x))


def test_corrupt_rates_and_shapes():
    torch.manual_seed(0)
    x = torch.ones(200, W, F)
    assert torch.equal(corrupt(x), x)
    m = corrupt(x, mask_prob=0.25)
    assert abs((m == 0).float().mean().item() - 0.25) < 0.02
    s = corrupt(x, sensor_mask_prob=0.25)
    zero_sensor = (s == 0).all(dim=1)            # whole 60 s of a sensor masked together
    assert abs(zero_sensor.float().mean().item() - 0.25) < 0.03
    assert ((s == 0) | (s == 1)).all()
    n = corrupt(x, noise_std=0.1)
    assert abs((n - x).std().item() - 0.1) < 0.01
    assert torch.equal(x, torch.ones(200, W, F))  # input untouched


@pytest.mark.parametrize("kwargs", [{"weight_decay": 1e-3}, {"noise_std": 0.05},
                                    {"mask_prob": 0.2}, {"sensor_mask_prob": 0.2}])
def test_regularized_training_runs_and_learns(kwargs):
    x = _data(1_000)
    windows = make_windows(x, W, 10)
    hist = train_ae(ConvAE(F, W, dropout=0.1), windows, epochs=4, batch_size=32,
                    device="cpu", verbose=False, **kwargs)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]


def test_group_by_stage_merges_small_stages():
    from src.models import group_by_stage
    names = ["FIT101", "LIT101", "P101", "AIT201", "FIT201", "P201", "FIT601", "P602"]
    assert group_by_stage(names) == [[0, 1, 2], [3, 4, 5, 6, 7]]
    with pytest.raises(ValueError):
        group_by_stage(["FIT", "LIT101"])


def test_stage_model_keeps_stages_separate():
    from src.models import PerStageConvAE
    groups = [[0, 2, 4], [1, 3, 5, 6]]
    m = PerStageConvAE(groups, window=W).eval()
    x = torch.rand(2, W, 7)
    with torch.no_grad():
        base = m(x)
        x2 = x.clone()
        x2[:, :, 1] += 5.0                      # change a sensor of the second group only
        out = m(x2)
    assert out.shape == x.shape
    assert torch.allclose(out[:, :, [0, 2, 4]], base[:, :, [0, 2, 4]])   # first group unaffected
    assert not torch.allclose(out[:, :, [1, 3, 5, 6]], base[:, :, [1, 3, 5, 6]])
    with pytest.raises(ValueError):
        PerStageConvAE([[0, 1], [1, 2]])


def test_stage_model_trains_and_scores():
    from src.models import PerStageConvAE, sensor_errors
    x = _data(800)
    windows = make_windows(x, W, 10)
    groups = [list(range(20)), list(range(20, F))]
    m = PerStageConvAE(groups, window=W)
    hist = train_ae(m, windows, epochs=3, batch_size=32, device="cpu", verbose=False)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert sensor_errors(m, make_windows(x, W, 1)[:50], device="cpu").shape == (50, F)
