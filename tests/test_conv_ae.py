import numpy as np
import pytest
import torch

from src.data_loaders.preprocessing import make_windows
from src.models import ConvAE, score_windows, train_ae

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
