import numpy as np
import pytest
import torch

from src.data_loaders.preprocessing import make_windows
from src.models import LSTMAE, score_windows, train_ae

W, F = 60, 43


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, F)).astype(np.float32)


def test_output_has_input_shape():
    model = LSTMAE(F, W)
    x = torch.rand(8, W, F)
    assert model(x).shape == (8, W, F)


def test_stacked_layers_keep_shape():
    model = LSTMAE(F, W, hidden=32, latent=8, num_layers=2, dropout=0.1)
    assert model(torch.rand(3, W, F)).shape == (3, W, F)


def test_rejects_wrong_layout():
    # (B, sensors, time) must not be passed
    with pytest.raises(ValueError):
        LSTMAE(F, W)(torch.rand(2, F, W))


def test_rejects_zero_layers():
    with pytest.raises(ValueError):
        LSTMAE(F, W, num_layers=0)


def test_latent_is_a_bottleneck():
    # The decoder only sees the latent vector: identical latents -> identical output,
    # whatever the rest of the window looked like.
    model = LSTMAE(F, W).eval()
    x = torch.rand(1, W, F)
    with torch.no_grad():
        _, (h, _) = model.encoder(x)
        z = model.to_latent(h[-1])
        step = torch.relu(model.from_latent(z)).unsqueeze(1).expand(-1, W, -1)
        manual = model.out(model.decoder(step)[0])
        np.testing.assert_allclose(model(x).numpy(), manual.numpy(), rtol=1e-5, atol=1e-6)


def test_scores_read_only_windows_one_per_window():
    x = _data()
    windows = make_windows(x, W, 1)
    assert not windows.flags.writeable
    s = score_windows(LSTMAE(F, W), windows, batch_size=100, device="cpu")
    assert s.shape == (len(x) - W + 1,)
    assert np.isfinite(s).all() and (s >= 0).all()


def test_score_matches_manual_mse():
    model = LSTMAE(F, W).eval()
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
    model = LSTMAE(F, W, hidden=32, latent=8)
    hist = train_ae(model, windows, epochs=5, batch_size=32, lr=1e-2,
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
        m = LSTMAE(F, W, hidden=16, latent=4)
        runs.append(train_ae(m, windows, epochs=2, batch_size=8, device="cpu",
                             seed=1, verbose=False)[-1]["train_loss"])
    assert runs[0] == pytest.approx(runs[1])
