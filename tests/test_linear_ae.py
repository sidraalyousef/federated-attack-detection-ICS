import numpy as np
import pytest
import torch

from src.data_loaders.preprocessing import make_windows
from src.eval import threshold_from_normal, window_scores_to_points
from src.models import LinearAE, get_parameters, score_windows, set_parameters, train_ae

W, F = 60, 43


def _normal(n=1_500, seed=0):
    """Low-rank 'plant': every sensor is a mix of 3 slow signals, plus a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)[:, None]
    base = np.hstack([np.sin(t / 15), np.cos(t / 40), np.sin(t / 90)])  # (n, 3)
    mix = rng.random((3, F))
    x = 0.5 + 0.15 * base @ mix / 3 + 0.01 * rng.standard_normal((n, F))
    return x.astype(np.float32)


@pytest.mark.parametrize("activation", [None, "relu", "tanh"])
def test_output_has_input_shape(activation):
    model = LinearAE(F, W, latent=8, activation=activation)
    assert model(torch.rand(5, W, F)).shape == (5, W, F)


def test_linear_option_is_linear():
    # No hidden activation: f(a*x + b*y) == a*f(x) + b*f(y) once the biases are removed.
    model = LinearAE(F, W, latent=4).eval()
    x, y = torch.rand(1, W, F), torch.rand(1, W, F)
    f0 = model(torch.zeros(1, W, F))
    with torch.no_grad():
        lhs = model(2 * x + 3 * y) - f0
        rhs = 2 * (model(x) - f0) + 3 * (model(y) - f0)
    np.testing.assert_allclose(lhs.numpy(), rhs.numpy(), rtol=1e-4, atol=1e-5)


def test_rejects_wrong_layout_and_bad_config():
    with pytest.raises(ValueError):
        LinearAE(F, W)(torch.rand(2, F, W))
    with pytest.raises(ValueError):
        LinearAE(F, W, activation="sigmoid")
    with pytest.raises(ValueError):
        LinearAE(F, W, latent=0)


def test_scores_read_only_windows_one_per_window():
    x = _normal(300)
    windows = make_windows(x, W, 1)
    assert not windows.flags.writeable
    s = score_windows(LinearAE(F, W), windows, batch_size=50, device="cpu")
    assert s.shape == (len(x) - W + 1,)
    assert np.isfinite(s).all() and (s >= 0).all()


def test_training_lowers_loss_and_leaves_data_untouched():
    x = _normal()
    before = x.copy()
    windows = make_windows(x, W, 5)
    model = LinearAE(F, W, latent=8)
    hist = train_ae(model, windows, epochs=5, batch_size=32, lr=1e-3,
                    val_windows=windows, device="cpu", verbose=False)
    assert [h["epoch"] for h in hist] == [1, 2, 3, 4, 5]
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("activation", [None, "relu"])
def test_anomalous_windows_score_higher(activation):
    train = make_windows(_normal(seed=0), W, 5)
    model = LinearAE(F, W, latent=8, activation=activation)
    train_ae(model, train, epochs=20, batch_size=32, lr=1e-2, device="cpu", verbose=False)

    test = _normal(600, seed=1)
    attacked = test.copy()
    attacked[300:400, :5] += 0.5          # five sensors jump during the "attack"
    labels = np.zeros(len(test), dtype=int)
    labels[300:400] = 1

    normal_scores = score_windows(model, make_windows(test, W, 1), device="cpu")
    attack_scores = score_windows(model, make_windows(attacked, W, 1), device="cpu")
    # windows overlapping the attack get much larger reconstruction errors
    hit = slice(300, 400 - W + 1)
    assert attack_scores[hit].min() > 3 * normal_scores.max()

    # ... and the scores plug into the eval helpers unchanged
    points = window_scores_to_points(attack_scores, len(test), W, 1)
    thr = threshold_from_normal(normal_scores)
    assert points.shape == (len(test),)
    assert (points[labels == 1] > thr).mean() > 0.9


def test_parameters_round_trip():
    torch.manual_seed(0)
    a = LinearAE(F, W, latent=8)
    torch.manual_seed(1)
    b = LinearAE(F, W, latent=8)
    params = get_parameters(a)
    assert all(isinstance(p, np.ndarray) for p in params)
    assert [p.shape for p in params] == [v.shape for v in a.state_dict().values()]
    set_parameters(b, params)
    x = torch.rand(3, W, F)
    with torch.no_grad():
        np.testing.assert_array_equal(a(x).numpy(), b(x).numpy())
    # returned arrays are copies: changing them does not touch the model
    params[0][:] = 0
    assert not np.allclose(get_parameters(a)[0], 0)


def test_averaged_parameters_load_back():
    # What FedAvg does: element-wise (weighted) mean of the clients' arrays.
    torch.manual_seed(0)
    m1 = LinearAE(F, W, latent=8, activation="relu")
    torch.manual_seed(1)
    m2 = LinearAE(F, W, latent=8, activation="relu")
    p1, p2 = get_parameters(m1), get_parameters(m2)
    avg = [(0.25 * x + 0.75 * y) for x, y in zip(p1, p2)]
    glob = LinearAE(F, W, latent=8, activation="relu")
    set_parameters(glob, avg)
    for got, want in zip(get_parameters(glob), avg):
        np.testing.assert_allclose(got, want, rtol=1e-6)
    # float64 averages (e.g. from numpy) also load
    set_parameters(glob, [p.astype(np.float64) for p in avg])
    assert all(v.dtype == torch.float32 for v in glob.state_dict().values())


def test_set_parameters_rejects_wrong_architecture():
    params = get_parameters(LinearAE(F, W, latent=8))
    with pytest.raises(ValueError):
        set_parameters(LinearAE(F, W, latent=4), params)
    with pytest.raises(ValueError):
        set_parameters(LinearAE(F, W, latent=8), params[:-1])


def test_helpers_work_for_other_models():
    from src.models import ConvAE
    a, b = ConvAE(F, W), ConvAE(F, W)
    set_parameters(b, get_parameters(a))
    x = torch.rand(2, W, F)
    with torch.no_grad():
        np.testing.assert_array_equal(a.eval()(x).numpy(), b.eval()(x).numpy())
