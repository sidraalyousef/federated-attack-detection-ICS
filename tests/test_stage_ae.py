from itertools import pairwise

import numpy as np
import pytest
import torch

from src.data_loaders.preprocessing import make_windows
from src.models import (
    LinearAE,
    StageAE,
    get_parameters,
    local_keys,
    score_windows,
    set_parameters,
    shared_keys,
    train_ae,
)

W = 60
STAGE_SIZES = (5, 11, 9)          # different number of sensors per stage client


def _plant(n=1_500, seed=0):
    """Whole plant: 25 sensors driven by 3 shared slow signals, plus a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)[:, None]
    base = np.hstack([np.sin(t / 15), np.cos(t / 40), np.sin(t / 90)])
    mix = np.random.default_rng(99).random((3, sum(STAGE_SIZES)))
    x = 0.5 + 0.15 * base @ mix / 3 + 0.01 * rng.standard_normal((n, sum(STAGE_SIZES)))
    return x.astype(np.float32)


def _stages(x):
    """Column blocks of the plant, like split_clients_by_stage."""
    edges = np.cumsum((0,) + STAGE_SIZES)
    return [x[:, a:b] for a, b in pairwise(edges)]


def _fedavg(arrays_per_client, weights):
    w = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    return [sum(wk * a[i] for wk, a in zip(w, arrays_per_client))
            for i in range(len(arrays_per_client[0]))]


@pytest.mark.parametrize("activation", [None, "relu", "tanh"])
def test_output_has_input_shape(activation):
    for f in STAGE_SIZES:
        assert StageAE(f, W, activation=activation)(torch.rand(4, W, f)).shape == (4, W, f)


def test_rejects_wrong_layout_and_bad_config():
    with pytest.raises(ValueError):
        StageAE(5, W)(torch.rand(2, 5, W))
    with pytest.raises(ValueError):
        StageAE(5, W, activation="sigmoid")
    with pytest.raises(ValueError):
        StageAE(5, W, latent=0)


def test_shared_part_has_the_same_shapes_on_every_stage():
    models = [StageAE(f, W) for f in STAGE_SIZES]
    keys = shared_keys(models[0])
    assert keys and all(k.startswith("core.") for k in keys)
    assert all(shared_keys(m) == keys for m in models)
    shapes = [[p.shape for p in get_parameters(m, keys)] for m in models]
    assert shapes[0] == shapes[1] == shapes[2]
    # the local part is exactly the sensor-specific layers, and differs in shape
    assert all(k.startswith(("adapter_in.", "adapter_out.")) for k in local_keys(models[0]))
    assert set(keys) | set(local_keys(models[0])) == set(models[0].state_dict())
    assert models[0].adapter_in.weight.shape != models[1].adapter_in.weight.shape


def test_fully_shared_models_have_no_local_keys():
    m = LinearAE(7, W, latent=4)
    assert local_keys(m) == []
    assert shared_keys(m) == list(m.state_dict())


def test_loading_shared_part_keeps_local_layers():
    a, b = StageAE(5, W), StageAE(11, W)
    keys = shared_keys(a)
    local_before = get_parameters(b, local_keys(b))
    set_parameters(b, get_parameters(a, keys), keys)
    for x, y in zip(get_parameters(a, keys), get_parameters(b, keys)):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(local_before, get_parameters(b, local_keys(b))):
        np.testing.assert_array_equal(x, y)


def test_set_parameters_checks_keys_and_shapes():
    a, b = StageAE(5, W), StageAE(11, W)
    with pytest.raises(ValueError):           # local layers of another stage do not fit
        set_parameters(b, get_parameters(a, local_keys(a)), local_keys(a))
    with pytest.raises(ValueError):
        set_parameters(b, get_parameters(a), None)      # full model of another stage
    with pytest.raises(ValueError):
        get_parameters(a, ["no.such.key"])
    keys = shared_keys(a)
    with pytest.raises(ValueError):
        set_parameters(b, get_parameters(a, keys)[:-1], keys)


def test_scores_and_training_work_unchanged():
    x = _stages(_plant())[1]
    windows = make_windows(x, W, 5)
    model = StageAE(x.shape[1], W)
    hist = train_ae(model, windows, epochs=5, batch_size=32, lr=1e-2,
                    device="cpu", verbose=False)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    s = score_windows(model, make_windows(x, W, 1), device="cpu")
    assert s.shape == (len(x) - W + 1,) and np.isfinite(s).all()


def test_partial_fedavg_rounds_train_every_stage():
    """A few FedAvg rounds as Flower would run them, without Flower."""
    train = [make_windows(c, W, 5) for c in _stages(_plant(seed=0))]
    torch.manual_seed(0)
    models = [StageAE(f, W) for f in STAGE_SIZES]
    keys = shared_keys(models[0])
    global_core = get_parameters(models[0], keys)      # server's initial parameters

    val = [make_windows(c, W, 1) for c in _stages(_plant(600, seed=1))]
    start = [score_windows(m, v, device="cpu").mean() for m, v in zip(models, val)]
    for rnd in range(8):
        sent = []
        for m, w in zip(models, train):
            set_parameters(m, global_core, keys)                       # fit()
            train_ae(m, w, epochs=2, batch_size=32, lr=1e-2, seed=rnd,
                     device="cpu", verbose=False)
            sent.append(get_parameters(m, keys))
        global_core = _fedavg(sent, [len(w) for w in train])            # server
    for m in models:
        set_parameters(m, global_core, keys)
    end = [score_windows(m, v, device="cpu").mean() for m, v in zip(models, val)]

    # every stage ends with the same core and a much better reconstruction
    for m in models[1:]:
        for x, y in zip(get_parameters(models[0], keys), get_parameters(m, keys)):
            np.testing.assert_array_equal(x, y)
    assert all(e < 0.2 * s for s, e in zip(start, end))

    # an attack on stage 1's sensors is visible in stage 1's scores
    test = _stages(_plant(600, seed=1))[0].copy()
    normal = score_windows(models[0], make_windows(test, W, 1), device="cpu")
    test[300:400, :2] += 0.5
    attacked = score_windows(models[0], make_windows(test, W, 1), device="cpu")
    assert attacked[300:341].min() > 3 * normal.max()
