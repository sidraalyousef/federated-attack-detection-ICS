import numpy as np
import pandas as pd
import pytest

from src.data_loaders.clients import (
    CLIENT_STARTS,
    check_bounds,
    client_windows,
    points_from_clients,
    split_clients,
)


def _fake_prepared(n_attack_rows=449_919):
    """Same shape and timestamps as the real SWaT output; attacks placed
    away from the client cuts."""
    ts = pd.date_range("2015-12-28 10:00:00", periods=n_attack_rows, freq="s").to_numpy()
    labels = np.zeros(n_attack_rows, dtype=np.int8)
    labels[1_000:1_500] = 1      # in C1
    labels[150_000:150_300] = 1  # in C2
    return {
        "train": np.arange(1_000, dtype=np.float32).reshape(-1, 1),
        "val": np.arange(200, dtype=np.float32).reshape(-1, 1),
        "attack": np.zeros((n_attack_rows, 1), dtype=np.float32),
        "attack_labels": labels,
        "train_timestamps": np.arange(1_000),
        "val_timestamps": np.arange(200),
        "attack_timestamps": ts,
    }


@pytest.mark.parametrize("n", sorted(CLIENT_STARTS))
def test_pieces_join_back_to_centralized_data(n):
    d = _fake_prepared()
    clients = split_clients(d, n)
    assert [c["name"] for c in clients] == [f"C{k}" for k in range(1, n + 1)]
    for key in ("train", "val", "attack_labels"):
        assert np.array_equal(np.concatenate([c[key] for c in clients]), d[key])


def test_five_client_split_has_attack_free_c4():
    clients = split_clients(_fake_prepared(), 5)
    assert clients[3]["attack_labels"].sum() == 0


def test_unknown_client_count_rejected():
    with pytest.raises(ValueError):
        split_clients(_fake_prepared(), 4)


def test_check_bounds_rejects_cut_inside_attack():
    y = np.array([0, 0, 1, 1, 0, 0])
    with pytest.raises(ValueError):
        check_bounds(y, [(0, 3), (3, 6)], min_quiet=0)


def test_check_bounds_rejects_short_recovery():
    y = np.array([0, 1, 0, 0, 0, 0])
    check_bounds(y, [(0, 6)], min_quiet=4)
    with pytest.raises(ValueError):
        check_bounds(y, [(0, 4), (4, 6)], min_quiet=4)


def test_windows_stay_inside_client_and_eval_stride_is_one():
    c = {"train": np.zeros((100, 2)), "val": np.zeros((80, 2)), "attack": np.zeros((70, 2))}
    w = client_windows(c, window=10)
    assert w["train"].shape == (10, 10, 2)   # stride 10
    assert w["val"].shape == (71, 10, 2)     # stride 1
    assert w["attack"].shape == (61, 10, 2)


def test_points_from_clients_aligns_scores_labels_groups():
    clients = [
        {"name": "C1", "attack": np.zeros((20, 1)), "attack_labels": np.zeros(20)},
        {"name": "C2", "attack": np.zeros((15, 1)), "attack_labels": np.ones(15)},
    ]
    scores, labels, groups = points_from_clients([np.ones(11), np.ones(6)], clients, window=10)
    assert scores.shape == labels.shape == groups.shape == (35,)
    assert np.isnan(scores[:9]).all() and np.isnan(scores[20:29]).all()  # warm-up per client
    assert (groups[:20] == "C1").all() and (groups[20:] == "C2").all()
