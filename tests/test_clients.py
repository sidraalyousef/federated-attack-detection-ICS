import numpy as np
import pandas as pd
import pytest

from src.data_loaders.clients import (
    CLIENT_STARTS,
    check_bounds,
    client_windows,
    points_from_clients,
    sensor_stage,
    split_clients,
    split_clients_by_stage,
    summary,
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


# ---- split by process stage ------------------------------------------------

# One or two sensors per stage; stage 6 is small, as in the real data.
FEATURES = ["FIT101", "P101", "AIT201", "MV201", "FIT301", "LIT401", "AIT501", "PIT501", "FIT601", "P602"]


def _stage_data(n_attack=50, random=False):
    """Column j holds the value j (or random numbers), so each client's columns are easy to check."""
    rng = np.random.default_rng(0)

    def block(n):
        if random:
            return rng.random((n, len(FEATURES))).astype(np.float32)
        return np.tile(np.arange(len(FEATURES), dtype=np.float32), (n, 1))

    labels = np.zeros(n_attack, dtype=np.int8)
    labels[10:20] = 1
    return {
        "features": FEATURES,
        "train": block(30),
        "val": block(12),
        "attack": block(n_attack),
        "attack_labels": labels,
        "train_timestamps": np.arange(30),
        "val_timestamps": np.arange(12),
        "attack_timestamps": np.arange(n_attack),
    }


def test_sensor_stage_reads_the_first_digit():
    names = ("FIT101", "P102", "DPIT301", "UV401", "PIT503", "P602")
    assert [sensor_stage(f) for f in names] == [1, 1, 3, 4, 5, 6]
    for bad in ("Timestamp", "P0", "FIT701", "FIT001"):
        with pytest.raises(ValueError):
            sensor_stage(bad)


def test_default_stage_split_merges_stages_5_and_6():
    clients = split_clients_by_stage(_stage_data())
    assert [c["name"] for c in clients] == ["S1", "S2", "S3", "S4", "S5+6"]
    assert [c["stages"] for c in clients] == [(1,), (2,), (3,), (4,), (5, 6)]
    assert [c["features"] for c in clients] == [
        ["FIT101", "P101"], ["AIT201", "MV201"], ["FIT301"], ["LIT401"],
        ["AIT501", "PIT501", "FIT601", "P602"],
    ]


def test_stage_clients_hold_only_their_columns():
    clients = split_clients_by_stage(_stage_data())
    # the value stored in a column is its index in the full array
    assert [c["train"][0].tolist() for c in clients] == [[0, 1], [2, 3], [4], [5], [6, 7, 8, 9]]
    assert [c["attack"].shape[1] for c in clients] == [2, 2, 1, 1, 4]


def test_stage_clients_stack_back_to_the_centralized_data():
    d = _stage_data(random=True)
    clients = split_clients_by_stage(d)
    order = [FEATURES.index(f) for c in clients for f in c["features"]]
    assert sorted(order) == list(range(len(FEATURES)))     # every sensor exactly once
    for k in ("train", "val", "attack"):
        stacked = np.concatenate([c[k] for c in clients], axis=1)
        assert np.array_equal(stacked[:, np.argsort(order)], d[k])


def test_stage_clients_share_the_test_timeline_and_labels():
    d = _stage_data()
    clients = split_clients_by_stage(d)
    for c in clients:
        assert np.array_equal(c["attack_labels"], d["attack_labels"])
        assert c["attack_rows"] == (0, len(d["attack"]))
        assert len(c["train"]) == len(d["train"]) and len(c["val"]) == len(d["val"])
    table = summary(clients)                                # works unchanged
    assert (table["attacks"] == 1).all() and (table["attack_s"] == 10).all()


def test_stage_clients_work_with_client_windows_and_different_widths():
    clients = split_clients_by_stage(_stage_data(n_attack=40))
    for c in clients:
        w = client_windows(c, window=10)
        assert w["train"].shape == (3, 10, c["train"].shape[1])     # stride 10
        assert w["attack"].shape == (31, 10, c["attack"].shape[1])  # stride 1


def test_one_client_per_stage_option():
    clients = split_clients_by_stage(_stage_data(), groups=((1,), (2,), (3,), (4,), (5,), (6,)))
    assert [c["name"] for c in clients] == ["S1", "S2", "S3", "S4", "S5", "S6"]
    assert [c["train"].shape[1] for c in clients] == [2, 2, 1, 1, 2, 2]


def test_stage_groups_are_validated():
    d = _stage_data()
    with pytest.raises(ValueError, match="more than one group"):
        split_clients_by_stage(d, groups=((1, 2), (2, 3), (4,), (5, 6)))
    with pytest.raises(ValueError, match="not in any group"):
        split_clients_by_stage(d, groups=((1,), (2,), (3,), (4,), (5,)))     # stage 6 left out
    with pytest.raises(ValueError, match="no sensors"):
        split_clients_by_stage(d, groups=((1,), (2,), (3,), (4,), (5, 6), (7,)))
