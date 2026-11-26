"""Federated averaging (FedAvg) with the placeholder autoencoder, run in-process.

Two runners share the same Flower ``NumPyClient`` (``AEClient``):

* ``run_fedavg``: a plain in-process loop with Flower's own weighted average
  (``aggregate``). Works anywhere flwr and torch are installed. Used because
  Flower's simulation engine needs Ray, which has no build for Python 3.14 on
  Windows.
* ``run_flower_simulation``: the real Flower simulation (``run_simulation``,
  ``FedAvg`` strategy). Needs ``pip install "flwr[simulation]"``; run it on
  Linux / WSL2, where Ray is supported.

Only normal training windows are used for training; the attack windows are
never touched here. ``make_model`` is the one place to change when the real
model is chosen.

    python -m src.federated.fedavg --split time --clients 5 --rounds 5
    python -m src.federated.fedavg --split time --clients 5 --rounds 5 --simulation
"""
from __future__ import annotations

import argparse

import numpy as np
from flwr.client import NumPyClient
from flwr.server.strategy.aggregate import aggregate

from src.data_loaders.clients import split_clients, split_clients_by_stage
from src.data_loaders.preprocessing import EVAL_STRIDE, TRAIN_STRIDE, make_windows, prepare_swat
from src.models import (LinearAE, StageAE, get_parameters, local_keys,
                        score_windows, set_parameters, shared_keys, train_ae)

WINDOW = 60


def make_model(n_features: int, split: str = "time"):
    """Placeholder model. Swap the body when the final architecture is decided."""
    if split == "stage":
        return StageAE(n_features, window=WINDOW)
    return LinearAE(n_features=n_features, window=WINDOW, latent=32)


class AEClient(NumPyClient):
    """One federated client. Uses its normal train windows and val windows only."""

    def __init__(self, client, split: str = "time"):
        self.name = client["name"]
        # Only normal data is cut into windows; the attack data is never touched.
        self.train = make_windows(client["train"], WINDOW, TRAIN_STRIDE)
        self.val = make_windows(client["val"], WINDOW, EVAL_STRIDE)
        self.model = make_model(self.train.shape[2], split)
        self.keys = shared_keys(self.model)

    def get_parameters(self, config):
        return get_parameters(self.model, self.keys)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters, self.keys)
        train_ae(self.model, self.train, epochs=config.get("local_epochs", 1),
                 seed=config.get("server-round", config.get("server_round", 0)), device="cpu", verbose=False)
        return get_parameters(self.model, self.keys), len(self.train), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters, self.keys)
        s = score_windows(self.model, self.val, device="cpu")
        return float(s.mean()), len(self.val), {}


def run_fedavg(clients, split: str = "time", rounds: int = 5, local_epochs: int = 1,
               verbose: bool = True):
    """FedAvg over ``clients`` (output of ``split_clients`` / ``split_clients_by_stage``).

    Returns ``(global_params, history, workers)``. ``global_params`` is the final
    averaged list of arrays. ``history`` has one dict per round: ``round`` and
    ``val_loss`` (mean validation reconstruction error over clients, weighted by
    val windows). ``workers`` are the ``AEClient`` objects with the final global
    weights loaded, each model also holding the client's own local layers (stage
    split), ready to score val and attack windows.
    """
    workers = [AEClient(c, split) for c in clients]
    # Every client starts from the same weights.
    global_params = workers[0].get_parameters({})
    history = []
    for rnd in range(1, rounds + 1):
        cfg = {"local_epochs": local_epochs, "server-round": rnd}
        results = []
        for wk in workers:
            params, n, _ = wk.fit(global_params, cfg)
            results.append((params, n))
        global_params = aggregate(results)
        losses = [wk.evaluate(global_params, cfg)[:2] for wk in workers]
        val_loss = sum(l * n for l, n in losses) / sum(n for _, n in losses)
        history.append({"round": rnd, "val_loss": val_loss})
        if verbose:
            print(f"round {rnd}/{rounds}  val_loss {val_loss:.6f}", flush=True)
    for wk in workers:
        set_parameters(wk.model, global_params, wk.keys)
    return global_params, history, workers


def run_flower_simulation(clients, split: str = "time", rounds: int = 5,
                          local_epochs: int = 1, out: str = "global_params.npz"):
    """Real Flower simulation (Ray backend). Saves the final global arrays to ``out``.

    Simulated clients are recreated every round, so a stage client's own layers
    are kept in ``context.state`` between rounds (empty for the time split).
    """
    from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
    from flwr.clientapp import ClientApp
    from flwr.serverapp import ServerApp
    from flwr.serverapp.strategy import FedAvg
    from flwr.simulation import run_simulation

    data = [{k: c[k] for k in ("name", "train", "val")} for c in clients]

    class _SimClient(AEClient):
        def __init__(self, client, split, context):
            super().__init__(client, split)
            self.context = context
            saved = context.state.get("local")
            if saved is not None:
                set_parameters(self.model, saved.to_numpy_ndarrays(), local_keys(self.model))

        def fit(self, parameters, config):
            res = super().fit(parameters, config)
            lk = local_keys(self.model)
            if lk:
                self.context.state["local"] = ArrayRecord.from_numpy_ndarrays(
                    get_parameters(self.model, lk))
            return res

    def make_client(context):
        return _SimClient(data[int(context.node_config["partition-id"])], split, context)

    client_app = ClientApp()

    @client_app.train()
    def train(msg, context):
        params, n, _ = make_client(context).fit(
            msg.content["arrays"].to_numpy_ndarrays(), dict(msg.content["config"]))
        content = RecordDict({"arrays": ArrayRecord.from_numpy_ndarrays(params),
                              "metrics": MetricRecord({"num-examples": n})})
        return Message(content, reply_to=msg)

    @client_app.evaluate()
    def evaluate(msg, context):
        loss, n, _ = make_client(context).evaluate(
            msg.content["arrays"].to_numpy_ndarrays(), dict(msg.content["config"]))
        content = RecordDict({"metrics": MetricRecord({"eval_loss": loss, "num-examples": n})})
        return Message(content, reply_to=msg)

    server = ServerApp()

    @server.main()
    def main(grid, context):
        m = make_model(data[0]["train"].shape[1], split)
        start = ArrayRecord.from_numpy_ndarrays(get_parameters(m, shared_keys(m)))
        result = FedAvg().start(grid=grid, initial_arrays=start, num_rounds=rounds,
                                train_config=ConfigRecord({"local_epochs": local_epochs}))
        print(result.evaluate_metrics_clientapp, flush=True)
        np.savez(out, *result.arrays.to_numpy_ndarrays())

    run_simulation(server_app=server, client_app=client_app,
                   num_supernodes=len(data))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["time", "stage"], default="time")
    ap.add_argument("--clients", type=int, default=5, help="time split only: 5 or 3")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--data", default="data/raw/swat")
    ap.add_argument("--simulation", action="store_true", help="real Flower simulation (needs Ray)")
    a = ap.parse_args()
    d = prepare_swat(a.data)
    cl = split_clients_by_stage(d) if a.split == "stage" else split_clients(d, a.clients)
    run = run_flower_simulation if a.simulation else run_fedavg
    run(cl, a.split, a.rounds, a.local_epochs)
