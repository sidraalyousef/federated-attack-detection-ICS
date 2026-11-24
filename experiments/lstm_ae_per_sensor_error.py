"""Which sensors drive the LSTM-AE reconstruction error on one client's test slice?

Loads weights saved by ``lstm_ae_centralized`` and prints, per sensor, the mean
squared reconstruction error on the validation data and on a client's attack-file
windows, plus the ratio. Run it on the attack-free client (C4) to find sensors
that drift away from their normal range and raise false alarms.

    python -m experiments.lstm_ae_per_sensor_error                       # default run, C4
    python -m experiments.lstm_ae_per_sensor_error --run-dir experiments/lstm_ae_centralized/outputs/drop_drifting --drop-drifting
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.lstm_ae_centralized import DRIFTING, OUT_DIR
from src.data_loaders.clients import client_windows, split_clients
from src.data_loaders.preprocessing import EXCLUDE, WINDOW, make_windows, prepare_swat
from src.models import LSTMAE


@torch.no_grad()
def per_sensor_mse(model, windows, batch_size=4096):
    """Mean squared error of every sensor, averaged over windows and time."""
    total, n = 0.0, 0
    for i in range(0, len(windows), batch_size):
        x = torch.tensor(windows[i:i + batch_size])
        total = total + ((model(x) - x) ** 2).mean(dim=1).sum(0).numpy()
        n += len(x)
    return total / n


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw-dir", default="data/raw/swat")
    p.add_argument("--run-dir", type=Path, default=OUT_DIR, help="folder with lstm_ae.pt and config.json")
    p.add_argument("--client", type=int, default=4, help="client number (1-based)")
    p.add_argument("--n-clients", type=int, default=5, choices=(3, 5))
    p.add_argument("--drop-drifting", action="store_true", help="the run dropped the drifting analysers")
    p.add_argument("--top", type=int, default=8)
    args = p.parse_args()

    exclude = (*EXCLUDE, *(DRIFTING if args.drop_drifting else ()))
    d = prepare_swat(args.raw_dir, exclude=exclude)
    model = LSTMAE(len(d["features"]), WINDOW)
    model.load_state_dict(torch.load(args.run_dir / "lstm_ae.pt"))
    model.eval()

    client = split_clients(d, n_clients=args.n_clients)[args.client - 1]
    val = per_sensor_mse(model, make_windows(d["val"], stride=10))
    test = per_sensor_mse(model, client_windows(client)["attack"][::10])
    table = pd.DataFrame({"val": val, "test": test}, index=d["features"])
    table["ratio"] = table["test"] / table["val"]
    top = table.sort_values("test", ascending=False)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(top.head(args.top))
    share = top["test"].head(3).sum() / table["test"].sum()
    print(f"\ntop 3 sensors = {share:.0%} of the test error on client {args.client}")


if __name__ == "__main__":
    main()
