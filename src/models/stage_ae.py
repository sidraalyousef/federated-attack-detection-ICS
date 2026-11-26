"""Autoencoder for the by-stage split: local sensor layers + a shared core.

In ``split_clients_by_stage`` every client (S1, S2, S3, S4, S5+6) has a
different number of sensors, so a model's input and output layers have a
different shape on every client and FedAvg cannot average them. This model
splits into two parts:

* **local** ``adapter_in`` / ``adapter_out``: per-second linear maps between
  the client's own sensors and a small ``step_dim`` embedding. Their shape
  depends on the number of sensors, so they stay on the client.
* **shared** ``core``: a tiny (linear or one-hidden-layer) autoencoder over the
  ``(window, step_dim)`` embedding sequence, the same shape on every client.
  Only this part goes to the server, and plain FedAvg averages it.

This is "partial" federated averaging (as in FedPer / LG-FedAvg): the clients
jointly learn how a normal 60-second stretch of the plant evolves in the
shared embedding space, while each stage keeps its own mapping to its sensors.

    x (B, W, F_k) --adapter_in--> (B, W, d) --core--> (B, W, d) --adapter_out--> (B, W, F_k)
       local                         shared                        local

With ``activation=None`` the whole model is linear (rank at most ``latent``,
PCA-like), like ``LinearAE``. Input/output shapes, training (``train_ae``) and
scoring (``score_windows``, ``sensor_errors``) are the same as for the other
models. Flower uses :func:`src.models.params.shared_keys`:

    model = StageAE(n_features=client["train"].shape[1])
    keys = shared_keys(model)                    # core.* only
    arrays = get_parameters(model, keys)         # what the client sends
    set_parameters(model, global_arrays, keys)   # local adapters untouched

The local adapters must survive between rounds on the client (Flower
simulation clients are recreated every round): save
``get_parameters(model, local_keys(model))`` and load it back with
``set_parameters(model, arrays, local_keys(model))``.
"""
from __future__ import annotations

import torch
from torch import nn

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class StageAE(nn.Module):
    """Per-stage ``adapter_in`` / ``adapter_out`` around a shared ``core`` autoencoder."""

    local_prefixes = ("adapter_in", "adapter_out")

    def __init__(self, n_features: int, window: int = 60, step_dim: int = 8,
                 latent: int = 16, activation: str | None = None):
        super().__init__()
        if step_dim < 1 or latent < 1:
            raise ValueError("step_dim and latent must be at least 1")
        if activation is not None and activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be None or one of {sorted(_ACTIVATIONS)}")
        self.n_features = n_features
        self.window = window
        self.step_dim = step_dim
        self.activation = activation
        self.adapter_in = nn.Linear(n_features, step_dim)    # local, per second
        enc = [nn.Linear(window * step_dim, latent)]
        if activation is not None:
            enc.append(_ACTIVATIONS[activation]())
        self.core = nn.Sequential(*enc, nn.Linear(latent, window * step_dim))  # shared
        self.adapter_out = nn.Linear(step_dim, n_features)   # local, per second

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, window, n_features)`` -> reconstruction of the same shape."""
        if x.dim() != 3 or x.shape[1:] != (self.window, self.n_features):
            raise ValueError(
                f"expected (batch, {self.window}, {self.n_features}), got {tuple(x.shape)}"
            )
        b = x.shape[0]
        e = self.adapter_in(x)                                    # (B, W, d)
        e = self.core(e.reshape(b, -1)).reshape(b, self.window, self.step_dim)
        return self.adapter_out(e)                                # (B, W, F)
