"""Tiny linear / one-hidden-layer autoencoder: the baseline model.

It flattens a window ``(window, sensors)`` into one vector, squeezes it into
``latent`` numbers and maps it back. With ``activation=None`` (the default)
the whole model is linear, and a linear autoencoder trained with MSE learns
the same subspace as PCA with ``latent`` components. Unlike PCA it is a plain
``nn.Module`` trained by gradient descent, so FedAvg can average it like any
other model. With ``activation="relu"`` or ``"tanh"`` the latent layer becomes
a non-linear hidden layer (a one-hidden-layer autoencoder).

It is also the stand-in model for the Flower simulation: it has the same
interface as ``ConvAE`` and ``LSTMAE``, so the real model can be swapped in
later without changing anything around it:

* train on normal windows only: :func:`src.models.conv_ae.train_ae`;
* anomaly score = mean squared reconstruction error per window:
  :func:`src.models.conv_ae.score_windows` (and ``sensor_errors``);
* parameters to / from Flower as a list of NumPy arrays:
  :func:`src.models.params.get_parameters` / ``set_parameters`` (these work
  for any of the models, not only this one).

For the by-stage split (a different number of sensors per client) use
``StageAE`` instead: same idea, but with per-stage input/output layers.

Shapes
------
Input and output are ``(batch, window, sensors)`` = ``(B, 60, 43)``, like the
other models; ``forward`` flattens to ``(B, 60*43)`` and reshapes back. The
last layer is linear (no output activation), as in the other models.

    model = LinearAE(n_features=43, window=60, latent=32)
    history = train_ae(model, make_windows(d["train"]), epochs=20)
    val_scores = score_windows(model, make_windows(d["val"], stride=1))

    # Flower NumPyClient
    def get_parameters(self, config):
        return get_parameters(self.model)
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        train_ae(self.model, self.train_windows, epochs=1, verbose=False)
        return get_parameters(self.model), len(self.train_windows), {}
"""
from __future__ import annotations

import torch
from torch import nn

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class LinearAE(nn.Module):
    """Flatten -> Linear(window*sensors, latent) [-> activation] -> Linear(latent, window*sensors).

    ``activation=None``: fully linear, PCA-like. ``"relu"`` / ``"tanh"``: one
    non-linear hidden layer of size ``latent``.
    """

    def __init__(self, n_features: int = 43, window: int = 60, latent: int = 32,
                 activation: str | None = None):
        super().__init__()
        if latent < 1:
            raise ValueError("latent must be at least 1")
        if activation is not None and activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be None or one of {sorted(_ACTIVATIONS)}")
        self.n_features = n_features
        self.window = window
        self.activation = activation
        dim = window * n_features
        enc = [nn.Linear(dim, latent)]
        if activation is not None:
            enc.append(_ACTIVATIONS[activation]())
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Linear(latent, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, window, n_features)`` -> reconstruction of the same shape."""
        if x.dim() != 3 or x.shape[1:] != (self.window, self.n_features):
            raise ValueError(
                f"expected (batch, {self.window}, {self.n_features}), got {tuple(x.shape)}"
            )
        z = self.encoder(x.reshape(x.shape[0], -1))
        return self.decoder(z).reshape(x.shape)
