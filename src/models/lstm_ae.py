"""LSTM autoencoder for SWaT windows.

The model reconstructs a window of sensor readings; the anomaly score of a
window is its mean squared reconstruction error (same score as ``ConvAE``).

Architecture
------------
Encoder: an LSTM reads the window one second at a time; its last hidden state
summarises the whole window and is squeezed into a ``latent`` vector.
Decoder: the latent vector is expanded and fed to a second LSTM at every
time step; a linear layer maps each step back to the sensors. The latent
vector is the only thing the decoder sees, so the model cannot copy the input
and has to learn what normal windows look like.

Shapes
------
Windows from ``make_windows`` are ``(batch, window, sensors)`` = ``(B, 60, 43)``.
``nn.LSTM`` is used with ``batch_first=True``, so no permutes are needed and
the output has the same shape as the input. The loss is a plain MSE. The last
layer is linear: scaled test values can leave the 0-1 range, and the model
should be free to reconstruct (or fail to reconstruct) them.

Training and scoring reuse :func:`src.models.conv_ae.train_ae` and
:func:`src.models.conv_ae.score_windows`, which accept any ``nn.Module`` that
maps ``(B, window, sensors)`` to the same shape. The same helpers therefore
train the centralized model, a local-only client model, or one FedAvg round.

    model = LSTMAE(n_features=43, window=60)
    history = train_ae(model, make_windows(d["train"]), epochs=20)
    val_scores = score_windows(model, make_windows(d["val"], stride=1))
"""
from __future__ import annotations

import torch
from torch import nn


class LSTMAE(nn.Module):
    """LSTM encoder -> latent vector -> LSTM decoder."""

    def __init__(self, n_features: int = 43, window: int = 60, hidden: int = 64,
                 latent: int = 16, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        # nn.LSTM only applies dropout between stacked layers.
        dropout = dropout if num_layers > 1 else 0.0
        self.n_features = n_features
        self.window = window
        self.encoder = nn.LSTM(n_features, hidden, num_layers=num_layers,
                               batch_first=True, dropout=dropout)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, num_layers=num_layers,
                               batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, window, n_features)`` -> reconstruction of the same shape."""
        if x.dim() != 3 or x.shape[1:] != (self.window, self.n_features):
            raise ValueError(
                f"expected (batch, {self.window}, {self.n_features}), got {tuple(x.shape)}"
            )
        _, (h, _) = self.encoder(x)              # h: (layers, B, hidden)
        z = self.to_latent(h[-1])                # (B, latent)
        step = torch.relu(self.from_latent(z))   # (B, hidden)
        seq = step.unsqueeze(1).expand(-1, self.window, -1)  # same input at every step
        dec, _ = self.decoder(seq)               # (B, W, hidden)
        return self.out(dec)                     # (B, W, F)
