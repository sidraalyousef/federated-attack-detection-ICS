"""1-D convolutional autoencoder for SWaT windows.

The model reconstructs a window of sensor readings; the anomaly score of a
window is its mean squared reconstruction error.

Shapes
------
Windows from ``make_windows`` are ``(batch, window, sensors)`` = ``(B, 60, 43)``.
``nn.Conv1d`` wants ``(batch, channels, time)``, so ``forward`` permutes to
``(B, 43, 60)`` on the way in and back to ``(B, 60, 43)`` on the way out. The
output therefore has the same shape as the input, and the loss is a plain MSE.

Batches
-------
``make_windows`` returns read-only views that share memory with the 2-D array.
:func:`train_ae` and :func:`score_windows` copy each batch with
``torch.tensor(...)`` (not ``torch.from_numpy``), so only one batch at a time
is materialised and PyTorch never touches the read-only view.

These helpers take any array of windows, so the same code trains the
centralized model, a local-only client model, or one FedAvg round on a client.

    model = ConvAE(n_features=43, window=60)
    history = train_ae(model, make_windows(d["train"]), epochs=20)
    val_scores = score_windows(model, make_windows(d["val"], stride=1))
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


class ConvAE(nn.Module):
    """Conv1d encoder (two stride-2 downsamplings) + ConvTranspose1d decoder.

    Time axis: ``window -> window/2 -> window/4 -> window/2 -> window``, so
    ``window`` must be divisible by 4 (60 is). The last layer is linear: scaled
    test values can leave the 0-1 range, and the model should be free to
    reconstruct (or fail to reconstruct) them.
    """

    def __init__(self, n_features: int = 43, window: int = 60,
                 channels: tuple[int, int] = (64, 32), latent: int = 16,
                 kernel: int = 5):
        super().__init__()
        if window % 4:
            raise ValueError(f"window must be divisible by 4, got {window}")
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd so 'same' padding keeps the length")
        c1, c2 = channels
        pad = kernel // 2
        self.n_features = n_features
        self.window = window
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, c1, kernel, padding=pad), nn.ReLU(),
            nn.Conv1d(c1, c2, kernel, stride=2, padding=pad), nn.ReLU(),     # W/2
            nn.Conv1d(c2, latent, kernel, stride=2, padding=pad), nn.ReLU(),  # W/4
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent, c2, 4, stride=2, padding=1), nn.ReLU(),  # W/2
            nn.ConvTranspose1d(c2, c1, 4, stride=2, padding=1), nn.ReLU(),      # W
            nn.Conv1d(c1, n_features, kernel, padding=pad),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, window, n_features)`` -> reconstruction of the same shape."""
        if x.dim() != 3 or x.shape[1:] != (self.window, self.n_features):
            raise ValueError(
                f"expected (batch, {self.window}, {self.n_features}), got {tuple(x.shape)}"
            )
        z = self.encoder(x.permute(0, 2, 1))   # (B, F, W) for Conv1d
        return self.decoder(z).permute(0, 2, 1)  # back to (B, W, F)


def _device(device):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batch(windows: np.ndarray, idx, device) -> torch.Tensor:
    # torch.tensor copies: the windows are a read-only view of the 2-D data.
    return torch.tensor(windows[idx], dtype=torch.float32, device=device)


def train_ae(model: nn.Module, train_windows: np.ndarray, epochs: int = 20,
             batch_size: int = 256, lr: float = 1e-3,
             val_windows: np.ndarray | None = None, device=None, seed: int = 0,
             verbose: bool = True) -> list[dict]:
    """Train ``model`` to reconstruct ``train_windows`` (normal data only).

    Returns one dict per epoch: ``epoch``, ``train_loss`` and, if
    ``val_windows`` is given, ``val_loss`` (mean window MSE; for monitoring
    only, nothing is selected on it). Shuffling uses ``seed``.
    """
    device = _device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = len(train_windows)
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = np.sort(order[start:start + batch_size])  # sorted: faster gather
            x = _batch(train_windows, idx, device)
            opt.zero_grad()
            loss = loss_fn(model(x), x)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        row = {"epoch": epoch, "train_loss": total / n}
        if val_windows is not None:
            row["val_loss"] = float(score_windows(model, val_windows, device=device).mean())
        row["seconds"] = round(time.perf_counter() - t0, 1)
        history.append(row)
        if verbose:
            val = f"  val {row['val_loss']:.6f}" if "val_loss" in row else ""
            print(f"epoch {epoch:3d}  train {row['train_loss']:.6f}{val}  ({row['seconds']} s)",
                  flush=True)
    return history


@torch.no_grad()
def score_windows(model: nn.Module, windows: np.ndarray, batch_size: int = 2048,
                  device=None) -> np.ndarray:
    """Mean squared reconstruction error of each window, shape ``(len(windows),)``.

    Feed the result to ``window_scores_to_points`` (or ``points_from_clients``).
    Score validation and test windows cut with stride 1.
    """
    device = _device(device)
    model.to(device).eval()
    out = np.empty(len(windows), dtype=np.float64)
    for start in range(0, len(windows), batch_size):
        sl = slice(start, start + batch_size)
        x = _batch(windows, sl, device)
        out[sl] = ((model(x) - x) ** 2).mean(dim=(1, 2)).cpu().numpy()
    return out
