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
                 kernel: int = 5, dropout: float = 0.0):
        """``dropout`` > 0 adds a Dropout layer after every ReLU (regularization;
        only active in training mode). With the default 0 the layers and the
        saved state_dict are exactly as before."""
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if window % 4:
            raise ValueError(f"window must be divisible by 4, got {window}")
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd so 'same' padding keeps the length")
        c1, c2 = channels
        pad = kernel // 2
        self.n_features = n_features
        self.window = window
        def act():
            return [nn.ReLU(), nn.Dropout(dropout)] if dropout > 0 else [nn.ReLU()]

        self.dropout = dropout
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, c1, kernel, padding=pad), *act(),
            nn.Conv1d(c1, c2, kernel, stride=2, padding=pad), *act(),     # W/2
            nn.Conv1d(c2, latent, kernel, stride=2, padding=pad), *act(),  # W/4
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent, c2, 4, stride=2, padding=1), *act(),  # W/2
            nn.ConvTranspose1d(c2, c1, 4, stride=2, padding=1), *act(),      # W
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


def group_by_stage(features, min_size: int = 3) -> list[list[int]]:
    """Column indices grouped by SWaT process stage (the first digit in the
    sensor name: FIT101 -> stage 1, AIT504 -> stage 5).

    A stage with fewer than ``min_size`` sensors is merged into the previous
    stage: an autoencoder on 1-2 sensors has no real bottleneck. On SWaT this
    merges stage 6 (FIT601, P602) into stage 5.
    """
    import re

    stages: dict[int, list[int]] = {}
    for i, name in enumerate(features):
        m = re.search(r"\d", str(name))
        if m is None:
            raise ValueError(f"no stage digit in sensor name {name!r}")
        stages.setdefault(int(m.group()), []).append(i)
    groups: list[list[int]] = []
    for k in sorted(stages):
        if groups and len(stages[k]) < min_size:
            groups[-1].extend(stages[k])
        else:
            groups.append(list(stages[k]))
    if len(groups) > 1 and len(groups[0]) < min_size:   # a small first stage joins the next
        groups[1] = groups[0] + groups[1]
        groups.pop(0)
    return groups


class StageConvAE(nn.Module):
    """One small ConvAE per group of sensors (e.g. per plant stage).

    Same interface as :class:`ConvAE`: ``(B, window, n_features)`` in and out,
    so ``train_ae``, ``sensor_errors``, ``score_windows`` and ``stream_scores``
    work unchanged. Each group only sees its own sensors, so drift or an attack
    in one stage cannot change the reconstruction of another (Kravchik &
    Shabtai 2018 found per-stage models far better on SWaT).

    Each part keeps the full model's compression (input numbers / latent
    numbers, about 10.75 for 43 sensors -> 16 x 15), with at least one latent
    channel, so no part can just copy its input.
    """

    def __init__(self, groups, window: int = 60, channels: tuple[int, int] = (32, 16),
                 compression: float = 10.75, kernel: int = 5, dropout: float = 0.0):
        super().__init__()
        groups = [list(map(int, g)) for g in groups]
        flat = sorted(i for g in groups for i in g)
        if flat != list(range(len(flat))):
            raise ValueError("groups must split the sensors 0..n-1, each exactly once")
        self.n_features = len(flat)
        self.window = window
        self.groups = groups
        latent_steps = window // 4
        self.parts = nn.ModuleList(
            ConvAE(len(g), window, channels,
                   max(1, round(len(g) * window / (latent_steps * compression))),
                   kernel, dropout)
            for g in groups
        )
        order = torch.tensor([i for g in groups for i in g])
        self.register_buffer("order", order, persistent=False)
        self.register_buffer("inverse", torch.argsort(order), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1:] != (self.window, self.n_features):
            raise ValueError(
                f"expected (batch, {self.window}, {self.n_features}), got {tuple(x.shape)}"
            )
        outs = [part(x[:, :, g]) for part, g in zip(self.parts, self.groups)]
        return torch.cat(outs, dim=2)[:, :, self.inverse]


def _device(device):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batch(windows: np.ndarray, idx, device) -> torch.Tensor:
    # torch.tensor copies: the windows are a read-only view of the 2-D data.
    return torch.tensor(windows[idx], dtype=torch.float32, device=device)


def corrupt(x: torch.Tensor, noise_std: float = 0.0, mask_prob: float = 0.0,
            sensor_mask_prob: float = 0.0) -> torch.Tensor:
    """Corrupted copy of a batch ``(B, window, sensors)`` for denoising training.

    noise_std:        add Gaussian noise with this standard deviation.
    mask_prob:        set each single value to 0 with this probability.
    sensor_mask_prob: set a sensor's whole window to 0 with this probability
                      (so the model learns to rebuild a sensor from the others).
    Uses torch's global random generator (seeded by ``train_ae``).
    """
    out = x
    if noise_std > 0:
        out = out + noise_std * torch.randn_like(out)
    if mask_prob > 0:
        out = out * (torch.rand_like(out) >= mask_prob)
    if sensor_mask_prob > 0:
        keep = torch.rand(x.shape[0], 1, x.shape[2], device=x.device) >= sensor_mask_prob
        out = out * keep
    return out


def train_ae(model: nn.Module, train_windows: np.ndarray, epochs: int = 20,
             batch_size: int = 256, lr: float = 1e-3,
             val_windows: np.ndarray | None = None, device=None, seed: int = 0,
             verbose: bool = True, weight_decay: float = 0.0, noise_std: float = 0.0,
             mask_prob: float = 0.0, sensor_mask_prob: float = 0.0) -> list[dict]:
    """Train ``model`` to reconstruct ``train_windows`` (normal data only).

    Returns one dict per epoch: ``epoch``, ``train_loss`` and, if
    ``val_windows`` is given, ``val_loss`` (mean window MSE; for monitoring
    only, nothing is selected on it). Shuffling uses ``seed``.

    Regularization (all off by default):
    ``weight_decay`` > 0 switches the optimizer to AdamW with that decay.
    ``noise_std`` / ``mask_prob`` / ``sensor_mask_prob`` make it a denoising
    autoencoder: the input is corrupted with :func:`corrupt`, the loss
    compares the reconstruction with the *clean* window. The reported
    ``train_loss`` is then the denoising loss.
    """
    device = _device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    if weight_decay > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    denoise = noise_std > 0 or mask_prob > 0 or sensor_mask_prob > 0
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
            x_in = corrupt(x, noise_std, mask_prob, sensor_mask_prob) if denoise else x
            opt.zero_grad()
            loss = loss_fn(model(x_in), x)
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
def sensor_errors(model: nn.Module, windows: np.ndarray, batch_size: int = 2048,
                  device=None) -> np.ndarray:
    """Squared reconstruction error of each sensor, averaged over the window.

    Shape ``(len(windows), n_features)``. Its mean over sensors equals
    :func:`score_windows`. Use it for per-sensor scoring
    (``src.eval.adaptive``) or to see which sensors drive an alarm.
    """
    device = _device(device)
    model.to(device).eval()
    out = np.empty((len(windows), windows.shape[2]), dtype=np.float32)
    for start in range(0, len(windows), batch_size):
        sl = slice(start, start + batch_size)
        x = _batch(windows, sl, device)
        out[sl] = ((model(x) - x) ** 2).mean(dim=1).cpu().numpy()
    return out


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
