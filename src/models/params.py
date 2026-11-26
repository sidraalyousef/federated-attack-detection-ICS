"""Model weights <-> lists of NumPy arrays, for Flower / FedAvg.

Flower moves a model as ``NDArrays``: a list of NumPy arrays that FedAvg
averages element-wise. :func:`get_parameters` / :func:`set_parameters` convert
between that list and a PyTorch model; they work for every model in
``src.models``.

Partial sharing (by-stage split)
--------------------------------
In the by-stage split each client has a different number of sensors, so the
layers that touch the sensors have different shapes on every client and
cannot be averaged. A model can mark those layers as *local* by listing their
name prefixes in a ``local_prefixes`` class attribute (see ``StageAE``). Then:

* :func:`shared_keys` - the ``state_dict`` keys every client has with the same
  shape. Only these go to the server; plain FedAvg averages them.
* :func:`local_keys`  - the sensor-specific keys. They never leave the client
  and must be kept by the client between rounds.

A model without ``local_prefixes`` is fully shared (``shared_keys`` is every
key, ``local_keys`` is empty), so the time-split clients use the same code.

    keys = shared_keys(model)
    arrays = get_parameters(model, keys)        # send to Flower
    set_parameters(model, arrays, keys)         # load the global average
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


def _is_local(model: nn.Module, key: str) -> bool:
    prefixes = getattr(model, "local_prefixes", ())
    return any(key == p or key.startswith(p + ".") for p in prefixes)


def shared_keys(model: nn.Module) -> list[str]:
    """``state_dict`` keys that are averaged across clients, in ``state_dict`` order."""
    return [k for k in model.state_dict() if not _is_local(model, k)]


def local_keys(model: nn.Module) -> list[str]:
    """``state_dict`` keys that stay on the client (empty for fully shared models)."""
    return [k for k in model.state_dict() if _is_local(model, k)]


def get_parameters(model: nn.Module, keys: Sequence[str] | None = None) -> list[np.ndarray]:
    """The model's weights as a list of NumPy arrays (Flower ``NDArrays``).

    ``keys=None``: every ``state_dict`` entry, in ``state_dict`` order.
    Otherwise only ``keys``, in that order (e.g. ``shared_keys(model)``).
    The arrays are CPU copies: changing them does not change the model.
    """
    state = model.state_dict()
    keys = list(state) if keys is None else list(keys)
    missing = [k for k in keys if k not in state]
    if missing:
        raise ValueError(f"model has no parameters {missing}")
    return [state[k].detach().cpu().numpy().copy() for k in keys]


def set_parameters(model: nn.Module, parameters: Sequence[np.ndarray],
                   keys: Sequence[str] | None = None) -> None:
    """Load arrays from :func:`get_parameters` (or averaged by FedAvg) into ``model``.

    ``keys`` must be the same list that was passed to :func:`get_parameters`
    (``None`` = every entry). Entries not in ``keys`` are left as they are,
    so loading the shared part keeps the client's local layers. Shapes must
    match; float64 averages are cast to the model's dtype. Raises
    ``ValueError`` on a wrong count, name or shape.
    """
    state = model.state_dict()
    keys = list(state) if keys is None else list(keys)
    if len(parameters) != len(keys):
        raise ValueError(f"expected {len(keys)} arrays, got {len(parameters)}")
    for k, p in zip(keys, parameters):
        if k not in state:
            raise ValueError(f"model has no parameter {k!r}")
        p = np.asarray(p)
        if tuple(p.shape) != tuple(state[k].shape):
            raise ValueError(f"{k}: expected shape {tuple(state[k].shape)}, got {p.shape}")
    with torch.no_grad():
        for k, p in zip(keys, parameters):
            state[k].copy_(torch.as_tensor(np.asarray(p)))  # state_dict shares storage
