from __future__ import annotations

from typing import TypeVar

import torch
from torch import Tensor


def select_at_indexes(indexes: Tensor, tensor: Tensor):
    """Returns the contents of ``tensor`` at the multi-dimensional integer
    array ``indexes``. Leading dimensions of ``tensor`` must match the
    dimensions of ``indexes``.
    """
    dim = len(indexes.shape)
    assert indexes.shape == tensor.shape[:dim]
    num = indexes.numel()
    t_flat = tensor.view((num,) + tensor.shape[dim:])
    s_flat = t_flat[torch.arange(num), indexes.view(-1)]
    return s_flat.view(tensor.shape[:dim] + tensor.shape[dim + 1 :])


def to_onehot(indexes: Tensor, num: int, dtype: torch.dtype | None = None):
    """Converts integer values in multi-dimensional tensor ``indexes``
    to one-hot values of size ``num``; expanded in an additional
    trailing dimension."""
    if dtype is None:
        dtype = indexes.dtype
    onehot = torch.zeros(indexes.shape + (num,), dtype=dtype, device=indexes.device)
    onehot.scatter_(-1, indexes.unsqueeze(-1).type(torch.long), 1)
    return onehot


def from_onehot(onehot: Tensor, dim: int = -1, dtype: torch.dtype | None = None):
    """Argmax over trailing dimension of tensor ``onehot``. Optional return
    dtype specification."""
    indexes = torch.argmax(onehot, dim=dim)
    if dtype is not None:
        indexes = indexes.type(dtype)
    return indexes


def valid_mean(
    tensor: Tensor,
    valid: Tensor | None = None,
    dim: tuple[int, ...] | None = None,
) -> Tensor:
    """Mean of ``tensor``, accounting for optional mask ``valid``, optionally
    along a dimension. Valid mask is "broadcast" across trailing dimensions of
    tensor, if tensor has more dimensions than valid. nan and inf values are
    also masked out properly.
    """
    if valid is None:
        dim = () if dim is None else dim
        return tensor.mean(dim=dim)
    if dim is None:
        # broadcasts over trailing dimensions
        # e.g. if tensor has shape [T,B,N] and valid has [T,B]
        return tensor[valid].mean()
    # add extra trailing dimensions to valid mask
    valid = valid[(...,) + (None,) * (tensor.ndim - valid.ndim)]
    masked_tensor = torch.where(valid, tensor, 0)
    return masked_tensor.sum(dim=dim) / masked_tensor.count_nonzero(dim=dim)


def infer_leading_dims(
    tensor: Tensor,
    dim: int,
) -> tuple[int, int, int, tuple[int, ...]]:
    """Looks for up to two leading dimensions in ``tensor``, before
    the data dimensions, of which there are assumed to be ``dim`` number.
    For use at beginning of model's ``forward()`` method, which should
    finish with ``restore_leading_dims()`` (see that function for help.)
    Returns:
    lead_dim: int --number of leading dims found.
    T: int --size of first leading dim, if two leading dims, o/w 1.
    B: int --size of first leading dim if one, second leading dim if two, o/w 1.
    shape: tensor shape after leading dims.
    """
    lead_dim = tensor.dim() - dim
    assert lead_dim in (0, 1, 2)
    if lead_dim == 2:
        T, B = tensor.shape[:2]
    else:
        T = 1
        B = 1 if lead_dim == 0 else tensor.shape[0]
    shape = tensor.shape[lead_dim:]
    return lead_dim, T, B, shape


TensorOrSequence = TypeVar("TensorOrSequence", Tensor, tuple[Tensor, ...])


def restore_leading_dims(
    tensors: TensorOrSequence,
    lead_dim: int,
    T: int = 1,
    B: int = 1,
) -> TensorOrSequence:
    """Reshapes ``tensors`` (one or `tuple`, `list`) to to have ``lead_dim``
    leading dimensions, which will become [], [B], or [T,B].  Assumes input
    tensors already have a leading Batch dimension, which might need to be
    removed. (Typically the last layer of model will compute with leading
    batch dimension.)  For use in model ``forward()`` method, so that output
    dimensions match input dimensions, and the same model can be used for any
    such case.  Use with outputs from ``infer_leading_dims()``."""
    is_seq = isinstance(tensors, (tuple, list))
    tensors_seq = tensors if is_seq else (tensors,)
    if lead_dim == 2:  # (Put T dim.)
        tensors_seq = tuple(t.view((T, B) + t.shape[1:]) for t in tensors_seq)
    if lead_dim == 0:  # (Remove B=1 dim.)
        assert B == 1
        tensors_seq = tuple(t.squeeze(0) for t in tensors_seq)
    return tensors_seq if is_seq else tensors_seq[0]


def update_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, Tensor],
    tau: float | int = 1,
) -> None:
    """Update the state dict of ``model`` using the input ``state_dict``, which
    must match format.  ``tau==1`` applies hard update, copying the values, ``0<tau<1``
    applies soft update: ``tau * new + (1 - tau) * old``.
    """
    if tau == 1:
        model.load_state_dict(state_dict)
    elif tau > 0:
        update_sd = {
            k: tau * state_dict[k] + (1 - tau) * v
            for k, v in model.state_dict().items()
        }
        model.load_state_dict(update_sd)


def explained_variance(y_pred: Tensor, y_true: Tensor) -> float:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    Copied from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/utils.py

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == y_pred.ndim
    var_y = y_true.var()
    if torch.allclose(var_y, var_y.new_zeros()):
        return torch.nan  # surprisingly, torch.nan is a float
    # call `item()` here to prevent the caller from doing `torch.nan.item()`
    return (1 - (y_true - y_pred).var() / var_y).item()
