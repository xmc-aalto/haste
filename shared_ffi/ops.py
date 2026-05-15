import torch
from . import bindings as _bindings
from typing import Optional
from dataclasses import dataclass


def group_ffi_mul(features: torch.Tensor,
            weights: torch.Tensor,
            group_locations: torch.Tensor,
            bias: Optional[torch.Tensor],
            gsize: int = 32):
    """
    Performs a Group fixed fan-in batched matrix multiplication.
    Multiplies `features` with the fixed fan-in sparse matrix described by `weights` and `locations`, possibly
    adding a `bias` term.

    """
    if features.ndim > 2:
        original_shape = features.shape[:-1]
        reshaped = features.view(-1, features.shape[-1])
        result = _bindings.GroupFFIMulFunction.apply(reshaped, weights, group_locations, bias, gsize)
        return result.view(original_shape + (weights.shape[0],))
    else:
        return _bindings.GroupFFIMulFunction.apply(features, weights, group_locations, bias, gsize)


__all__ = ['group_ffi_mul']
