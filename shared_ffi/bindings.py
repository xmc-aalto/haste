import pathlib
import torch
from typing import Any, Optional
import shared_ffi
from .sr_triton import stochastic_rounding


class GroupFFIMulFunction(torch.autograd.Function):
    # noinspection PyMethodOverriding
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx: Any,
                features: torch.Tensor,
                weights: torch.Tensor,
                group_locations: torch.Tensor,
                bias: Optional[torch.Tensor],
                gsize: int = 32) -> torch.Tensor:

        features_t = torch.transpose(features, 0, 1).contiguous()

        ctx.save_for_backward(features_t, weights, group_locations,torch.Tensor([gsize]))
        result = shared_ffi.ffi_forward(group_locations,features_t, weights, bias, gsize)

        if bias is not None:
            ctx.use_bias = True
        else:
            ctx.use_bias = False
        return result

    # noinspection PyMethodOverriding
    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        features_t, weights, group_locations, gsize = ctx.saved_tensors
        gsize = int(gsize.item())

        grad_output = grad_output.contiguous()

        if ctx.use_bias:
            bias_gradient = torch.sum(grad_output, dim=0).detach()
        else:
            bias_gradient = None

        input_gradient_ = shared_ffi.backward_features(features_t, weights, group_locations, grad_output,gsize)
        input_gradient_ = torch.transpose(input_gradient_, 0, 1).contiguous()
        if features_t.dtype == torch.bfloat16:
            input_gradient = torch.empty_like(input_gradient_, dtype=features_t.dtype)
            stochastic_rounding(input_gradient_,input_gradient,123) 
        else:
            input_gradient = input_gradient_.to(features_t.dtype)

        weight_gradient = shared_ffi.backward_weights(features_t, weights, group_locations, grad_output,gsize)

        return input_gradient, weight_gradient, None, bias_gradient, None
