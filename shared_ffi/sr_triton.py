import torch
import os
import triton
import triton.language as tl
from typing import Optional
from torch import Tensor


@triton.jit
def stochastic_rounding_to_bf16(source, seed, offs_out):
    rand = tl.randint(seed, offs_out) & 65535
    out = source.to(tl.int32, bitcast=True) + rand
    out = out & -65536
    out = out.to(tl.float32, bitcast=True) 
    out = out.to(tl.bfloat16)
    return out

@triton.jit
def stochastic_rounding_to_fp8(source, seed, offs_out):
    rand = tl.randint(seed, offs_out) & 1048575
    out = source.to(tl.int32, bitcast=True) + rand
    out = out & -1048576
    out = out.to(tl.float32, bitcast=True) 
    out = out.to(tl.float8e4nv)
    return out

@triton.jit
def _seeded_stochastic_rounding(
    x_ptr,
    output_ptr,
    n_elements,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    if (output_ptr.dtype.element_ty == tl.float8e4nv):
        output = stochastic_rounding_to_fp8(x, seed, offsets)
    else:
        output = stochastic_rounding_to_bf16(x, seed, offsets)
    tl.store(output_ptr + offsets, output, mask=mask)



def stochastic_rounding(x, output, seed):
    assert x.is_contiguous()
    assert output.is_contiguous()
    assert x.shape == output.shape
    assert x.dtype == torch.float32
    assert output.dtype != torch.float32
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _seeded_stochastic_rounding[grid](x, output, n_elements, seed, BLOCK_SIZE=1024)

@triton.jit
def _sgd(
    weights_ptr,
    gradient_ptr,
    n_elements,
    lr,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data
    mask = offsets < n_elements
    W = tl.load(weights_ptr + offsets, mask=mask)
    grad = tl.load(gradient_ptr + offsets, mask=mask)
    W = W.to(tl.float32)
    W = W - grad*lr

    if (weights_ptr.dtype.element_ty == tl.float8e4nv):
        W = stochastic_rounding_to_fp8(W, seed, offsets)
    else:
        W = stochastic_rounding_to_bf16(W, seed, offsets)
    tl.store(weights_ptr + offsets, W, mask=mask)

def sgd_update(weights: Tensor, gradient: Tensor, compensation: Optional[Tensor],
               learning_rate: float, weight_decay: float, stochastic_rounding: bool, seed: int):
    assert weights.is_contiguous()
    assert weights.dtype != torch.float32
    assert gradient.is_contiguous()
    assert gradient.shape == weights.shape
    n_elements = weights.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _sgd[grid](weights, gradient, n_elements, learning_rate, seed, BLOCK_SIZE=1024)
    
def test_stochastic_rounding():
    x = torch.randn(3, 5)
    x = x.cuda()
    fp8_x = stochastic_rounding(x, 123, torch.float8_e4m3fn)
    bf16_x = stochastic_rounding(x, 123, torch.bfloat16)

    print(x)
    print(fp8_x)
    print(bf16_x)    
    


@triton.jit
def _sgd_m_wd_sr(
    w_ptr,          # low-precision weights (bf16/fp16/fp8)
    g_ptr,          # gradients (bf16/fp16/fp32)
    m_ptr,          # momentum buffer (fp32) - may be ignored if use_mom == 0
    n_elements,
    lr,
    momentum,
    weight_decay,
    use_mom,        # 1 = use momentum, 0 = ignore m_ptr
    do_sr,          # 1 = stochastic rounding, 0 = normal cast
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused SGD with optional momentum + weight decay + optional stochastic rounding.

    Math (all in fp32):

      if use_mom:
          g = g + weight_decay * w
          m = momentum * m + g
          w = w - lr * m
      else:
          g = g + weight_decay * w
          w = w - lr * g
    """

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load weights and grads, cast to fp32
    w_low = tl.load(w_ptr + offsets, mask=mask)
    w32 = w_low.to(tl.float32)

    g_low = tl.load(g_ptr + offsets, mask=mask)
    g = g_low.to(tl.float32)

    # Weight decay: g = g + wd * w
    g = g + weight_decay * w32

    # Momentum vs plain SGD
    if use_mom != 0:
        m = tl.load(m_ptr + offsets, mask=mask)   # read momentum only when used
        m = momentum * m + g                      # m_t = mu * m_{t-1} + g_t
        w32 = w32 - lr * m                        # w_t = w_{t-1} - lr * m_t
        tl.store(m_ptr + offsets, m, mask=mask)   # write back momentum
    else:
        # no momentum: plain SGD
        w32 = w32 - lr * g

    # Cast back to low-precision with SR or normal cast
    if do_sr != 0:
        w_new = stochastic_rounding_to_bf16(w32, seed, offsets)
    else:
        w_new = w32.to(tl.bfloat16)

    tl.store(w_ptr + offsets, w_new, mask=mask)

    
    
def sgd_update(
    weights: torch.Tensor,        # low precision (bf16/fp16/fp8)
    gradient: torch.Tensor,       # bf16/fp16/fp32
    momentum_buf: torch.Tensor,   # fp32 (can be dummy if not used)
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    use_momentum: bool,
    stochastic_rounding: bool,
    seed: int,
):
    """
    Fused SGD + optional momentum + weight decay + SR, in-place on `weights`.

    - If use_momentum is False or momentum == 0.0 → plain SGD + weight decay.
    - If use_momentum is True and momentum > 0.0 → SGD with momentum + weight decay.
    """

    assert weights.is_contiguous(), "weights must be contiguous"
    assert gradient.is_contiguous(), "gradient must be contiguous"
    assert momentum_buf.is_contiguous(), "momentum_buf must be contiguous"
    assert weights.shape == gradient.shape == momentum_buf.shape

    n_elements = weights.numel()
    do_sr = 1 if stochastic_rounding else 0
    use_mom_flag = 1 if (use_momentum and momentum != 0.0) else 0

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _sgd_m_wd_sr[grid](
        weights,
        gradient,
        momentum_buf,
        n_elements,
        learning_rate,
        momentum,
        weight_decay,
        use_mom_flag,
        do_sr,
        seed,
        BLOCK_SIZE=1024,
    )