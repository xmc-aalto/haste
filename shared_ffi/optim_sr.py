import torch
from torch.optim import Optimizer

from shared_ffi.sr_triton import sgd_update


class SRSgd(Optimizer):
    """
    SGD + (optional) momentum + weight decay + stochastic rounding for low-precision weights.

    - fp32 params: standard SGD+momentum in Python (no SR)
    - low-precision params: Triton fused kernel with optional momentum + SR.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        use_momentum: bool = True,
        stochastic_rounding: bool = True,
        seed: int = 123,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            use_momentum=use_momentum,
            stochastic_rounding=stochastic_rounding,
            seed=seed,
        )
        super().__init__(params, defaults)

        # Always create a momentum buffer so the Triton kernel has a valid pointer.
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                if "momentum_buf" not in state:
                    state["momentum_buf"] = torch.zeros_like(
                        p.data, dtype=torch.float32, device=p.device
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            mu = group["momentum"]
            use_mom = group["use_momentum"]
            sr = group["stochastic_rounding"]
            seed = group["seed"]

            # change seed each step so SR pattern changes
            group["seed"] += 1

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                mom_buf = state["momentum_buf"]

                # -------- FP32 params: Python SGD+(optional)momentum (no SR) --------
                if p.dtype == torch.float32:
                    if wd != 0.0:
                        grad = grad.add(p, alpha=wd)

                    if use_mom and mu != 0.0:
                        mom_buf.mul_(mu).add_(grad)
                        update = mom_buf
                    else:
                        update = grad

                    p.add_(update, alpha=-lr)
                    continue

                # -------- Low-precision params: Triton fused SGD+m+wd+SR --------
                w = p.data
                if not w.is_contiguous():
                    w_contig = w.contiguous()
                else:
                    w_contig = w

                g = grad
                if not g.is_contiguous():
                    g_contig = g.contiguous()
                else:
                    g_contig = g

                sgd_update(
                    w_contig,
                    g_contig,
                    mom_buf,
                    learning_rate=lr,
                    weight_decay=wd,
                    momentum=mu,
                    use_momentum=use_mom,
                    stochastic_rounding=sr,
                    seed=seed,
                )

                if w_contig.data_ptr() != w.data_ptr():
                    p.data.copy_(w_contig)

        return loss





# class SRSgd(Optimizer):
#     """
#     SGD + (optional) momentum + weight decay + stochastic rounding for low-precision weights.

#     - fp32 params: standard SGD+momentum in Python (no SR)
#     - low-precision params: Triton fused kernel with optional momentum + SR.
#     """

#     def __init__(
#         self,
#         params,
#         lr: float = 1e-3,
#         weight_decay: float = 0.0,
#         momentum: float = 0.0,
#         use_momentum: bool = True,
#         stochastic_rounding: bool = True,
#         seed: int = 123,
#     ):
#         defaults = dict(
#             lr=lr,
#             weight_decay=weight_decay,
#             momentum=momentum,
#             use_momentum=use_momentum,
#             stochastic_rounding=stochastic_rounding,
#             seed=seed,
#         )
#         super().__init__(params, defaults)

#         # Always create a momentum buffer so the Triton kernel has a valid pointer.
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if not p.requires_grad:
#                     continue
#                 state = self.state[p]
#                 if "momentum_buf" not in state:
#                     state["momentum_buf"] = torch.zeros_like(
#                         p.data, dtype=torch.float32, device=p.device
#                     )

#     @torch.no_grad()
#     def step(self, closure=None):
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             lr = group["lr"]
#             wd = group["weight_decay"]
#             mu = group["momentum"]
#             use_mom = group["use_momentum"]
#             sr = group["stochastic_rounding"]
#             seed = group["seed"]

#             # change seed each step so SR pattern changes
#             group["seed"] += 1

#             for p in group["params"]:
#                 if p.grad is None:
#                     continue

#                 grad = p.grad
#                 state = self.state[p]
#                 mom_buf = state["momentum_buf"]

#                 # -------- FP32 params: Python SGD+(optional)momentum (no SR) --------
#                 if p.dtype == torch.float32:
#                     if wd != 0.0:
#                         grad = grad.add(p, alpha=wd)

#                     if use_mom and mu != 0.0:
#                         mom_buf.mul_(mu).add_(grad)
#                         update = mom_buf
#                     else:
#                         update = grad

#                     p.add_(update, alpha=-lr)
#                     continue

#                 # -------- Low-precision params: Triton fused SGD+m+wd+SR --------
#                 w = p.data
#                 if not w.is_contiguous():
#                     w_contig = w.contiguous()
#                 else:
#                     w_contig = w

#                 g = grad
#                 if not g.is_contiguous():
#                     g_contig = g.contiguous()
#                 else:
#                     g_contig = g

#                 sgd_update(
#                     w_contig,
#                     g_contig,
#                     mom_buf,
#                     learning_rate=lr,
#                     weight_decay=wd,
#                     momentum=mu,
#                     use_momentum=use_mom,
#                     stochastic_rounding=sr,
#                     seed=seed,
#                 )

#                 if w_contig.data_ptr() != w.data_ptr():
#                     p.data.copy_(w_contig)

#         return loss

#     @torch.no_grad()
#     def warmup_triton(self, mu_on=0.9):
#         # find one low-precision param
#         p = None
#         for group in self.param_groups:
#             for pp in group["params"]:
#                 if pp.requires_grad and pp.is_cuda and pp.dtype != torch.float32:
#                     p = pp
#                     break
#             if p is not None:
#                 break
#         if p is None:
#             return

#         # ensure grad exists
#         p.grad = torch.zeros_like(p)

#         # snapshot group state
#         saved = [{
#             "lr": g["lr"],
#             "weight_decay": g["weight_decay"],
#             "momentum": g["momentum"],
#             "use_momentum": g["use_momentum"],
#             "seed": g["seed"],
#         } for g in self.param_groups]

#         # make warmup side-effect free
#         for g in self.param_groups:
#             g["lr"] = 0.0
#             g["weight_decay"] = 0.0
#             g["use_momentum"] = True

#         # compile momentum-off path
#         for g in self.param_groups:
#             g["momentum"] = 0.0
#         self.step()

#         # compile momentum-on path
#         for g in self.param_groups:
#             g["momentum"] = float(mu_on)
#         self.step()

#         self.zero_grad(set_to_none=True)

#         # restore everything (including seed so SR stream doesn't shift)
#         for g, s in zip(self.param_groups, saved):
#             g["lr"] = s["lr"]
#             g["weight_decay"] = s["weight_decay"]
#             g["momentum"] = s["momentum"]
#             g["use_momentum"] = s["use_momentum"]
#             g["seed"] = s["seed"]


