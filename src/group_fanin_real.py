import torch
import math
from shared_ffi import ops

GROWTH_MODES = ['random']  # [ 'random','momentum', 'momentum_neuron', 'gradient']
PRUNE_MODES = ['threshold', 'fraction']
INIT_MODES = ["random", "zero"]
SCORE_MODES  = ["mean", "max"]


# pytorch provides `unravel_index` only starting with version 2.2
# we use this if it is available, and provide a backport of this function otherwise.
if hasattr(torch, 'unravel_index'):
    unravel_function = torch.unravel_index
else:
    import itertools
    import operator

    def unravel_index_old(indices, shape):
        if indices.is_complex() or indices.is_floating_point() or indices.dtype == torch.bool:
            raise TypeError(f"Expected 'indices' to be an integer tensor, but got {indices.dtype}")

        if not all(dim >= 0 for dim in shape):
            raise ValueError(f"'shape' cannot have negative values, but got {tuple(shape)}")

        # unravel implementation based on pytorch 2.2's code
        coefs = list(reversed(list(itertools.accumulate(reversed(shape[1:] + torch.Size([1])), func=operator.mul))))
        combined = indices.unsqueeze(-1).floor_divide(
            torch.tensor(coefs, device=indices.device, dtype=torch.int64)
        ) % torch.tensor(shape, device=indices.device, dtype=torch.int64)
        return combined.unbind(-1)

    unravel_function = unravel_index_old



class GroupFixedFanInReal(torch.nn.Module):
    '''
    Implements a fixed fan-in sparse linear layer for PyTorch models.

    This layer preserves a constant number of incoming connections (fan-in) per output neuron,
    offering dynamic adjustments through pruning and regrowth strategies based on the layer's
    configuration. Supports customizable pruning modes, growth modes, and weight initialization methods.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        fan_in (int): Number of incoming connections per output neuron.
        bias (bool, optional): Whether to include a bias term. Defaults to ``True``.
        device, dtype: Specifications for the device and data type of the module's parameters.
        transpose (bool, optional): Whether the feature matrix should be transposed. Defaults to True.
        prune_mode (str, optional): Strategy for pruning weights ('threshold' or 'fraction'). Defaults to 'threshold'.
        rewire_threshold (float, optional): Threshold for pruning in 'threshold' mode. Must be specified if using this mode.
        rewire_fraction (float, optional): Fraction of weights to prune in 'fraction' mode. Must be specified if using this mode.
        growth_mode (str, optional): Strategy for adding new weights during the rewire process. Defaults to 'random'.
        init_mode (str, optional): Method for initializing new weights ('random' or 'zero'). Defaults to 'random'.
        strategy (int, optional): Which kernel implementation to use. Can be tuned for improved performance.

    '''
    __constants__ = ['in_features', 'out_features', 'fan_in']
    in_features: int
    out_features: int
    fan_in: int

    def __init__(
            self,
            in_features: int, out_features: int,
            fan_in: int,
            bias: bool = True,
            device=None,
            dtype=None,
            *,
            group_size=32,
            prune_mode="threshold",
            rewire_threshold=None,
            rewire_fraction=None,
            growth_mode="random",
            init_mode="random",
            score_mode="mean",
            **kwargs
    ):
        super().__init__(**kwargs)
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.fan_in = fan_in
        #self.gsize = gsize

        # validation for different modes and their parameters.
        assert prune_mode in PRUNE_MODES, f"Invalid prune_mode={prune_mode}; supported: {PRUNE_MODES}"
        assert growth_mode in GROWTH_MODES, f"Invalid growth_mode={growth_mode}; supported: {GROWTH_MODES}"
        assert init_mode in INIT_MODES, f"Invalid init_mode={init_mode}; supported: {INIT_MODES}"
        assert score_mode in SCORE_MODES,  f"Invalid score_mode={score_mode}; supported: {SCORE_MODES}"


        if prune_mode == "threshold" and rewire_threshold is None:
            raise ValueError("rewire_threshold must be specified for prune_mode 'threshold'")
        elif prune_mode == "fraction" and rewire_fraction is None:
            raise ValueError("rewire_fraction must be specified for prune_mode 'fraction'")

        self.prune_mode = prune_mode
        self.growth_mode = growth_mode
        self.init_mode = init_mode
        self.score_mode = score_mode  
        self.use_grad = False

        # assignment and verification of pruning parameters.
        if prune_mode == 'fraction':
            self.rewire_fraction = rewire_fraction
            assert rewire_fraction >= 0 and rewire_fraction < 1, f"Invalid reware_fraction={rewire_fraction}; supported range [0,1)"
        elif prune_mode == 'threshold':
            self.rewire_threshold = rewire_threshold

        weights = torch.empty((out_features, fan_in), **factory_kwargs)
        self.weights = torch.nn.parameter.Parameter(weights)
        bound = 1 / math.sqrt(self.fan_in)
        torch.nn.init.uniform_(self.weights, -bound, bound)

        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.empty(out_features, **factory_kwargs))
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
            
        assert group_size in [16,32,64] , f"Current kernels only support group size from 16,32,64. Selected group size={group_size}"
        self.group_size = group_size
        self.num_groups = (out_features + self.group_size - 1) // self.group_size


        group_locations = torch.randint(high=self.in_features, size=(self.num_groups, fan_in), dtype=torch.int32)

        self.register_buffer("group_locations", group_locations, persistent=True)
        
        
    def forward(self, features):
        result = ops.group_ffi_mul(
            features, self.weights, self.group_locations, self.bias,gsize=self.group_size)
        
        return result

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, fan_in={}, bias={}'.format(
            self.in_features, self.out_features, self.fan_in, self.bias is not None
        )
        
        

    @torch.no_grad()
    def rewire(self):
        """
        Vectorized prune + regrow with grouped connectivity.

        Scoring:
          - if score_mode == "mean":
                score[g, j] = mean_{rows in group g} |w[row, j]|
          - if score_mode == "max":
                score[g, j] = max_{rows in group g} |w[row, j]|

        Pruning:
          - threshold:  elementwise compare against global scalar threshold
          - fraction:   GLOBAL bottom (num_groups * fan_in * rewire_fraction) scores

        """
        device = self.weights.device
        w_dtype = self.weights.dtype

        G  = self.num_groups
        GS = self.group_size
        F  = self.fan_in
        L  = self.out_features

        if L == 0:
            return

        # row -> group mapping
        row_idx  = torch.arange(L, device=device)
        group_id = row_idx // GS                         # (L,)

        # --- 1) per-group score sums in fp32 for stability ---
        scores = torch.full((G, F), 
                            -float("inf") if self.score_mode == "max" else 0.0,
                            device=device, dtype=torch.float32)

        row_counts = torch.zeros(G, device=device, dtype=torch.float32)  # only used for mean
        
        grad = self.weights.grad     # may be None on first step / odd cases
        use_grad = self.use_grad #and (grad is not None)

        chunk_size = 32 * 1024
        for i in range(0, L, chunk_size):
            end = min(i + chunk_size, L)

            chunk_w   = self.weights[i:end]                # (chunk, F)
            chunk_gid = group_id[i:end]                    # (chunk,)

            if use_grad:
                chunk_g = grad[i:end]                      # (chunk, F)
                # saliency ≈ |w * grad|
                chunk_val = (chunk_w * chunk_g).abs().to(torch.float32)
                # alternatively: chunk_val = chunk_g.abs().to(torch.float32)
            else:
                chunk_val = chunk_w.abs().to(torch.float32)

            if self.score_mode == "mean":
                scores.index_add_(0, chunk_gid, chunk_val)
                row_counts.index_add_(
                    0,
                    chunk_gid,
                    torch.ones(end - i, device=device, dtype=torch.float32),
                )
            else:  # "max"
                scores.scatter_reduce_(
                    dim=0,
                    index=chunk_gid[:, None].expand(-1, F),
                    src=chunk_val,
                    reduce="amax",
                    include_self=True,
                )

        if self.score_mode == "mean":
            row_counts = row_counts.clamp_min(1).unsqueeze(1)
            scores = scores / row_counts                # (G,F), fp32

        # --- 2) decide pruned fan-in slots per group ---
        if self.prune_mode == 'threshold':
            # still a global scalar threshold, applied elementwise
            prune_mask = scores < float(self.rewire_threshold)   # (G, F) bool

        elif self.prune_mode == 'fraction':
            # GLOBAL fraction over all (group, slot) pairs
            total_slots  = G * F
            num_selected = int(total_slots * self.rewire_fraction)
            if num_selected == 0:
                return

            scores_flat = scores.view(-1)  # (G*F,)
            # pick globally smallest scores
            _, idx_flat = torch.topk(
                scores_flat,
                k=num_selected,
                largest=False,
                sorted=False,
            )

            prune_mask_flat = torch.zeros_like(scores_flat, dtype=torch.bool)
            prune_mask_flat[idx_flat] = True
            prune_mask = prune_mask_flat.view(G, F)  # back to (G, F)

        else:
            raise RuntimeError(f"Unsupported prune_mode={self.prune_mode}")

        if not prune_mask.any():
            return

        # --- 3) update group_locations for pruned slots ---
        num_pruned_slots = int(prune_mask.sum().item())
        new_locs = torch.randint(
            high=self.in_features,
            size=(num_pruned_slots,),
            dtype=self.group_locations.dtype,
            device=self.group_locations.device,
        )
        self.group_locations[prune_mask] = new_locs

        # --- 4) reinit weights for all rows in each pruned slot ---
        # expand (G,F) -> (L,F) via group_id
        prune_mask_row = prune_mask[group_id]           # (L,F)

        mask_flat = prune_mask_row.view(-1)
        num_pruned_weights = int(mask_flat.sum().item())
        if num_pruned_weights == 0:
            return

        regrowth_flat = torch.empty(
            (num_pruned_weights,),
            device=device,
            dtype=w_dtype,
        )
        if self.init_mode == "random":
            bound = 1.0 / math.sqrt(self.fan_in)
            regrowth_flat.uniform_(-bound, bound)
        else:
            regrowth_flat.zero_()

        self.weights.view(-1)[mask_flat] = regrowth_flat
    


