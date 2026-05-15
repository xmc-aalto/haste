from . import _C

def ffi_forward(locations, features, weights, bias=None, gsize=32):
    B = features.size(1)
    H = features.size(0)
    FANIN = weights.size(1)
    L = weights.size(0)

    return _C.ffi_forward_cuda(
        locations, features, weights, bias,
        int(B), int(H), int(FANIN), int(L),
        int(gsize)
    )

def backward_features(features, weights, locations, grad_outputs, gsize=32):
    B = features.size(1)
    H = features.size(0)
    FANIN = weights.size(1)
    L = weights.size(0)
    return _C.ffi_backward_features_cuda(
        features, weights, locations, grad_outputs,
        int(B), int(H), int(FANIN), int(L), int(gsize)
    )

def backward_weights(features, weights, locations, grad_outputs, gsize=32):
    B = features.size(1)
    H = features.size(0)
    FANIN = weights.size(1)
    L = weights.size(0)
    return _C.ffi_backward_weights_cuda(
        features, weights, locations, grad_outputs,
        int(B), int(H), int(FANIN), int(L), int(gsize)
    )
