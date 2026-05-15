#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

torch::Tensor ffi_backward_features_cuda(
    torch::Tensor features,
    torch::Tensor weights,
    torch::Tensor locations,
    torch::Tensor grad_outputs,
    int B, int H, int FANIN, int L, int gsize);

torch::Tensor ffi_backward_weights_cuda(
    torch::Tensor features,
    torch::Tensor weights,
    torch::Tensor locations,
    torch::Tensor grad_outputs,
    int B, int H, int FANIN, int L, int gsize);

torch::Tensor ffi_forward_cuda(
    torch::Tensor locations,
    torch::Tensor features,
    torch::Tensor weights,
    c10::optional<torch::Tensor> bias_opt,
    int B, int H, int FANIN, int L, int gsize);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ffi_backward_features_cuda", &ffi_backward_features_cuda);
    m.def("ffi_backward_weights_cuda", &ffi_backward_weights_cuda);

    m.def(
        "ffi_forward_cuda",
        &ffi_forward_cuda,
        py::arg("locations"),
        py::arg("features"),
        py::arg("weights"),
        py::arg("bias") = c10::optional<torch::Tensor>(),
        py::arg("B"),
        py::arg("H"),
        py::arg("FANIN"),
        py::arg("L"),
        py::arg("gsize")  = 32
    );
}
