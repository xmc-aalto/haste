from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="shared_ffi",
    ext_modules=[
        CUDAExtension(
            name="shared_ffi._C",
            sources=[
                "shared_ffi/ffi_forward_kernels.cu",
                "shared_ffi/ffi_backward_features.cu",
                "shared_ffi/ffi_backward_weights.cu",
                "shared_ffi/ffi_wrapper.cpp",
            ],
            extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-lineinfo",
                    # "-gencode=arch=compute_80,code=sm_80",
                    # "-gencode=arch=compute_90,code=sm_90a",
                    # "-gencode=arch=compute_90,code=compute_90",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["shared_ffi"],
)
