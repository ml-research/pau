import glob

import airspeed
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# degrees
degrees = [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (5, 4)]
#degrees = [(5, 4)]


def generate_cpp_module(fname='pau_cuda.cpp', degrees=degrees):
    file_content = airspeed.Template("""
\#include <torch/extension.h>
\#include <vector>
\#include <iostream>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#foreach ($degs in $degrees)
at::Tensor pau_cuda_forwardA_$degs[0]_$degs[1](torch::Tensor x, torch::Tensor n, torch::Tensor d);
std::vector<torch::Tensor> pau_cuda_backwardA_$degs[0]_$degs[1](torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);

at::Tensor pau_cuda_forwardB_$degs[0]_$degs[1](torch::Tensor x, torch::Tensor n, torch::Tensor d);
std::vector<torch::Tensor> pau_cuda_backwardB_$degs[0]_$degs[1](torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);

at::Tensor pau_cuda_forwardC_$degs[0]_$degs[1](torch::Tensor x, torch::Tensor n, torch::Tensor d);
std::vector<torch::Tensor> pau_cuda_backwardC_$degs[0]_$degs[1](torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);

#end

#foreach ($degs in $degrees)
at::Tensor pau_forwardA__$degs[0]_$degs[1](torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_forwardA_$degs[0]_$degs[1](x, n, d);
}
std::vector<torch::Tensor> pau_backwardA__$degs[0]_$degs[1](torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_backwardA_$degs[0]_$degs[1](grad_output, x, n, d);
}

at::Tensor pau_forwardB__$degs[0]_$degs[1](torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_forwardB_$degs[0]_$degs[1](x, n, d);
}
std::vector<torch::Tensor> pau_backwardB__$degs[0]_$degs[1](torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_backwardB_$degs[0]_$degs[1](grad_output, x, n, d);
}

at::Tensor pau_forwardC__$degs[0]_$degs[1](torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_forwardC_$degs[0]_$degs[1](x, n, d);
}
std::vector<torch::Tensor> pau_backwardC__$degs[0]_$degs[1](torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_backwardC_$degs[0]_$degs[1](grad_output, x, n, d);
}
#end

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#foreach ($degs in $degrees)
    m.def("forwardA_$degs[0]_$degs[1]", &pau_forwardA__$degs[0]_$degs[1], "PAU forward A_$degs[0]_$degs[1]");
    m.def("backwardA_$degs[0]_$degs[1]", &pau_backwardA__$degs[0]_$degs[1], "PAU backward A_$degs[0]_$degs[1]");
    
    m.def("forwardB_$degs[0]_$degs[1]", &pau_forwardB__$degs[0]_$degs[1], "PAU forward B_$degs[0]_$degs[1]");
    m.def("backwardB_$degs[0]_$degs[1]", &pau_backwardB__$degs[0]_$degs[1], "PAU backward B_$degs[0]_$degs[1]");
    
    m.def("forwardC_$degs[0]_$degs[1]", &pau_forwardC__$degs[0]_$degs[1], "PAU forward C_$degs[0]_$degs[1]");
    m.def("backwardC_$degs[0]_$degs[1]", &pau_backwardC__$degs[0]_$degs[1], "PAU backward C_$degs[0]_$degs[1]");
#end
}
    """)

    content = file_content.merge(locals())

    with open(fname, "w") as text_file:
        text_file.write(content)


def generate_cpp_kernels_module(fname='pau_cuda_kernels.cu', degrees=degrees):
    degrees = [[e[0], e[1], max(e[0], e[1])] for e in degrees]

    template = """
\#include <torch/extension.h>
\#include <ATen/cuda/CUDAContext.h>
\#include <cuda.h>
\#include <cuda_runtime.h>
\#include <vector>
\#include <stdlib.h>

constexpr uint32_t THREADS_PER_BLOCK = 512;
"""

    for template_fname in sorted(glob.glob("versions/*.cu")):
        with open(template_fname) as infile:
            template += infile.read()

    file_content = airspeed.Template(template)

    content = file_content.merge(locals())

    with open(fname, "w") as text_file:
        text_file.write(content)


generate_cpp_module(fname='pau_cuda.cpp')
generate_cpp_kernels_module(fname='pau_cuda_kernels.cu')

setup(
    name='pau',
    version='0.0.2',
    ext_modules=[
        CUDAExtension('pau_cuda', [
            'pau_cuda.cpp',
            'pau_cuda_kernels.cu',
        ],
                      extra_compile_args={'cxx': [],
                                          'nvcc': ['-gencode=arch=compute_60,code="sm_60,compute_60"', '-lineinfo']}
                      ),
        # CUDAExtension('pau_cuda_unrestricted', [
        #    'pau_cuda_unrestricted.cpp',
        #    'pau_cuda_kernels_unrestricted.cu',
        # ],
        #              extra_compile_args={'cxx': [],
        #                                  'nvcc': ['-gencode=arch=compute_60,code="sm_60,compute_60"', '-lineinfo']}
        #              )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
