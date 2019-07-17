#include <torch/extension.h>
#include <vector>

#include <iostream>

at::Tensor pau_cuda_forward_unrestricted(torch::Tensor x, torch::Tensor n, torch::Tensor d);
std::vector<torch::Tensor> pau_cuda_backward_unrestricted(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


using namespace std;

at::Tensor pau_forward_unrestricted(torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_forward_unrestricted(x, n, d);
}

std::vector<torch::Tensor> pau_backward_unrestricted(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(x);
    CHECK_INPUT(n);
    CHECK_INPUT(d);

    return pau_cuda_backward_unrestricted(grad_output, x, n, d);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &pau_forward_unrestricted, "PAU unrestricted forward");
  m.def("backward", &pau_backward_unrestricted, "PAU unrestricted backward");
}
