#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


#include "pau_cuda_runtime.h"

template <typename scalar_t>
__global__ void pau_cuda_forward_kernel_unrestricted( const scalar_t* __restrict__ x, const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ d, scalar_t* __restrict__ result, size_t x_size) {

    const int index =  blockDim.x * blockIdx.x + threadIdx.x;
    if (index < x_size) {
        const auto xp1 = x[index];
        const auto xp2 = xp1 * xp1;
        const auto xp3 = xp2 * xp1;
        const auto xp4 = xp3 * xp1;
        const auto xp5 = xp4 * xp1;


        const auto n_0 = n[0];
        const auto n_1 = n[1];
        const auto n_2 = n[2];
        const auto n_3 = n[3];
        const auto n_4 = n[4];
        const auto n_5 = n[5];

        const auto d_0 = d[0];
        const auto d_1 = d[1];
        const auto d_2 = d[2];
        const auto d_3 = d[3];

        const auto P = n_0 + xp1*n_1 + xp2*n_2 + xp3*n_3 + xp4*n_4 + xp5*n_5;
        const auto Q = 1.0 + xp1*d_0 + xp2*d_1 + xp3*d_2 + xp4*d_3;

        result[index] = P/Q;
    }
}

at::Tensor pau_cuda_forward_unrestricted(torch::Tensor x, torch::Tensor n, torch::Tensor d){
    auto result = at::empty_like(x);
    const auto x_size = x.numel();

    const dim3 block = dim3(std::min(static_cast<int64_t>(dim3(THREADS_PER_BLOCK).x), x_size));
    dim3 grid;
    int64_t curDevice = x.device().index();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
    AT_CHECK(getApplyGrid(x_size, grid, curDevice), "pau_cuda_forward_unrestricted: input too large or too many dimensions");


    AT_DISPATCH_FLOATING_TYPES(x.type(), "pau_cuda_forward_unrestricted", ([&] {
    pau_cuda_forward_kernel_unrestricted<scalar_t>
        <<<grid, block, 0, stream>>>(
            x.data<scalar_t>(),
            n.data<scalar_t>(),
            d.data<scalar_t>(),
            result.data<scalar_t>(),
            x_size);
        }));

    return result;
}


template <typename scalar_t>
__global__ void pau_cuda_backward_kernel_unrestricted(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ d,
    scalar_t* __restrict__ d_x,
    scalar_t* __restrict__ d_n,
    scalar_t* __restrict__ d_d,
    size_t x_size) {

    const int index =  blockDim.x * blockIdx.x + threadIdx.x;
    if (index < x_size) {
        const auto xp1 = x[index];
        const auto xp2 = xp1 * xp1;
        const auto xp3 = xp2 * xp1;
        const auto xp4 = xp3 * xp1;
        const auto xp5 = xp4 * xp1;


        const auto n_0 = n[0];
        const auto n_1 = n[1];
        const auto n_2 = n[2];
        const auto n_3 = n[3];
        const auto n_4 = n[4];
        const auto n_5 = n[5];

        const auto d_0 = d[0];
        const auto d_1 = d[1];
        const auto d_2 = d[2];
        const auto d_3 = d[3];

        const auto P = n_0 + xp1*n_1 + xp2*n_2 + xp3*n_3 + xp4*n_4 + xp5*n_5;
        const auto Q = 1.0 + xp1*d_0 + xp2*d_1 + xp3*d_2 + xp4*d_3;

        const auto R = n_1 + 2.0*n_2*xp1 + 3.0*n_3*xp2 + 4.0*n_4*xp3 + 5.0*n_5*xp4;
        const auto S = (d_0 + 2.0*d_1*xp1 + 3.0*d_2*xp2 + 4.0*d_3*xp3 );

        const auto mpq2 = -P/(Q*Q);

        const auto grad_o = grad_output[index];

        d_x[index] = (R/Q + S*mpq2) * grad_output[index];

        const auto mpq2go = mpq2 * grad_o;

        atomicAdd(&d_d[0], (mpq2go*xp1));
        atomicAdd(&d_d[1], (mpq2go*xp2));
        atomicAdd(&d_d[2], (mpq2go*xp3));
        atomicAdd(&d_d[3], (mpq2go*xp4));

        atomicAdd(&d_n[0], (1.0/Q) * grad_o);
        atomicAdd(&d_n[1], (xp1/Q) * grad_o);
        atomicAdd(&d_n[2], (xp2/Q) * grad_o);
        atomicAdd(&d_n[3], (xp3/Q) * grad_o);
        atomicAdd(&d_n[4], (xp4/Q) * grad_o);
        atomicAdd(&d_n[5], (xp5/Q) * grad_o);

    }
}

std::vector<torch::Tensor> pau_cuda_backward_unrestricted(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
    const auto x_size = x.numel();
    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n);
    auto d_d = at::zeros_like(d);


    const dim3 block = dim3(std::min(static_cast<int64_t>(dim3(THREADS_PER_BLOCK).x), x_size));
    dim3 grid;
    int64_t curDevice = x.device().index();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
    AT_CHECK(getApplyGrid(x_size, grid, curDevice), "pau_cuda_backward_unrestricted: input too large or too many dimensions");

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pau_cuda_backward_unrestricted", ([&] {
    pau_cuda_backward_kernel_unrestricted<scalar_t>
        <<<grid, block, 0, stream>>>(
            grad_output.data<scalar_t>(),
            x.data<scalar_t>(),
            n.data<scalar_t>(),
            d.data<scalar_t>(),
            d_x.data<scalar_t>(),
            d_n.data<scalar_t>(),
            d_d.data<scalar_t>(),
            x_size);
    }));

    return {d_x, d_n, d_d};
}
