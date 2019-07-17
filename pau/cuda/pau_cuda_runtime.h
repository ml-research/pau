#ifndef PAU_RUNTIME
#define PAU_RUNTIME

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr uint32_t THREADS_PER_BLOCK = 512;

//using namespace std;

template <typename T>
__host__ __device__ __forceinline__ T ATenCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <int step = 1>
inline bool getApplyGrid(uint64_t totalElements, dim3& grid, int64_t curDevice) {
  if (curDevice == -1) return false;
  uint64_t blockSize = static_cast<uint64_t>(THREADS_PER_BLOCK) * static_cast<uint64_t>(step);
  uint64_t numBlocks = ATenCeilDiv(totalElements, blockSize);
  uint64_t maxGridX = at::cuda::getDeviceProperties(curDevice)->maxGridSize[0];

  //cout << "maxGridX " << maxGridX  << "\n";
  //cout << "numBlocks " << numBlocks  << "\n";

  if (numBlocks > maxGridX)
      numBlocks = maxGridX;
  grid = dim3(numBlocks);
  return true;
}


#endif