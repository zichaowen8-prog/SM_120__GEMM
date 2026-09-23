#include "kernels.cuh"

namespace int4gemm {
namespace {

__global__ void v00_kernel(const half* __restrict__ a, const uint8_t* __restrict__ b,
                           const half* __restrict__ scales, half* __restrict__ c,
                           int M, int N, int K, int group_size) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    const size_t e = static_cast<size_t>(k) * N + n;
    const uint8_t x = b[e >> 1];
    const int q = (e & 1) ? unpack_high(x) : unpack_low(x);
    const float bv = q * __half2float(scales[static_cast<size_t>(k / group_size) * N + n]);
    acc = fmaf(__half2float(a[static_cast<size_t>(m) * K + k]), bv, acc);
  }
  c[static_cast<size_t>(m) * N + n] = __float2half(acc);
}

}  // namespace

void launch_v00(const half* a, const uint8_t* b, const half* s, half* c,
                int M, int N, int K, int group, cudaStream_t stream) {
  v00_kernel<<<dim3((N + 15) / 16, (M + 15) / 16), dim3(16, 16), 0, stream>>>(a, b, s, c, M, N, K, group);
  CUDA_CHECK(cudaGetLastError());
}

KernelResources query_v00_resources() {
  cudaFuncAttributes a{}; CUDA_CHECK(cudaFuncGetAttributes(&a, v00_kernel));
  int blocks = 0; CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, v00_kernel, 256, 0));
  return {a.numRegs, int(a.sharedSizeBytes), a.maxThreadsPerBlock, blocks, blocks * 256 / 1536.0f};
}

}  // namespace int4gemm
