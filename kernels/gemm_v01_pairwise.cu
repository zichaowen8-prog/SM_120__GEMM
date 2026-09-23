#include "kernels.cuh"

namespace int4gemm {
namespace {

__global__ void v01_kernel(const half* __restrict__ a, const uint8_t* __restrict__ b,
                           const half* __restrict__ scales, half* __restrict__ c,
                           int M, int N, int K, int group_size) {
  const size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t pairs_per_row = N / 2;
  if (pair >= static_cast<size_t>(M) * pairs_per_row) return;
  const int m = int(pair / pairs_per_row);
  const int n = int((pair - static_cast<size_t>(m) * pairs_per_row) * 2);
  float acc0 = 0.0f, acc1 = 0.0f;
  const half* ap = a + static_cast<size_t>(m) * K;
  for (int k = 0; k < K; ++k) {
    const uint8_t x = b[(static_cast<size_t>(k) * N + n) >> 1];
    const half* sp = scales + static_cast<size_t>(k / group_size) * N + n;
    const float av = __half2float(ap[k]);
    acc0 = fmaf(av, unpack_low(x) * __half2float(sp[0]), acc0);
    acc1 = fmaf(av, unpack_high(x) * __half2float(sp[1]), acc1);
  }
  reinterpret_cast<half2*>(c + static_cast<size_t>(m) * N + n)[0] = __floats2half2_rn(acc0, acc1);
}

}  // namespace

void launch_v01(const half* a, const uint8_t* b, const half* s, half* c,
                int M, int N, int K, int group, cudaStream_t stream) {
  const size_t pairs = static_cast<size_t>(M) * N / 2;
  v01_kernel<<<(pairs + 255) / 256, 256, 0, stream>>>(a, b, s, c, M, N, K, group);
  CUDA_CHECK(cudaGetLastError());
}

KernelResources query_v01_resources() {
  cudaFuncAttributes a{}; CUDA_CHECK(cudaFuncGetAttributes(&a, v01_kernel));
  int blocks = 0; CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, v01_kernel, 256, 0));
  return {a.numRegs, int(a.sharedSizeBytes), a.maxThreadsPerBlock, blocks, blocks * 256 / 1536.0f};
}

}  // namespace int4gemm
