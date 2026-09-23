#include "quant.cuh"

namespace int4gemm {
namespace {

__global__ void find_scales_kernel(const half* __restrict__ b, half* __restrict__ scales,
                                   int K, int N, int group_size) {
  const int n = blockIdx.x;
  const int group = blockIdx.y;
  float vmax = 0.0f;
  for (int k = group * group_size + threadIdx.x;
       k < min(K, (group + 1) * group_size); k += blockDim.x) {
    vmax = fmaxf(vmax, fabsf(__half2float(b[static_cast<size_t>(k) * N + n])));
  }
  __shared__ float reduction[128];
  reduction[threadIdx.x] = vmax;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0) scales[static_cast<size_t>(group) * N + n] = __float2half(fmaxf(reduction[0] / 7.0f, 1.0e-8f));
}

__global__ void pack_kernel(const half* __restrict__ b, const half* __restrict__ scales,
                            uint8_t* __restrict__ packed, size_t bytes, int N, int group_size) {
  size_t byte = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (byte >= bytes) return;
  const size_t e0 = byte * 2;
  const int k = int(e0 / N);
  const int n = int(e0 - static_cast<size_t>(k) * N);
  const float x0 = __half2float(b[e0]);
  const float x1 = __half2float(b[e0 + 1]);
  const float s0 = __half2float(scales[static_cast<size_t>(k / group_size) * N + n]);
  const float s1 = __half2float(scales[static_cast<size_t>(k / group_size) * N + n + 1]);
  int q0 = max(-8, min(7, __float2int_rn(x0 / s0)));
  int q1 = max(-8, min(7, __float2int_rn(x1 / s1)));
  packed[byte] = static_cast<uint8_t>((q0 & 15) | ((q1 & 15) << 4));
}

__global__ void dequant_kernel(const uint8_t* __restrict__ packed, const half* __restrict__ scales,
                               half* __restrict__ b, size_t elements, int N, int group_size) {
  size_t e = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= elements) return;
  const uint8_t byte = packed[e >> 1];
  const int q = (e & 1) ? unpack_high(byte) : unpack_low(byte);
  const int k = int(e / N);
  const int n = int(e - static_cast<size_t>(k) * N);
  b[e] = __float2half(float(q) * __half2float(scales[static_cast<size_t>(k / group_size) * N + n]));
}

}  // namespace

void launch_quantize(const half* b, uint8_t* packed, half* scales,
                     int K, int N, int group_size, cudaStream_t stream) {
  const int groups = (K + group_size - 1) / group_size;
  find_scales_kernel<<<dim3(N, groups), 128, 0, stream>>>(b, scales, K, N, group_size);
  CUDA_CHECK(cudaGetLastError());
  const size_t bytes = static_cast<size_t>(K) * N / 2;
  pack_kernel<<<(bytes + 255) / 256, 256, 0, stream>>>(b, scales, packed, bytes, N, group_size);
  CUDA_CHECK(cudaGetLastError());
}

void launch_dequantize(const uint8_t* packed, const half* scales, half* b,
                       int K, int N, int group_size, cudaStream_t stream) {
  const size_t elements = static_cast<size_t>(K) * N;
  dequant_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(packed, scales, b, elements, N, group_size);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace int4gemm

