#include "quant.cuh"
#include "layout.cuh"

namespace int4gemm {
namespace {

constexpr int BN = 128;
constexpr int BK = 32;

__global__ void repack_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                              size_t bytes, int K, int N) {
  size_t out_byte = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out_byte >= bytes) return;
  const size_t tile_elements = BK * BN;
  const size_t e0 = out_byte * 2;
  const size_t tile = e0 / tile_elements;
  const int within = int(e0 - tile * tile_elements);
  const int nt = int(tile / (K / BK));
  const int kt = int(tile - static_cast<size_t>(nt) * (K / BK));
  const int ki = within / BN;
  const int ni = within - ki * BN;
  const int k = kt * BK + ki;
  const int n = nt * BN + ni;
  const uint8_t a = src[packed_row_byte(k, n, N)];
  const uint8_t b = src[packed_row_byte(k, n + 1, N)];
  const int q0 = (n & 1) ? unpack_high(a) : unpack_low(a);
  const int q1 = ((n + 1) & 1) ? unpack_high(b) : unpack_low(b);
  dst[out_byte] = static_cast<uint8_t>((q0 & 15) | ((q1 & 15) << 4));
}

__global__ void repack_scales_kernel(const half* __restrict__ src, half* __restrict__ dst,
                                     size_t elements, int N, int groups) {
  size_t out = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out >= elements) return;
  const int ni = int(out % BN);
  const size_t x = out / BN;
  const int group = int(x % groups);
  const int nt = int(x / groups);
  const int n = nt * BN + ni;
  dst[out] = src[static_cast<size_t>(group) * N + n];
}

__global__ void inverse_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                               size_t bytes, int K, int N) {
  size_t byte = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (byte >= bytes) return;
  const size_t e0 = byte * 2;
  const int k = int(e0 / N);
  const int n = int(e0 - static_cast<size_t>(k) * N);
  const size_t r0 = repacked_element<BN, BK>(k, n, N, K);
  const size_t r1 = repacked_element<BN, BK>(k, n + 1, N, K);
  const uint8_t a = src[r0 >> 1];
  const uint8_t b = src[r1 >> 1];
  const int q0 = (r0 & 1) ? unpack_high(a) : unpack_low(a);
  const int q1 = (r1 & 1) ? unpack_high(b) : unpack_low(b);
  dst[byte] = static_cast<uint8_t>((q0 & 15) | ((q1 & 15) << 4));
}

__device__ __forceinline__ uint16_t load_q4(const uint8_t* src, int k, int n,
                                             int N) {
  const size_t logical = static_cast<size_t>(k) * N + n;
  const uint8_t byte = src[logical >> 1];
  return (logical & 1) ? uint16_t(byte >> 4) : uint16_t(byte & 0x0f);
}

__global__ void repack_mma_kernel(const uint8_t* __restrict__ src,
                                  uint16_t* __restrict__ dst,
                                  size_t words, int K, int N) {
  const size_t out = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out >= words) return;
  constexpr int WORDS_PER_TILE = BK * BN / 4;
  const size_t tile = out / WORDS_PER_TILE;
  const int within = int(out - tile * WORDS_PER_TILE);
  const int lane = within & 31;
  const int n8 = (within >> 5) & 7;
  const int warp_n = (within >> 8) & 1;
  const int k16 = within >> 9;
  const int nt = int(tile / (K / BK));
  const int kt = int(tile - static_cast<size_t>(nt) * (K / BK));
  const int group_id = lane >> 2;
  const int thread_in_group = lane & 3;
  const int n = nt * BN + warp_n * 64 + n8 * 8 + group_id;
  const int k0 = kt * BK + k16 * 16 + thread_in_group * 2;
  const uint16_t q0 = load_q4(src, k0, n, N);
  const uint16_t q1 = load_q4(src, k0 + 1, n, N);
  const uint16_t q2 = load_q4(src, k0 + 8, n, N);
  const uint16_t q3 = load_q4(src, k0 + 9, n, N);
  dst[out] = q0 | (q1 << 4) | (q2 << 8) | (q3 << 12);
}

__device__ __forceinline__ uint16_t make_mma_fragment(
    const uint8_t* src, int k0, int n, int N) {
  const uint16_t q0 = load_q4(src, k0, n, N);
  const uint16_t q1 = load_q4(src, k0 + 1, n, N);
  const uint16_t q2 = load_q4(src, k0 + 8, n, N);
  const uint16_t q3 = load_q4(src, k0 + 9, n, N);
  return q0 | (q1 << 4) | (q2 << 8) | (q3 << 12);
}

__global__ void repack_mma_pair_kernel(const uint8_t* __restrict__ src,
                                       uint32_t* __restrict__ dst,
                                       size_t pairs, int K, int N) {
  const size_t out = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out >= pairs) return;
  constexpr int PAIRS_PER_TILE = BK * BN / 8;
  const size_t tile = out / PAIRS_PER_TILE;
  const int within = int(out - tile * PAIRS_PER_TILE);
  const int lane = within & 31;
  const int n8 = (within >> 5) & 7;
  const int warp_n = within >> 8;
  const int nt = int(tile / (K / BK));
  const int kt = int(tile - static_cast<size_t>(nt) * (K / BK));
  const int group_id = lane >> 2;
  const int thread_in_group = lane & 3;
  const int n = nt * BN + warp_n * 64 + n8 * 8 + group_id;
  const int k0 = kt * BK + thread_in_group * 2;
  const uint32_t low = make_mma_fragment(src, k0, n, N);
  const uint32_t high = make_mma_fragment(src, k0 + 16, n, N);
  dst[out] = low | (high << 16);
}

}  // namespace

void launch_repack_128x32(const uint8_t* packed, const half* scales,
                          uint8_t* repacked, half* repacked_scales,
                          int K, int N, int group_size, cudaStream_t stream) {
  if ((K % BK) || (N % BN)) throw std::runtime_error("repack requires K%32==0 and N%128==0");
  const size_t bytes = static_cast<size_t>(K) * N / 2;
  repack_kernel<<<(bytes + 255) / 256, 256, 0, stream>>>(packed, repacked, bytes, K, N);
  CUDA_CHECK(cudaGetLastError());
  const int groups = K / group_size;
  const size_t scale_count = static_cast<size_t>(groups) * N;
  repack_scales_kernel<<<(scale_count + 255) / 256, 256, 0, stream>>>(scales, repacked_scales, scale_count, N, groups);
  CUDA_CHECK(cudaGetLastError());
}

void launch_inverse_repack_128x32(const uint8_t* repacked, uint8_t* packed,
                                  int K, int N, cudaStream_t stream) {
  const size_t bytes = static_cast<size_t>(K) * N / 2;
  inverse_kernel<<<(bytes + 255) / 256, 256, 0, stream>>>(repacked, packed, bytes, K, N);
  CUDA_CHECK(cudaGetLastError());
}

void launch_repack_mma_128x32(const uint8_t* packed, uint8_t* repacked,
                              int K, int N, cudaStream_t stream) {
  if ((K % BK) || (N % BN))
    throw std::runtime_error("MMA repack requires K%32==0 and N%128==0");
  const size_t words = static_cast<size_t>(K) * N / 4;
  repack_mma_kernel<<<(words + 255) / 256, 256, 0, stream>>>(
      packed, reinterpret_cast<uint16_t*>(repacked), words, K, N);
  CUDA_CHECK(cudaGetLastError());
}

void launch_repack_mma_pair_128x32(const uint8_t* packed, uint8_t* repacked,
                                   int K, int N, cudaStream_t stream) {
  if ((K % BK) || (N % BN))
    throw std::runtime_error("MMA pair repack requires K%32==0 and N%128==0");
  const size_t pairs = static_cast<size_t>(K) * N / 8;
  repack_mma_pair_kernel<<<(pairs + 255) / 256, 256, 0, stream>>>(
      packed, reinterpret_cast<uint32_t*>(repacked), pairs, K, N);
  CUDA_CHECK(cudaGetLastError());
}

void launch_repack_scales_128(const half* scales, half* repacked_scales,
                              int K, int N, int group_size,
                              cudaStream_t stream) {
  if ((K % group_size) || (N % BN))
    throw std::runtime_error("scale repack requires K%group_size==0 and N%128==0");
  const int groups = K / group_size;
  const size_t scale_count = static_cast<size_t>(groups) * N;
  repack_scales_kernel<<<(scale_count + 255) / 256, 256, 0, stream>>>(
      scales, repacked_scales, scale_count, N, groups);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace int4gemm
