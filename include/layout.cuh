#pragma once
#include "common.cuh"

namespace int4gemm {

// Logical row-major linear element e=k*N+n. Even e is the low nibble;
// odd e is the high nibble. N and BN are even in all supported experiments.
__host__ __device__ constexpr size_t packed_row_byte(int k, int n, int N) {
  return (static_cast<size_t>(k) * N + n) >> 1;
}

template <int BN, int BK>
__host__ __device__ constexpr size_t repacked_element(int k, int n, int N, int K) {
  const int nt = n / BN;
  const int kt = k / BK;
  const int ni = n % BN;
  const int ki = k % BK;
  const int k_tiles = K / BK;
  return (static_cast<size_t>(nt) * k_tiles + kt) * (BK * BN) + ki * BN + ni;
}

template <int BN, int BK>
__host__ __device__ constexpr size_t repacked_byte(int k, int n, int N, int K) {
  return repacked_element<BN, BK>(k, n, N, K) >> 1;
}

template <int BN>
__host__ __device__ constexpr size_t repacked_scale_index(int group, int n, int N, int groups) {
  const int nt = n / BN;
  const int ni = n % BN;
  return (static_cast<size_t>(nt) * groups + group) * BN + ni;
}

// One uint16_t is exactly the four B operands consumed by one lane of
// mma.sync.m16n8k16.  A 32x128 CTA tile contains 1024 such words:
// [k16=2][warp_n=2][n8=8][lane=32].
__host__ __device__ constexpr int mma_fragment_word(int k16, int warp_n,
                                                     int n8, int lane) {
  return (((k16 * 2 + warp_n) * 8 + n8) * 32 + lane);
}

// One uint32_t contains the two k16 fragments consumed by the same lane.
// A BN128/BK32 tile is [warp_n=2][n8=8][lane=32], with k16=0 in the low
// half and k16=1 in the high half.
__host__ __device__ constexpr int mma_fragment_pair_word(int warp_n, int n8,
                                                          int lane) {
  return ((warp_n * 8 + n8) * 32 + lane);
}

}  // namespace int4gemm
