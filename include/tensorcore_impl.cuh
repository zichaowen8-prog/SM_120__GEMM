#pragma once
#include "common.cuh"
#include <mma.h>

namespace int4gemm {
namespace tc {

namespace wmma = nvcuda::wmma;

template <int BM, int BN, int BK, int WARP_N, int GROUP, bool REPACKED, bool VECTOR_B, bool B_REUSE>
__global__ void tensorcore_kernel(const half* __restrict__ a,
                                  const uint8_t* __restrict__ b,
                                  const half* __restrict__ scales,
                                  half* __restrict__ c,
                                  int M, int N, int K) {
  constexpr int PHYSICAL_BK = 32;
  constexpr int PHYSICAL_BN = 128;
  constexpr int WARP_M = 32;
  constexpr int WARPS_N = BN / WARP_N;
  constexpr int WARPS = (BM / WARP_M) * WARPS_N;
  constexpr int THREADS = WARPS * 32;
  extern __shared__ __align__(16) unsigned char raw[];
  half* as = reinterpret_cast<half*>(raw);
  half* bs = as + BM * BK;
  float* out = reinterpret_cast<float*>(raw);

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int block_m = (B_REUSE ? blockIdx.x : blockIdx.y) * BM;
  const int block_n = (B_REUSE ? blockIdx.y : blockIdx.x) * BN;
  const int nt = block_n / PHYSICAL_BN;
  const int physical_n0 = block_n % PHYSICAL_BN;
  const int ktiles = K / BK;
  const int physical_ktiles = K / PHYSICAL_BK;
  const int groups = K / GROUP;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][WARP_N / 16];
  #pragma unroll
  for (int i = 0; i < 2; ++i)
    #pragma unroll
    for (int j = 0; j < WARP_N / 16; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

  for (int kt = 0; kt < ktiles; ++kt) {
    // A is copied as aligned half2 values. The fixed target dimensions make all
    // rows and tile origins naturally aligned.
    for (int p = tid; p < BM * BK / 2; p += THREADS) {
      const int e = p * 2;
      const int mi = e / BK, ki = e % BK;
      reinterpret_cast<half2*>(as)[p] = *reinterpret_cast<const half2*>(a + static_cast<size_t>(block_m + mi) * K + kt * BK + ki);
    }

    if constexpr (REPACKED && VECTOR_B) {
      static_assert(BK % PHYSICAL_BK == 0, "compute BK must combine physical BK tiles");
      const uint8_t* tile = b + (static_cast<size_t>(nt) * physical_ktiles + kt * (BK / PHYSICAL_BK)) * (PHYSICAL_BK * PHYSICAL_BN / 2);
      constexpr int WORDS_PER_ROW = BN / 16;
      constexpr int WORDS = BK * WORDS_PER_ROW;
      for (int wi = tid; wi < WORDS; wi += THREADS) {
        const int word_ki = wi / WORDS_PER_ROW;
        const int word_n = wi % WORDS_PER_ROW;
        const uint8_t* word_ptr = tile + word_ki * (PHYSICAL_BN / 2) + physical_n0 / 2 + word_n * 8;
        const uint2 word = *reinterpret_cast<const uint2*>(word_ptr);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&word);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int pair = wi * 8 + j;
          const int e = pair * 2;
          const int ki = e / BN, ni = e % BN;
          const uint8_t x = bytes[j];
          const half* sp = scales + (static_cast<size_t>(nt) * groups + (kt * BK + ki) / GROUP) * PHYSICAL_BN + ni;
          reinterpret_cast<half2*>(bs + e)[0] = __floats2half2_rn(unpack_low(x) * __half2float(sp[0]),
                                                                   unpack_high(x) * __half2float(sp[1]));
        }
      }
    } else {
      for (int pair = tid; pair < BK * BN / 2; pair += THREADS) {
        const int e = pair * 2;
        const int ki = e / BN, ni = e % BN;
        uint8_t x;
        const half* sp;
        if constexpr (REPACKED) {
          const size_t tile_base = (static_cast<size_t>(nt) * physical_ktiles + kt * (BK / PHYSICAL_BK)) * (PHYSICAL_BK * PHYSICAL_BN / 2);
          x = b[tile_base + (ki * PHYSICAL_BN + physical_n0 + ni) / 2];
          sp = scales + (static_cast<size_t>(nt) * groups + (kt * BK + ki) / GROUP) * PHYSICAL_BN + physical_n0 + ni;
        } else {
          x = b[(static_cast<size_t>(kt * BK + ki) * N + block_n + ni) >> 1];
          sp = scales + static_cast<size_t>((kt * BK + ki) / GROUP) * N + block_n + ni;
        }
        reinterpret_cast<half2*>(bs + e)[0] = __floats2half2_rn(unpack_low(x) * __half2float(sp[0]),
                                                                 unpack_high(x) * __half2float(sp[1]));
      }
    }
    __syncthreads();

    const int wm = warp / WARPS_N;
    const int wn = warp % WARPS_N;
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[WARP_N / 16];
      #pragma unroll
      for (int i = 0; i < 2; ++i)
        wmma::load_matrix_sync(af[i], as + (wm * WARP_M + i * 16) * BK + kk, BK);
      #pragma unroll
      for (int j = 0; j < WARP_N / 16; ++j)
        wmma::load_matrix_sync(bf[j], bs + kk * BN + wn * WARP_N + j * 16, BN);
      #pragma unroll
      for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < WARP_N / 16; ++j) wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    __syncthreads();
  }

  // WMMA accumulator register layout is intentionally unspecified. Store to a
  // shared FP32 tile, then coalesced-convert to the requested FP16 output.
  const int wm = warp / WARPS_N;
  const int wn = warp % WARPS_N;
  #pragma unroll
  for (int i = 0; i < 2; ++i)
    #pragma unroll
    for (int j = 0; j < WARP_N / 16; ++j)
      wmma::store_matrix_sync(out + (wm * WARP_M + i * 16) * BN + wn * WARP_N + j * 16,
                              acc[i][j], BN, wmma::mem_row_major);
  __syncthreads();
  for (int p = tid; p < BM * BN / 2; p += THREADS) {
    const int e = p * 2;
    const int mi = e / BN, ni = e % BN;
    reinterpret_cast<half2*>(c + static_cast<size_t>(block_m + mi) * N + block_n + ni)[0] =
        __floats2half2_rn(out[e], out[e + 1]);
  }
}

template <int BM, int BN, int BK, int WARP_N, bool REPACKED, bool VECTOR_B, bool B_REUSE = false>
void launch(const half* a, const uint8_t* b, const half* s, half* c,
            int M, int N, int K, int group, cudaStream_t stream) {
  constexpr int threads = (BM / 32) * (BN / WARP_N) * 32;
  constexpr int input_bytes = (BM * BK + BK * BN) * int(sizeof(half));
  constexpr int output_bytes = BM * BN * int(sizeof(float));
  constexpr int smem = input_bytes > output_bytes ? input_bytes : output_bytes;
  auto launch_group = [&](auto group_tag) {
    constexpr int G = decltype(group_tag)::value;
    auto kernel = tensorcore_kernel<BM, BN, BK, WARP_N, G, REPACKED, VECTOR_B, B_REUSE>;
    if constexpr (smem > 48 * 1024) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    dim3 grid = B_REUSE ? dim3(M / BM, N / BN) : dim3(N / BN, M / BM);
    kernel<<<grid, threads, smem, stream>>>(a, b, s, c, M, N, K);
  };
  if (group == 32) launch_group(std::integral_constant<int, 32>{});
  else if (group == 64) launch_group(std::integral_constant<int, 64>{});
  else if (group == 128) launch_group(std::integral_constant<int, 128>{});
  else throw std::runtime_error("group size must be 32, 64, or 128");
  CUDA_CHECK(cudaGetLastError());
}

template <int BM, int BN, int BK, int WARP_N, int GROUP, bool REPACKED, bool VECTOR_B, bool B_REUSE = false>
KernelResources resources() {
  constexpr int threads = (BM / 32) * (BN / WARP_N) * 32;
  constexpr int input_bytes = (BM * BK + BK * BN) * int(sizeof(half));
  constexpr int output_bytes = BM * BN * int(sizeof(float));
  constexpr int smem = input_bytes > output_bytes ? input_bytes : output_bytes;
  auto kernel = tensorcore_kernel<BM, BN, BK, WARP_N, GROUP, REPACKED, VECTOR_B, B_REUSE>;
  if constexpr (smem > 48 * 1024) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaFuncAttributes a{};
  CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
  int blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, smem));
  return {a.numRegs, int(a.sharedSizeBytes), a.maxThreadsPerBlock, blocks, blocks * threads / 1536.0f};
}

}  // namespace tc
}  // namespace int4gemm
