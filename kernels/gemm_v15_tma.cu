#include "kernels.cuh"

#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <mma.h>

#include <sstream>

namespace int4gemm {
namespace {
namespace wmma = nvcuda::wmma;
namespace cde = cuda::device::experimental;
using barrier_t = cuda::barrier<cuda::thread_scope_block>;

constexpr int BM = 64;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int THREADS = 128;
constexpr int WARP_M = 32;
constexpr int WARP_N = 64;
constexpr int PACKED_BYTES = BK * BN / 2;
constexpr int OUTPUT_SMEM_BYTES = BM * BN * int(sizeof(float));

template <bool TMA_A, int STAGES, int B_PAD>
__host__ __device__ constexpr int data_smem_bytes() {
  constexpr int a_stride = TMA_A ? BK : BK + 8;
  constexpr int input_bytes = STAGES * (BM * a_stride * int(sizeof(half)) + PACKED_BYTES)
                            + BK * (BN + B_PAD) * int(sizeof(half));
  return input_bytes > OUTPUT_SMEM_BYTES ? input_bytes : OUTPUT_SMEM_BYTES;
}

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(addr), "l"(gmem));
}

__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_wait_one() {
  asm volatile("cp.async.wait_group 1;");
}

template <int GROUP, bool TMA_A, int STAGES, int B_PAD, bool LEADER_WAIT>
__global__ void tma_kernel(const __grid_constant__ CUtensorMap a_map,
                           const __grid_constant__ CUtensorMap b_map,
                           const half* __restrict__ a,
                           const half* __restrict__ scales,
                           half* __restrict__ c, int M, int N, int K) {
  constexpr int A_STRIDE = TMA_A ? BK : BK + 8;
  constexpr int B_STRIDE = BN + B_PAD;
  constexpr int A_HALF = BM * A_STRIDE;
  constexpr int INPUT_BYTES = STAGES * (A_HALF * int(sizeof(half)) + PACKED_BYTES)
                            + BK * B_STRIDE * int(sizeof(half));
  constexpr int DATA_SMEM_BYTES = data_smem_bytes<TMA_A, STAGES, B_PAD>();
  static_assert(INPUT_BYTES <= DATA_SMEM_BYTES);

  extern __shared__ __align__(128) unsigned char raw[];
  half* astage = reinterpret_cast<half*>(raw);
  uint8_t* bpack = raw + STAGES * A_HALF * sizeof(half);
  half* bs = reinterpret_cast<half*>(bpack + STAGES * PACKED_BYTES);
  float* out = reinterpret_cast<float*>(raw);

  // Keep the TMA data region at the 128-byte-aligned dynamic-smem base.  A
  // separate static __shared__ barrier array would shift this base by 16 B
  // on SM120 and make the TMA destination misaligned.
  barrier_t* full = reinterpret_cast<barrier_t*>(raw + DATA_SMEM_BYTES);
  if (threadIdx.x == 0) {
    #pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) init(&full[stage], 1);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int ktiles = K / BK;
  const int groups = K / GROUP;
  const int nt = int(blockIdx.x);
  const int block_m = int(blockIdx.y) * BM;
  const int block_n = nt * BN;
  unsigned phase[STAGES] = {0, 0};

  auto issue_tile = [&](int stage, int kt) {
    half* adst = astage + stage * A_HALF;
    if constexpr (!TMA_A) {
      for (int chunk = tid; chunk < BM * (BK / 8); chunk += THREADS) {
        const int mi = chunk / (BK / 8);
        const int ki = (chunk % (BK / 8)) * 8;
        cp_async_16(adst + mi * A_STRIDE + ki,
                    a + static_cast<size_t>(block_m + mi) * K + kt * BK + ki);
      }
      cp_commit();
    }

    if (tid == 0) {
      if constexpr (TMA_A) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            adst, &a_map, kt * BK, block_m, full[stage]);
      }
      const int b_tile = nt * ktiles + kt;
      cde::cp_async_bulk_tensor_2d_global_to_shared(
          bpack + stage * PACKED_BYTES, &b_map, 0, b_tile, full[stage]);
      constexpr int tx_bytes = PACKED_BYTES + (TMA_A ? BM * BK * int(sizeof(half)) : 0);
      (void)cuda::device::barrier_arrive_tx(full[stage], 1, tx_bytes);
    }
  };

  auto wait_tma = [&](int stage) {
    uint64_t* native = cuda::device::barrier_native_handle(full[stage]);
    if constexpr (LEADER_WAIT) {
      if (tid == 0) {
        while (!cuda::ptx::mbarrier_try_wait_parity(native, phase[stage], 1000000)) {}
        phase[stage] ^= 1;
      }
    } else {
      while (!cuda::ptx::mbarrier_try_wait_parity(native, phase[stage], 1000000)) {}
      phase[stage] ^= 1;
    }
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
  #pragma unroll
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
      wmma::fill_fragment(acc[i][j], 0.0f);

  #pragma unroll
  for (int stage = 0; stage < STAGES; ++stage) issue_tile(stage, stage);
  for (int kt = 0; kt < ktiles; ++kt) {
    if constexpr (!TMA_A) {
      if constexpr (STAGES == 2) cp_wait_one();
      else asm volatile("cp.async.wait_group 2;");
    }
    const int stage = kt % STAGES;
    wait_tma(stage);
    __syncthreads();

    const uint8_t* packed = bpack + stage * PACKED_BYTES;
    for (int pair = tid; pair < PACKED_BYTES; pair += THREADS) {
      const int e = pair * 2;
      const int ki = e / BN;
      const int ni = e - ki * BN;
      const uint8_t x = packed[pair];
      const half* sp = scales
          + (static_cast<size_t>(nt) * groups + (kt * BK + ki) / GROUP) * BN + ni;
      reinterpret_cast<half2*>(bs + ki * B_STRIDE + ni)[0] = __floats2half2_rn(
          unpack_low(x) * __half2float(sp[0]), unpack_high(x) * __half2float(sp[1]));
    }
    __syncthreads();

    const half* as = astage + stage * A_HALF;
    const int wm = warp / 2;
    const int wn = warp & 1;
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[4];
      #pragma unroll
      for (int i = 0; i < 2; ++i)
        wmma::load_matrix_sync(af[i], as + (wm * WARP_M + i * 16) * A_STRIDE + kk,
                               A_STRIDE);
      #pragma unroll
      for (int j = 0; j < 4; ++j)
        wmma::load_matrix_sync(bf[j], bs + kk * B_STRIDE + wn * WARP_N + j * 16,
                               B_STRIDE);
      #pragma unroll
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
          wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    __syncthreads();
    if (kt + STAGES < ktiles) issue_tile(stage, kt + STAGES);
  }
  if constexpr (!TMA_A) asm volatile("cp.async.wait_all;");
  __syncthreads();

  const int wm = warp / 2;
  const int wn = warp & 1;
  #pragma unroll
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
      wmma::store_matrix_sync(out + (wm * WARP_M + i * 16) * BN
                                  + wn * WARP_N + j * 16,
                              acc[i][j], BN, wmma::mem_row_major);
  __syncthreads();
  for (int p = tid; p < BM * BN / 2; p += THREADS) {
    const int e = p * 2;
    const int mi = e / BN;
    const int ni = e - mi * BN;
    reinterpret_cast<half2*>(c + static_cast<size_t>(block_m + mi) * N + block_n + ni)[0]
        = __floats2half2_rn(out[e], out[e + 1]);
  }
}

void check_driver(CUresult status, const char* operation) {
  if (status == CUDA_SUCCESS) return;
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &message);
  std::ostringstream out;
  out << operation << " failed: " << (name ? name : "unknown")
      << " (" << (message ? message : "no description") << ')';
  throw std::runtime_error(out.str());
}

struct TensorMapCache {
  const half* a = nullptr;
  const uint8_t* b = nullptr;
  int M = 0;
  int N = 0;
  int K = 0;
  CUtensorMap a_map{};
  CUtensorMap b_map{};
};

const TensorMapCache& tensor_maps(const half* a, const uint8_t* b, int M, int N, int K) {
  static thread_local TensorMapCache cache;
  if (cache.a == a && cache.b == b && cache.M == M && cache.N == N && cache.K == K)
    return cache;

  const cuuint64_t a_dims[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(M)};
  const cuuint64_t a_strides[1] = {static_cast<cuuint64_t>(K) * sizeof(half)};
  const cuuint32_t a_box[2] = {BK, BM};
  const cuuint32_t a_element_strides[2] = {1, 1};
  check_driver(cuTensorMapEncodeTiled(
      &cache.a_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, const_cast<half*>(a),
      a_dims, a_strides, a_box, a_element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "cuTensorMapEncodeTiled(A)");

  // The repacked layout is [N/BN][K/BK][PACKED_BYTES].  Express each
  // contiguous 2 KiB tile as 256 uint64_t elements.  A rank-2 map also
  // avoids the driver restriction hit by the equivalent rank-1 encoding.
  const cuuint64_t b_dims[2] = {
      PACKED_BYTES / static_cast<cuuint64_t>(sizeof(uint64_t)),
      static_cast<cuuint64_t>(K / BK) * static_cast<cuuint64_t>(N / BN)};
  const cuuint64_t b_strides[1] = {PACKED_BYTES};
  const cuuint32_t b_box[2] = {
      PACKED_BYTES / static_cast<cuuint32_t>(sizeof(uint64_t)), 1};
  const cuuint32_t b_element_strides[2] = {1, 1};
  check_driver(cuTensorMapEncodeTiled(
      &cache.b_map, CU_TENSOR_MAP_DATA_TYPE_UINT64, 2, const_cast<uint8_t*>(b),
      b_dims, b_strides, b_box, b_element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "cuTensorMapEncodeTiled(B)");

  cache.a = a;
  cache.b = b;
  cache.M = M;
  cache.N = N;
  cache.K = K;
  return cache;
}

template <bool TMA_A, int STAGES, int B_PAD, bool LEADER_WAIT = false>
void launch_tma(const half* a, const uint8_t* b, const half* scales, half* c,
                int M, int N, int K, int group, cudaStream_t stream) {
  const TensorMapCache& maps = tensor_maps(a, b, M, N, K);
  constexpr int smem_bytes = data_smem_bytes<TMA_A, STAGES, B_PAD>()
                           + STAGES * int(sizeof(barrier_t));
  auto launch_group = [&](auto group_tag) {
    constexpr int G = decltype(group_tag)::value;
    tma_kernel<G, TMA_A, STAGES, B_PAD, LEADER_WAIT>
        <<<dim3(N / BN, M / BM), THREADS, smem_bytes, stream>>>(
        maps.a_map, maps.b_map, a, scales, c, M, N, K);
  };
  if (group == 32) launch_group(std::integral_constant<int, 32>{});
  else if (group == 64) launch_group(std::integral_constant<int, 64>{});
  else launch_group(std::integral_constant<int, 128>{});
  CUDA_CHECK(cudaGetLastError());
}

template <int GROUP, bool TMA_A, int STAGES, int B_PAD, bool LEADER_WAIT = false>
KernelResources tma_resources() {
  auto fn = tma_kernel<GROUP, TMA_A, STAGES, B_PAD, LEADER_WAIT>;
  constexpr int smem_bytes = data_smem_bytes<TMA_A, STAGES, B_PAD>()
                           + STAGES * int(sizeof(barrier_t));
  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, fn));
  int blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, fn, THREADS, smem_bytes));
  return {attributes.numRegs, int(attributes.sharedSizeBytes),
          attributes.maxThreadsPerBlock, blocks, blocks * THREADS / 1536.0f};
}

}  // namespace

void launch_v15_tma_b(const half* a, const uint8_t* b, const half* scales, half* c,
                      int M, int N, int K, int group, cudaStream_t stream) {
  launch_tma<false, 2, 24>(a, b, scales, c, M, N, K, group, stream);
}

void launch_v15_tma_ab(const half* a, const uint8_t* b, const half* scales, half* c,
                       int M, int N, int K, int group, cudaStream_t stream) {
  launch_tma<true, 2, 24>(a, b, scales, c, M, N, K, group, stream);
}

void launch_v15_tma_ab_stage3(const half* a, const uint8_t* b, const half* scales, half* c,
                              int M, int N, int K, int group, cudaStream_t stream) {
  launch_tma<true, 3, 24>(a, b, scales, c, M, N, K, group, stream);
}

void launch_v15_tma_ab_stage4(const half* a, const uint8_t* b, const half* scales, half* c,
                              int M, int N, int K, int group, cudaStream_t stream) {
  launch_tma<true, 4, 0>(a, b, scales, c, M, N, K, group, stream);
}

void launch_v15_tma_ab_leader_wait(const half* a, const uint8_t* b, const half* scales, half* c,
                                   int M, int N, int K, int group, cudaStream_t stream) {
  launch_tma<true, 3, 24, true>(a, b, scales, c, M, N, K, group, stream);
}

KernelResources query_v15_resources(const std::string& version, int group) {
#define PICK(G, TA, S, BP) tma_resources<G, TA, S, BP>()
#define PICK_LEADER(G) tma_resources<G, true, 3, 24, true>()
  const bool tma_a = version != "v15a";
  const bool stage3 = version == "v15c";
  const bool stage4 = version == "v15d";
  const bool leader_wait = version == "v15e";
  if (tma_a) {
    if (leader_wait) {
      if (group == 32) return PICK_LEADER(32);
      if (group == 64) return PICK_LEADER(64);
      return PICK_LEADER(128);
    }
    if (stage4) {
      if (group == 32) return PICK(32, true, 4, 0);
      if (group == 64) return PICK(64, true, 4, 0);
      return PICK(128, true, 4, 0);
    }
    if (stage3) {
      if (group == 32) return PICK(32, true, 3, 24);
      if (group == 64) return PICK(64, true, 3, 24);
      return PICK(128, true, 3, 24);
    }
    if (group == 32) return PICK(32, true, 2, 24);
    if (group == 64) return PICK(64, true, 2, 24);
    return PICK(128, true, 2, 24);
  }
  if (group == 32) return PICK(32, false, 2, 24);
  if (group == 64) return PICK(64, false, 2, 24);
  return PICK(128, false, 2, 24);
#undef PICK
#undef PICK_LEADER
}

}  // namespace int4gemm
