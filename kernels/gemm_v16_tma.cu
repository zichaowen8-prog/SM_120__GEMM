#include "kernels.cuh"
#include "layout.cuh"

#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>

#include <sstream>

namespace int4gemm {
namespace {

namespace cde = cuda::device::experimental;
using barrier_t = cuda::barrier<cuda::thread_scope_block>;

constexpr int BM = 64;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WARP_M = 32;
constexpr int WARP_N = 64;
constexpr int COMPUTE_WARPS = 4;
constexpr int A_HALF = BM * BK;
constexpr int A_BYTES = A_HALF * int(sizeof(half));
constexpr int PACKED_BYTES = BK * BN / 2;
constexpr int SCALE_BYTES = BN * int(sizeof(half));
constexpr int STAGE_BYTES = A_BYTES + PACKED_BYTES + SCALE_BYTES;

struct AFragment { uint32_t x[4]; };
struct BFragment { uint32_t x[2]; };
struct AccFragment { float x[4]; };

template <int SWIZZLE_CHUNKS, int ROW_BYTES>
__device__ __forceinline__ const unsigned char* swizzled_ptr(
    const unsigned char* base, int logical_byte) {
  if constexpr (SWIZZLE_CHUNKS == 1) {
    return base + logical_byte;
  } else {
    // TMA swizzling always uses 16-byte atoms and takes the XOR selector
    // from the 128-byte row number.  In 64-byte mode address bit 6 is
    // deliberately preserved; treating each 64-byte span as a new row is
    // therefore incorrect.  Including the shared-base row implements the
    // documented pointer-offset rule for buffers that are 128-byte aligned
    // but not aligned to the full swizzle repetition interval.
    const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(base));
    const int selector = ((shared + logical_byte) >> 7) & (SWIZZLE_CHUNKS - 1);
    const int physical_byte = logical_byte ^ (selector << 4);
    (void)ROW_BYTES;
    return base + physical_byte;
  }
}

template <bool SWIZZLE>
__device__ __forceinline__ AFragment load_a_fragment(const unsigned char* tile,
                                                      int row, int kk, int lane) {
  constexpr int CHUNKS = SWIZZLE ? 4 : 1;
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int col = kk + thread_in_group * 2;
  auto load = [&](int r, int c) {
    const int logical_byte = (r * BK + c) * int(sizeof(half));
    return *reinterpret_cast<const uint32_t*>(
        swizzled_ptr<CHUNKS, 64>(tile, logical_byte));
  };
  return {{load(row + group, col), load(row + group + 8, col),
           load(row + group, col + 8), load(row + group + 8, col + 8)}};
}

__device__ __forceinline__ uint32_t sub_f16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ uint32_t mul_f16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ BFragment decode_b_fragment(uint16_t q, half scale) {
  uint32_t q01 = (uint32_t(q) & 0x000fu) | ((uint32_t(q) & 0x00f0u) << 12);
  uint32_t q23 = ((uint32_t(q) & 0x0f00u) >> 8) | ((uint32_t(q) & 0xf000u) << 4);
  q01 = (q01 ^ 0x00080008u) | 0x64006400u;
  q23 = (q23 ^ 0x00080008u) | 0x64006400u;
  q01 = sub_f16x2(q01, 0x64086408u);
  q23 = sub_f16x2(q23, 0x64086408u);
  const uint32_t sh = __half_as_ushort(scale);
  const uint32_t scale2 = sh | (sh << 16);
  return {{mul_f16x2(q01, scale2), mul_f16x2(q23, scale2)}};
}

__device__ __forceinline__ void mma_m16n8k16(AccFragment& d,
                                              const AFragment& a,
                                              const BFragment& b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(d.x[0]), "+f"(d.x[1]), "+f"(d.x[2]), "+f"(d.x[3])
      : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]),
        "r"(b.x[0]), "r"(b.x[1]));
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
  const half* scales = nullptr;
  int M = 0, N = 0, K = 0, group = 0;
  bool swizzle = false;
  CUtensorMap a_map{}, b_map{}, scale_map{};
};

const TensorMapCache& tensor_maps(const half* a, const uint8_t* b,
                                  const half* scales, int M, int N, int K,
                                  int group, bool swizzle) {
  static thread_local TensorMapCache cache;
  if (cache.a == a && cache.b == b && cache.scales == scales &&
      cache.M == M && cache.N == N && cache.K == K && cache.group == group &&
      cache.swizzle == swizzle)
    return cache;

  const auto a_swizzle = swizzle ? CU_TENSOR_MAP_SWIZZLE_64B
                                 : CU_TENSOR_MAP_SWIZZLE_NONE;
  const auto wide_swizzle = swizzle ? CU_TENSOR_MAP_SWIZZLE_128B
                                    : CU_TENSOR_MAP_SWIZZLE_NONE;
  const cuuint64_t a_dims[2] = {static_cast<cuuint64_t>(K),
                                static_cast<cuuint64_t>(M)};
  const cuuint64_t a_strides[1] = {static_cast<cuuint64_t>(K) * sizeof(half)};
  const cuuint32_t a_box[2] = {BK, BM};
  const cuuint32_t unit_strides[2] = {1, 1};
  check_driver(cuTensorMapEncodeTiled(
      &cache.a_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, const_cast<half*>(a),
      a_dims, a_strides, a_box, unit_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      a_swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "cuTensorMapEncodeTiled(V16 A)");

  // View every 2 KiB packed tile as sixteen 128-byte rows so a 128-byte
  // TMA swizzle is legal and directly addressable by the fragment consumer.
  const cuuint64_t b_dims[2] = {
      16, static_cast<cuuint64_t>(K / BK) * (N / BN) * 16};
  const cuuint64_t b_strides[1] = {128};
  const cuuint32_t b_box[2] = {16, 16};
  check_driver(cuTensorMapEncodeTiled(
      &cache.b_map, CU_TENSOR_MAP_DATA_TYPE_UINT64, 2,
      const_cast<uint8_t*>(b), b_dims, b_strides, b_box, unit_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, wide_swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled(V16 B)");

  // One 128-element scale row is represented as two 128-byte rows.
  const int scale_rows = (N / BN) * (K / group) * 2;
  const cuuint64_t s_dims[2] = {64, static_cast<cuuint64_t>(scale_rows)};
  const cuuint64_t s_strides[1] = {128};
  const cuuint32_t s_box[2] = {64, 2};
  check_driver(cuTensorMapEncodeTiled(
      &cache.scale_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
      const_cast<half*>(scales), s_dims, s_strides, s_box, unit_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, wide_swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled(V16 scales)");

  cache.a = a; cache.b = b; cache.scales = scales;
  cache.M = M; cache.N = N; cache.K = K; cache.group = group;
  cache.swizzle = swizzle;
  return cache;
}

template <int GROUP, int STAGES, bool SWIZZLE, bool DMA_WARP>
__global__ void tma_register_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map,
    const __grid_constant__ CUtensorMap scale_map,
    half* __restrict__ c, int N, int K) {
  constexpr int THREADS = (COMPUTE_WARPS + (DMA_WARP ? 1 : 0)) * 32;
  constexpr int DATA_BYTES = STAGES * STAGE_BYTES;
  extern __shared__ __align__(128) unsigned char raw[];
  unsigned char* astage = raw;
  unsigned char* bstage = astage + STAGES * A_BYTES;
  unsigned char* sstage = bstage + STAGES * PACKED_BYTES;
  barrier_t* full = reinterpret_cast<barrier_t*>(raw + DATA_BYTES);

  const int tid = threadIdx.x;
  if (tid == 0) {
    #pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) init(&full[stage], 1);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  const int warp = tid >> 5;
  const int compute_warp = warp - (DMA_WARP ? 1 : 0);
  const bool compute = compute_warp >= 0 && compute_warp < COMPUTE_WARPS;
  const int lane = tid & 31;
  const int wm = compute_warp >> 1;
  const int wn = compute_warp & 1;
  const int nt = int(blockIdx.x);
  const int block_m = int(blockIdx.y) * BM;
  const int block_n = nt * BN;
  const int ktiles = K / BK;
  const int groups = K / GROUP;
  unsigned phase[STAGES] = {};

  auto issue_tile = [&](int stage, int kt) {
    if (tid == 0) {
      cde::cp_async_bulk_tensor_2d_global_to_shared(
          astage + stage * A_BYTES, &a_map, kt * BK, block_m, full[stage]);
      cde::cp_async_bulk_tensor_2d_global_to_shared(
          bstage + stage * PACKED_BYTES, &b_map, 0,
          (nt * ktiles + kt) * 16, full[stage]);
      cde::cp_async_bulk_tensor_2d_global_to_shared(
          sstage + stage * SCALE_BYTES, &scale_map, 0,
          (nt * groups + (kt * BK) / GROUP) * 2, full[stage]);
      (void)cuda::device::barrier_arrive_tx(full[stage], 1, STAGE_BYTES);
    }
  };

  AccFragment acc[WARP_M / 16][WARP_N / 8];
  #pragma unroll
  for (int i = 0; i < WARP_M / 16; ++i)
    #pragma unroll
    for (int j = 0; j < WARP_N / 8; ++j)
      acc[i][j] = {{0.0f, 0.0f, 0.0f, 0.0f}};

  #pragma unroll
  for (int stage = 0; stage < STAGES; ++stage) issue_tile(stage, stage);

  for (int kt = 0; kt < ktiles; ++kt) {
    const int stage = kt % STAGES;
    if (tid == 0) {
      uint64_t* native = cuda::device::barrier_native_handle(full[stage]);
      while (!cuda::ptx::mbarrier_try_wait_parity(
          native, phase[stage], 1000000)) {}
      phase[stage] ^= 1;
    }
    __syncthreads();

    if (compute) {
      const unsigned char* atile = astage + stage * A_BYTES;
      const unsigned char* btile = bstage + stage * PACKED_BYTES;
      const unsigned char* stile = sstage + stage * SCALE_BYTES;
      #pragma unroll
      for (int k16 = 0; k16 < 2; ++k16) {
        AFragment af[WARP_M / 16];
        #pragma unroll
        for (int i = 0; i < WARP_M / 16; ++i)
          af[i] = load_a_fragment<SWIZZLE>(
              atile, wm * WARP_M + i * 16, k16 * 16, lane);
        #pragma unroll
        for (int j = 0; j < WARP_N / 8; ++j) {
          const int n = wn * WARP_N + j * 8 + (lane >> 2);
          const int word = mma_fragment_word(k16, wn, j, lane);
          constexpr int B_CHUNKS = SWIZZLE ? 8 : 1;
          constexpr int S_CHUNKS = SWIZZLE ? 8 : 1;
          const uint16_t q = *reinterpret_cast<const uint16_t*>(
              swizzled_ptr<B_CHUNKS, 128>(btile, word * int(sizeof(uint16_t))));
          const half scale = *reinterpret_cast<const half*>(
              swizzled_ptr<S_CHUNKS, 128>(stile, n * int(sizeof(half))));
          const BFragment bf = decode_b_fragment(q, scale);
          #pragma unroll
          for (int i = 0; i < WARP_M / 16; ++i)
            mma_m16n8k16(acc[i][j], af[i], bf);
        }
      }
    }
    __syncthreads();
    if (kt + STAGES < ktiles) issue_tile(stage, kt + STAGES);
  }

  if (compute) {
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    #pragma unroll
    for (int i = 0; i < WARP_M / 16; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_N / 8; ++j) {
        const int row0 = block_m + wm * WARP_M + i * 16 + group;
        const int col = block_n + wn * WARP_N + j * 8 + thread_in_group * 2;
        *reinterpret_cast<half2*>(c + static_cast<size_t>(row0) * N + col) =
            __floats2half2_rn(acc[i][j].x[0], acc[i][j].x[1]);
        *reinterpret_cast<half2*>(c + static_cast<size_t>(row0 + 8) * N + col) =
            __floats2half2_rn(acc[i][j].x[2], acc[i][j].x[3]);
      }
    }
  }
  (void)THREADS;
}

template <int STAGES, bool SWIZZLE, bool DMA_WARP>
void launch_tma_register(const half* a, const uint8_t* b, const half* scales,
                         half* c, int M, int N, int K, int group,
                         cudaStream_t stream) {
  const TensorMapCache& maps = tensor_maps(a, b, scales, M, N, K, group, SWIZZLE);
  constexpr int threads = (COMPUTE_WARPS + (DMA_WARP ? 1 : 0)) * 32;
  constexpr int smem = STAGES * (STAGE_BYTES + int(sizeof(barrier_t)));
  auto launch = [&](auto group_tag) {
    constexpr int G = decltype(group_tag)::value;
    tma_register_kernel<G, STAGES, SWIZZLE, DMA_WARP>
        <<<dim3(N / BN, M / BM), threads, smem, stream>>>(
            maps.a_map, maps.b_map, maps.scale_map, c, N, K);
  };
  if (group == 32) launch(std::integral_constant<int, 32>{});
  else if (group == 64) launch(std::integral_constant<int, 64>{});
  else launch(std::integral_constant<int, 128>{});
  CUDA_CHECK(cudaGetLastError());
}

template <int GROUP, int STAGES, bool SWIZZLE, bool DMA_WARP>
KernelResources resources() {
  constexpr int threads = (COMPUTE_WARPS + (DMA_WARP ? 1 : 0)) * 32;
  constexpr int smem = STAGES * (STAGE_BYTES + int(sizeof(barrier_t)));
  auto fn = tma_register_kernel<GROUP, STAGES, SWIZZLE, DMA_WARP>;
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr, fn));
  int blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, fn, threads, smem));
  return {attr.numRegs, int(attr.sharedSizeBytes), attr.maxThreadsPerBlock,
          blocks, blocks * threads / 1536.0f};
}

template <int STAGES, bool SWIZZLE, bool DMA_WARP>
KernelResources pick_resources(int group) {
  if (group == 32) return resources<32, STAGES, SWIZZLE, DMA_WARP>();
  if (group == 64) return resources<64, STAGES, SWIZZLE, DMA_WARP>();
  return resources<128, STAGES, SWIZZLE, DMA_WARP>();
}

}  // namespace

void launch_v16c0_tma(const half* a, const uint8_t* b, const half* scales,
                      half* c, int M, int N, int K, int group,
                      cudaStream_t stream) {
  launch_tma_register<2, false, false>(a, b, scales, c, M, N, K, group, stream);
}

void launch_v16c1_tma_swizzle(const half* a, const uint8_t* b,
                              const half* scales, half* c,
                              int M, int N, int K, int group,
                              cudaStream_t stream) {
  launch_tma_register<2, true, false>(a, b, scales, c, M, N, K, group, stream);
}

void launch_v16c2_tma_swizzle_stage3(const half* a, const uint8_t* b,
                                     const half* scales, half* c,
                                     int M, int N, int K, int group,
                                     cudaStream_t stream) {
  launch_tma_register<3, true, false>(a, b, scales, c, M, N, K, group, stream);
}

void launch_v16c3_tma_dma_warp(const half* a, const uint8_t* b,
                               const half* scales, half* c,
                               int M, int N, int K, int group,
                               cudaStream_t stream) {
  launch_tma_register<3, true, true>(a, b, scales, c, M, N, K, group, stream);
}

void launch_v16c4_tma_stage3(const half* a, const uint8_t* b,
                             const half* scales, half* c,
                             int M, int N, int K, int group,
                             cudaStream_t stream) {
  launch_tma_register<3, false, false>(a, b, scales, c, M, N, K, group, stream);
}

void launch_v16c5_tma_stage4(const half* a, const uint8_t* b,
                             const half* scales, half* c,
                             int M, int N, int K, int group,
                             cudaStream_t stream) {
  launch_tma_register<4, false, false>(a, b, scales, c, M, N, K, group, stream);
}

KernelResources query_v16c_resources(const std::string& version, int group) {
  if (version == "v16c0") return pick_resources<2, false, false>(group);
  if (version == "v16c1") return pick_resources<2, true, false>(group);
  if (version == "v16c2") return pick_resources<3, true, false>(group);
  if (version == "v16c3") return pick_resources<3, true, true>(group);
  if (version == "v16c4") return pick_resources<3, false, false>(group);
  if (version == "v16c5") return pick_resources<4, false, false>(group);
  throw std::runtime_error("unknown V16c kernel: " + version);
}

}  // namespace int4gemm
