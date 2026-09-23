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

__device__ __forceinline__ void mbarrier_arrive_discard(barrier_t* barrier) {
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], 1;"
               : : "r"(address) : "memory");
}

// The cuda::ptx wrapper takes a native uint64_t* handle, which can extend a
// generic pointer's live range across the polling loop.  The PTX instruction
// consumes a 32-bit shared address, so use it directly on the compact path.
__device__ __forceinline__ void mbarrier_wait_parity_compact(
    barrier_t* barrier, unsigned phase) {
  const uint32_t address =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  uint32_t done;
  do {
    asm volatile(
        "{\n\t"
        ".reg .pred ready;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 ready, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, ready;\n\t"
        "}"
        : "=r"(done) : "r"(address), "r"(phase) : "memory");
  } while (!done);
}

constexpr int BM = 64;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WARP_M = 64;
constexpr int WARP_N = 32;
constexpr int WARPS_N = BN / WARP_N;
constexpr int COMPUTE_WARPS = (BM / WARP_M) * WARPS_N;
constexpr int A_HALF = BM * BK;
constexpr int A_BYTES = A_HALF * int(sizeof(half));
constexpr int PACKED_BYTES = BK * BN / 2;
constexpr int SCALE_BYTES = BN * int(sizeof(half));
constexpr int STAGE_BYTES = A_BYTES + PACKED_BYTES + SCALE_BYTES;

static_assert(COMPUTE_WARPS == 4);

struct AFragment { uint32_t x[4]; };
struct BFragment { uint32_t x[2]; };
struct AccFragment { float x[4]; };

__device__ __forceinline__ AFragment load_a_scalar(
    const half* tile, int row, int kk, int lane) {
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int col = kk + thread_in_group * 2;
  AFragment a;
  a.x[0] = *reinterpret_cast<const uint32_t*>(tile + (row + group) * BK + col);
  a.x[1] = *reinterpret_cast<const uint32_t*>(tile + (row + group + 8) * BK + col);
  a.x[2] = *reinterpret_cast<const uint32_t*>(tile + (row + group) * BK + col + 8);
  a.x[3] = *reinterpret_cast<const uint32_t*>(tile + (row + group + 8) * BK + col + 8);
  return a;
}

// Four groups of eight lanes supply the row addresses for the four 8x8
// matrices making up a row-major 16x16 A fragment.  The returned register
// order matches mma.m16n8k16's A operand: TL, BL, TR, BR.
__device__ __forceinline__ AFragment load_a_ldmatrix(
    unsigned tile_address, int row, int kk, int lane) {
  const unsigned address = tile_address + unsigned(
      ((row + (lane & 7) + ((lane & 8) ? 8 : 0)) * BK +
       kk + ((lane & 16) ? 8 : 0)) * int(sizeof(half)));
  AFragment a;
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(a.x[0]), "=r"(a.x[1]), "=r"(a.x[2]), "=r"(a.x[3])
      : "r"(address));
  return a;
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

__device__ __forceinline__ uint32_t broadcast_half(half scale) {
  const uint32_t sh = __half_as_ushort(scale);
  return sh | (sh << 16);
}

__device__ __forceinline__ uint32_t pack_two_scales(half low, half high) {
  const uint16_t lo = __half_as_ushort(low);
  const uint16_t hi = __half_as_ushort(high);
  uint32_t packed;
  asm("mov.b32 %0, {%1, %2};" : "=r"(packed) : "h"(lo), "h"(hi));
  return packed;
}

template <bool HIGH>
__device__ __forceinline__ uint32_t broadcast_packed_scale(uint32_t packed) {
  uint32_t scale2;
  if constexpr (HIGH)
    asm("prmt.b32 %0, %1, %1, 0x3232;" : "=r"(scale2) : "r"(packed));
  else
    asm("prmt.b32 %0, %1, %1, 0x1010;" : "=r"(scale2) : "r"(packed));
  return scale2;
}

__device__ __forceinline__ BFragment decode_b_fragment(
    uint16_t q, uint32_t scale2) {
  uint32_t q01 = (uint32_t(q) & 0x000fu) | ((uint32_t(q) & 0x00f0u) << 12);
  uint32_t q23 = ((uint32_t(q) & 0x0f00u) >> 8) | ((uint32_t(q) & 0xf000u) << 4);
  q01 = (q01 ^ 0x00080008u) | 0x64006400u;
  q23 = (q23 ^ 0x00080008u) | 0x64006400u;
  q01 = sub_f16x2(q01, 0x64086408u);
  q23 = sub_f16x2(q23, 0x64086408u);
  return {{mul_f16x2(q01, scale2), mul_f16x2(q23, scale2)}};
}

// Keep nibble extraction out of the LOP3 pipe.  BFE produces zero-extended
// bytes and PRMT places two of them in the low nibble of each half word; only
// the signed-half magic injection remains a LOP3 operation.
__device__ __forceinline__ BFragment decode_b_fragment_bfe(
    uint16_t q, uint32_t scale2) {
  const uint32_t x = q;
  uint32_t q0, q1, q2, q3, q01, q23;
  asm("bfe.u32 %0, %1, 0, 4;" : "=r"(q0) : "r"(x));
  asm("bfe.u32 %0, %1, 4, 4;" : "=r"(q1) : "r"(x));
  asm("bfe.u32 %0, %1, 8, 4;" : "=r"(q2) : "r"(x));
  asm("bfe.u32 %0, %1, 12, 4;" : "=r"(q3) : "r"(x));
  asm("prmt.b32 %0, %1, %2, 0x1410;" : "=r"(q01) : "r"(q0), "r"(q1));
  asm("prmt.b32 %0, %1, %2, 0x1410;" : "=r"(q23) : "r"(q2), "r"(q3));
  q01 = (q01 ^ 0x00080008u) | 0x64006400u;
  q23 = (q23 ^ 0x00080008u) | 0x64006400u;
  q01 = sub_f16x2(q01, 0x64086408u);
  q23 = sub_f16x2(q23, 0x64086408u);
  return {{mul_f16x2(q01, scale2), mul_f16x2(q23, scale2)}};
}

// Keep only one nibble pair live at a time.  This is intentionally separate
// from the V17 decoder so the original SASS/results remain reproducible.
__device__ __forceinline__ BFragment decode_b_fragment_compact(
    uint16_t q, uint32_t scale2) {
  BFragment out;
  uint32_t v = (uint32_t(q) & 0x000fu) | ((uint32_t(q) & 0x00f0u) << 12);
  v = (v ^ 0x00080008u) | 0x64006400u;
  v = sub_f16x2(v, 0x64086408u);
  out.x[0] = mul_f16x2(v, scale2);
  v = ((uint32_t(q) & 0x0f00u) >> 8) | ((uint32_t(q) & 0xf000u) << 4);
  v = (v ^ 0x00080008u) | 0x64006400u;
  v = sub_f16x2(v, 0x64086408u);
  out.x[1] = mul_f16x2(v, scale2);
  return out;
}

__device__ __forceinline__ BFragment decode_b_fragment_bfe_compact(
    uint16_t q, uint32_t scale2) {
  const uint32_t x = q;
  BFragment out;
  uint32_t a, b, v;
  asm("bfe.u32 %0, %1, 0, 4;" : "=r"(a) : "r"(x));
  asm("bfe.u32 %0, %1, 4, 4;" : "=r"(b) : "r"(x));
  asm("prmt.b32 %0, %1, %2, 0x1410;" : "=r"(v) : "r"(a), "r"(b));
  v = (v ^ 0x00080008u) | 0x64006400u;
  v = sub_f16x2(v, 0x64086408u);
  out.x[0] = mul_f16x2(v, scale2);
  asm("bfe.u32 %0, %1, 8, 4;" : "=r"(a) : "r"(x));
  asm("bfe.u32 %0, %1, 12, 4;" : "=r"(b) : "r"(x));
  asm("prmt.b32 %0, %1, %2, 0x1410;" : "=r"(v) : "r"(a), "r"(b));
  v = (v ^ 0x00080008u) | 0x64006400u;
  v = sub_f16x2(v, 0x64086408u);
  out.x[1] = mul_f16x2(v, scale2);
  return out;
}

__device__ __forceinline__ void mma_m16n8k16(
    AccFragment& d, const AFragment& a, const BFragment& b) {
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
  CUtensorMap a_map{}, b_map{}, scale_map{};
};

const TensorMapCache& tensor_maps(const half* a, const uint8_t* b,
                                  const half* scales, int M, int N, int K,
                                  int group) {
  static thread_local TensorMapCache cache;
  if (cache.a == a && cache.b == b && cache.scales == scales &&
      cache.M == M && cache.N == N && cache.K == K && cache.group == group)
    return cache;

  const cuuint64_t a_dims[2] = {static_cast<cuuint64_t>(K),
                                static_cast<cuuint64_t>(M)};
  const cuuint64_t a_strides[1] = {static_cast<cuuint64_t>(K) * sizeof(half)};
  const cuuint32_t a_box[2] = {BK, BM};
  const cuuint32_t unit_strides[2] = {1, 1};
  check_driver(cuTensorMapEncodeTiled(
      &cache.a_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, const_cast<half*>(a),
      a_dims, a_strides, a_box, unit_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "cuTensorMapEncodeTiled(V17 A)");

  // Both fragment layouts occupy the same contiguous 2 KiB per BN128/BK32
  // tile, so the tensor-map shape is shared by the scalar and pair variants.
  const cuuint64_t b_dims[2] = {
      16, static_cast<cuuint64_t>(K / BK) * (N / BN) * 16};
  const cuuint64_t b_strides[1] = {128};
  const cuuint32_t b_box[2] = {16, 16};
  check_driver(cuTensorMapEncodeTiled(
      &cache.b_map, CU_TENSOR_MAP_DATA_TYPE_UINT64, 2,
      const_cast<uint8_t*>(b), b_dims, b_strides, b_box, unit_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled(V17 B)");

  const int scale_rows = (N / BN) * (K / group) * 2;
  const cuuint64_t s_dims[2] = {64, static_cast<cuuint64_t>(scale_rows)};
  const cuuint64_t s_strides[1] = {128};
  const cuuint32_t s_box[2] = {64, 2};
  check_driver(cuTensorMapEncodeTiled(
      &cache.scale_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
      const_cast<half*>(scales), s_dims, s_strides, s_box, unit_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled(V17 scales)");

  cache.a = a;
  cache.b = b;
  cache.scales = scales;
  cache.M = M;
  cache.N = N;
  cache.K = K;
  cache.group = group;
  return cache;
}

template <int GROUP, bool LDMATRIX_A, bool PAIR_REPACK, bool FULL_EMPTY,
          bool BFE_DECODE, bool COMPACT_REGS = false>
__global__ void v17_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map,
    const __grid_constant__ CUtensorMap scale_map,
    half* __restrict__ c, int N, int K) {
  constexpr int STAGES = 2;
  constexpr int DATA_BYTES = STAGES * STAGE_BYTES;
  extern __shared__ __align__(128) unsigned char raw[];
  unsigned char* astage = raw;
  unsigned char* bstage = astage + STAGES * A_BYTES;
  unsigned char* sstage = bstage + STAGES * PACKED_BYTES;
  barrier_t* full = reinterpret_cast<barrier_t*>(raw + DATA_BYTES);

  const int tid = threadIdx.x;
  if (tid == 0) {
    #pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) {
      init(&full[stage], 1);
      if constexpr (FULL_EMPTY)
        init(&full[STAGES + stage], COMPACT_REGS ? 128 : COMPUTE_WARPS);
    }
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wm = warp / WARPS_N;
  const int wn = warp - wm * WARPS_N;
  const int nt = int(blockIdx.x);
  const int ktiles = K / BK;
  const int groups = K / GROUP;
  unsigned full_phase[STAGES] = {};
  unsigned empty_phase[STAGES] = {};

  auto issue_tile = [&](int stage, int kt) {
    if (tid == 0) {
      cde::cp_async_bulk_tensor_2d_global_to_shared(
          astage + stage * A_BYTES, &a_map, kt * BK,
          int(blockIdx.y) * BM, full[stage]);
      cde::cp_async_bulk_tensor_2d_global_to_shared(
          bstage + stage * PACKED_BYTES, &b_map, 0,
          (nt * ktiles + kt) * 16, full[stage]);
      if constexpr (COMPACT_REGS && GROUP == 32) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            sstage + stage * SCALE_BYTES, &scale_map, 0,
            (nt * ktiles + kt) * 2, full[stage]);
      } else if constexpr (COMPACT_REGS && GROUP == 64) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            sstage + stage * SCALE_BYTES, &scale_map, 0,
            (nt * ktiles + kt) & ~1, full[stage]);
      } else if constexpr (COMPACT_REGS && GROUP == 128) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            sstage + stage * SCALE_BYTES, &scale_map, 0,
            ((nt * ktiles + kt) >> 1) & ~1, full[stage]);
      } else {
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            sstage + stage * SCALE_BYTES, &scale_map, 0,
            (nt * groups + (kt * BK) / GROUP) * 2, full[stage]);
      }
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
    const int stage = kt & 1;
    if constexpr (FULL_EMPTY) {
      // One elected lane performs the acquire, then __syncwarp establishes
      // the handoff to the other lanes before any shared-memory reads.
      if (lane == 0) {
        const unsigned phase = COMPACT_REGS ? unsigned((kt >> 1) & 1)
                                            : full_phase[stage];
        if constexpr (COMPACT_REGS) {
          mbarrier_wait_parity_compact(&full[stage], phase);
        } else {
          uint64_t* native = cuda::device::barrier_native_handle(full[stage]);
          while (!cuda::ptx::mbarrier_try_wait_parity(
              native, phase, 1000000)) {}
        }
        if constexpr (!COMPACT_REGS) full_phase[stage] ^= 1;
      }
      __syncwarp();
    } else {
      if (tid == 0) {
        const unsigned phase = COMPACT_REGS ? unsigned((kt >> 1) & 1)
                                            : full_phase[stage];
        if constexpr (COMPACT_REGS) {
          mbarrier_wait_parity_compact(&full[stage], phase);
        } else {
          uint64_t* native = cuda::device::barrier_native_handle(full[stage]);
          while (!cuda::ptx::mbarrier_try_wait_parity(
              native, phase, 1000000)) {}
        }
        if constexpr (!COMPACT_REGS) full_phase[stage] ^= 1;
      }
      __syncthreads();
    }

    const half* atile = reinterpret_cast<const half*>(astage + stage * A_BYTES);
    const unsigned atile_address = LDMATRIX_A
        ? static_cast<unsigned>(__cvta_generic_to_shared(atile)) : 0u;
    const unsigned char* btile = bstage + stage * PACKED_BYTES;
    const half* stile = reinterpret_cast<const half*>(sstage + stage * SCALE_BYTES);

    if constexpr (PAIR_REPACK) {
      uint32_t qpairs[COMPACT_REGS ? (WARP_N / 16) : (WARP_N / 8)];
      uint32_t scales[COMPACT_REGS ? (WARP_N / 16) : (WARP_N / 8)];
      if constexpr (COMPACT_REGS) {
        #pragma unroll
        for (int p = 0; p < WARP_N / 16; ++p) {
          const int segment64 = wn >> 1;
          const int n8_in_segment = (wn & 1) * 4 + p;
          const int word = mma_fragment_pair_word(
              segment64, n8_in_segment, lane);
          qpairs[p] = reinterpret_cast<const uint32_t*>(btile)[word];
          const int n0 = wn * WARP_N + (p * 2) * 8 + (lane >> 2);
          const int n1 = n0 + 8;
          scales[p] = pack_two_scales(stile[n0], stile[n1]);
        }
      } else {
        #pragma unroll
        for (int j = 0; j < WARP_N / 8; ++j) {
          const int segment64 = wn >> 1;
          const int n8_in_segment = (wn & 1) * 4 + j;
          const int word = mma_fragment_pair_word(
              segment64, n8_in_segment, lane);
          qpairs[j] = reinterpret_cast<const uint32_t*>(btile)[word];
          const int n = wn * WARP_N + j * 8 + (lane >> 2);
          scales[j] = broadcast_half(stile[n]);
        }
      }
      #pragma unroll
      for (int k16 = 0; k16 < 2; ++k16) {
        AFragment af[WARP_M / 16];
        #pragma unroll
        for (int i = 0; i < WARP_M / 16; ++i) {
          const int row = wm * WARP_M + i * 16;
          if constexpr (LDMATRIX_A)
            af[i] = load_a_ldmatrix(atile_address, row, k16 * 16, lane);
          else
            af[i] = load_a_scalar(atile, row, k16 * 16, lane);
        }
        if constexpr (COMPACT_REGS) {
          // Cache two uint32 fragment pairs across k16.  The other two n8
          // fragments are loaded as uint16 halves at their point of use,
          // removing two long-lived registers while retaining half of the
          // pair-layout reuse.
          #pragma unroll
          for (int j = 0; j < WARP_N / 16; ++j) {
            const uint16_t q = uint16_t(qpairs[j] >> (k16 * 16));
            const uint32_t scale2 = (j & 1)
                ? broadcast_packed_scale<true>(scales[j >> 1])
                : broadcast_packed_scale<false>(scales[j >> 1]);
            const BFragment bf = BFE_DECODE
                ? decode_b_fragment_bfe_compact(q, scale2)
                : decode_b_fragment_compact(q, scale2);
            #pragma unroll
            for (int i = 0; i < WARP_M / 16; ++i)
              mma_m16n8k16(acc[i][j], af[i], bf);
          }
          #pragma unroll
          for (int j = WARP_N / 16; j < WARP_N / 8; ++j) {
            const int segment64 = wn >> 1;
            const int n8_in_segment = (wn & 1) * 4 + j;
            const int word = mma_fragment_pair_word(
                segment64, n8_in_segment, lane);
            const uint16_t q = reinterpret_cast<const uint16_t*>(btile)
                [word * 2 + k16];
            const uint32_t scale2 = (j & 1)
                ? broadcast_packed_scale<true>(scales[j >> 1])
                : broadcast_packed_scale<false>(scales[j >> 1]);
            const BFragment bf = BFE_DECODE
                ? decode_b_fragment_bfe_compact(q, scale2)
                : decode_b_fragment_compact(q, scale2);
            #pragma unroll
            for (int i = 0; i < WARP_M / 16; ++i)
              mma_m16n8k16(acc[i][j], af[i], bf);
          }
        } else {
          #pragma unroll
          for (int j = 0; j < WARP_N / 8; ++j) {
            const uint16_t q = uint16_t(qpairs[j] >> (k16 * 16));
            const BFragment bf = BFE_DECODE
                                     ? decode_b_fragment_bfe(q, scales[j])
                                     : decode_b_fragment(q, scales[j]);
            #pragma unroll
            for (int i = 0; i < WARP_M / 16; ++i)
              mma_m16n8k16(acc[i][j], af[i], bf);
          }
        }
      }
    } else {
      #pragma unroll
      for (int k16 = 0; k16 < 2; ++k16) {
        AFragment af[WARP_M / 16];
        #pragma unroll
        for (int i = 0; i < WARP_M / 16; ++i) {
          const int row = wm * WARP_M + i * 16;
          if constexpr (LDMATRIX_A)
            af[i] = load_a_ldmatrix(atile_address, row, k16 * 16, lane);
          else
            af[i] = load_a_scalar(atile, row, k16 * 16, lane);
        }
        #pragma unroll
        for (int j = 0; j < WARP_N / 8; ++j) {
          const int segment64 = wn >> 1;
          const int n8_in_segment = (wn & 1) * 4 + j;
          const int word = mma_fragment_word(
              k16, segment64, n8_in_segment, lane);
          const uint16_t q = reinterpret_cast<const uint16_t*>(btile)[word];
          const int n = wn * WARP_N + j * 8 + (lane >> 2);
          const uint32_t scale2 = broadcast_half(stile[n]);
          const BFragment bf = BFE_DECODE
                                   ? decode_b_fragment_bfe(q, scale2)
                                   : decode_b_fragment(q, scale2);
          #pragma unroll
          for (int i = 0; i < WARP_M / 16; ++i)
            mma_m16n8k16(acc[i][j], af[i], bf);
        }
      }
    }

    if constexpr (FULL_EMPTY) {
      if constexpr (COMPACT_REGS) {
        // One arrival per consumer lane makes the empty phase itself the
        // cross-warp completion reduction, without a token result register.
        mbarrier_arrive_discard(&full[STAGES + stage]);
      } else {
        __syncwarp();
        if (lane == 0)
          (void)full[STAGES + stage].arrive();
      }
      if (tid == 0 && kt + STAGES < ktiles) {
        const unsigned phase = COMPACT_REGS ? unsigned((kt >> 1) & 1)
                                            : empty_phase[stage];
        if constexpr (COMPACT_REGS) {
          mbarrier_wait_parity_compact(&full[STAGES + stage], phase);
        } else {
          uint64_t* native = cuda::device::barrier_native_handle(
              full[STAGES + stage]);
          while (!cuda::ptx::mbarrier_try_wait_parity(
              native, phase, 1000000)) {}
        }
        if constexpr (!COMPACT_REGS) empty_phase[stage] ^= 1;
        issue_tile(stage, kt + STAGES);
      }
    } else {
      __syncthreads();
      if (kt + STAGES < ktiles) issue_tile(stage, kt + STAGES);
    }
  }

  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int block_m = int(blockIdx.y) * BM;
  const int block_n = int(blockIdx.x) * BN;
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

template <bool LDMATRIX_A, bool PAIR_REPACK, bool FULL_EMPTY, bool BFE_DECODE,
          bool COMPACT_REGS = false>
void launch_v17(const half* a, const uint8_t* b, const half* scales,
                half* c, int M, int N, int K, int group,
                cudaStream_t stream) {
  const TensorMapCache& maps = tensor_maps(a, b, scales, M, N, K, group);
  constexpr int threads = COMPUTE_WARPS * 32;
  constexpr int barriers = FULL_EMPTY ? 4 : 2;
  constexpr int smem = 2 * STAGE_BYTES + barriers * int(sizeof(barrier_t));
  auto launch = [&](auto group_tag) {
    constexpr int G = decltype(group_tag)::value;
    v17_kernel<G, LDMATRIX_A, PAIR_REPACK, FULL_EMPTY, BFE_DECODE, COMPACT_REGS>
        <<<dim3(N / BN, M / BM), threads, smem, stream>>>(
            maps.a_map, maps.b_map, maps.scale_map, c, N, K);
  };
  if (group == 32) launch(std::integral_constant<int, 32>{});
  else if (group == 64) launch(std::integral_constant<int, 64>{});
  else launch(std::integral_constant<int, 128>{});
  CUDA_CHECK(cudaGetLastError());
}

template <int GROUP, bool LDMATRIX_A, bool PAIR_REPACK, bool FULL_EMPTY,
          bool BFE_DECODE, bool COMPACT_REGS = false>
KernelResources resources() {
  constexpr int threads = COMPUTE_WARPS * 32;
  constexpr int barriers = FULL_EMPTY ? 4 : 2;
  constexpr int smem = 2 * STAGE_BYTES + barriers * int(sizeof(barrier_t));
  auto fn = v17_kernel<GROUP, LDMATRIX_A, PAIR_REPACK, FULL_EMPTY, BFE_DECODE,
                       COMPACT_REGS>;
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr, fn));
  int blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, fn, threads, smem));
  return {attr.numRegs, int(attr.sharedSizeBytes), attr.maxThreadsPerBlock,
          blocks, blocks * threads / 1536.0f};
}

template <bool LDMATRIX_A, bool PAIR_REPACK, bool FULL_EMPTY, bool BFE_DECODE,
          bool COMPACT_REGS = false>
KernelResources pick_resources(int group) {
  if (group == 32)
    return resources<32, LDMATRIX_A, PAIR_REPACK, FULL_EMPTY, BFE_DECODE,
                     COMPACT_REGS>();
  if (group == 64)
    return resources<64, LDMATRIX_A, PAIR_REPACK, FULL_EMPTY, BFE_DECODE,
                     COMPACT_REGS>();
  return resources<128, LDMATRIX_A, PAIR_REPACK, FULL_EMPTY, BFE_DECODE,
                   COMPACT_REGS>();
}

}  // namespace

void launch_v17a_tma_warp64x32(const half* a, const uint8_t* b,
                               const half* scales, half* c,
                               int M, int N, int K, int group,
                               cudaStream_t stream) {
  launch_v17<false, false, false, false>(a, b, scales, c, M, N, K, group, stream);
}

void launch_v17b_tma_ldmatrix(const half* a, const uint8_t* b,
                              const half* scales, half* c,
                              int M, int N, int K, int group,
                              cudaStream_t stream) {
  launch_v17<true, false, false, false>(a, b, scales, c, M, N, K, group, stream);
}

void launch_v17c_tma_fragment_pair(const half* a, const uint8_t* b,
                                   const half* scales, half* c,
                                   int M, int N, int K, int group,
                                   cudaStream_t stream) {
  launch_v17<true, true, false, false>(a, b, scales, c, M, N, K, group, stream);
}

void launch_v17d_tma_full_empty(const half* a, const uint8_t* b,
                                const half* scales, half* c,
                                int M, int N, int K, int group,
                                cudaStream_t stream) {
  launch_v17<true, true, true, false>(a, b, scales, c, M, N, K, group, stream);
}

void launch_v17e_tma_bfe_decode(const half* a, const uint8_t* b,
                                const half* scales, half* c,
                                int M, int N, int K, int group,
                                cudaStream_t stream) {
  launch_v17<true, true, true, true>(a, b, scales, c, M, N, K, group, stream);
}

void launch_v18a_tma_compact_pair(const half* a, const uint8_t* b,
                                  const half* scales, half* c,
                                  int M, int N, int K, int group,
                                  cudaStream_t stream) {
  launch_v17<true, true, false, false, true>(
      a, b, scales, c, M, N, K, group, stream);
}

void launch_v18b_tma_compact_full_empty(const half* a, const uint8_t* b,
                                        const half* scales, half* c,
                                        int M, int N, int K, int group,
                                        cudaStream_t stream) {
  launch_v17<true, true, true, false, true>(
      a, b, scales, c, M, N, K, group, stream);
}

void launch_v18c_tma_compact_bfe(const half* a, const uint8_t* b,
                                 const half* scales, half* c,
                                 int M, int N, int K, int group,
                                 cudaStream_t stream) {
  launch_v17<true, true, true, true, true>(
      a, b, scales, c, M, N, K, group, stream);
}

KernelResources query_v17_resources(const std::string& version, int group) {
  if (version == "v17a") return pick_resources<false, false, false, false>(group);
  if (version == "v17b") return pick_resources<true, false, false, false>(group);
  if (version == "v17c") return pick_resources<true, true, false, false>(group);
  if (version == "v17d") return pick_resources<true, true, true, false>(group);
  if (version == "v17e") return pick_resources<true, true, true, true>(group);
  if (version == "v18a")
    return pick_resources<true, true, false, false, true>(group);
  if (version == "v18b")
    return pick_resources<true, true, true, false, true>(group);
  if (version == "v18c")
    return pick_resources<true, true, true, true, true>(group);
  throw std::runtime_error("unknown V17 kernel: " + version);
}

}  // namespace int4gemm
