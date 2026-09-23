#include "kernels.cuh"
#include "layout.cuh"

namespace int4gemm {
namespace {

constexpr int BM = 64;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int A_STRIDE = 40;
constexpr int A_HALF = BM * A_STRIDE;
constexpr int PACKED_BYTES = BK * BN / 2;

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(addr), "l"(gmem));
}

__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ __forceinline__ void cp_wait() {
  if constexpr (N == 1) asm volatile("cp.async.wait_group 1;");
  else if constexpr (N == 2) asm volatile("cp.async.wait_group 2;");
  else asm volatile("cp.async.wait_group 3;");
}

struct AFragment { uint32_t x[4]; };
struct BFragment { uint32_t x[2]; };
struct AccFragment { float x[4]; };

__device__ __forceinline__ AFragment load_a_fragment(const half* tile, int row,
                                                      int kk, int lane) {
  const int group = lane >> 2;
  const int thread_in_group = lane & 3;
  const int col = kk + thread_in_group * 2;
  AFragment a;
  a.x[0] = *reinterpret_cast<const uint32_t*>(tile + (row + group) * A_STRIDE + col);
  a.x[1] = *reinterpret_cast<const uint32_t*>(tile + (row + group + 8) * A_STRIDE + col);
  a.x[2] = *reinterpret_cast<const uint32_t*>(tile + (row + group) * A_STRIDE + col + 8);
  a.x[3] = *reinterpret_cast<const uint32_t*>(tile + (row + group + 8) * A_STRIDE + col + 8);
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

// Convert four two's-complement INT4 values into the two f16x2 registers
// required by mma.m16n8k16.  XOR 8 maps signed-nibble encoding to a biased
// unsigned value; OR-ing it into 1024.0h and subtracting 1032.0h performs the
// conversion without scalar integer-to-float instructions.
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

template <int GROUP, int WARP_M, int WARP_N, int STAGES>
__global__ void register_mma_kernel(const half* __restrict__ a,
                                    const uint8_t* __restrict__ b,
                                    const half* __restrict__ scales,
                                    half* __restrict__ c,
                                    int M, int N, int K) {
  constexpr int WARPS_N = BN / WARP_N;
  constexpr int THREADS = (BM / WARP_M) * WARPS_N * 32;
  static_assert(BM % WARP_M == 0 && BN % WARP_N == 0);
  static_assert(WARP_M % 16 == 0 && WARP_N % 8 == 0);
  extern __shared__ __align__(16) unsigned char raw[];
  half* astage = reinterpret_cast<half*>(raw);
  uint8_t* bstage = raw + STAGES * A_HALF * sizeof(half);
  half* sstage = reinterpret_cast<half*>(bstage + STAGES * PACKED_BYTES);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wm = warp / WARPS_N;
  const int wn = warp - wm * WARPS_N;
  const int block_m = int(blockIdx.y) * BM;
  const int nt = int(blockIdx.x);
  const int block_n = nt * BN;
  const int ktiles = K / BK;
  const int groups = K / GROUP;

  auto issue_tile = [&](int stage, int kt) {
    half* adst = astage + stage * A_HALF;
    for (int chunk = tid; chunk < BM * (BK / 8); chunk += THREADS) {
      const int mi = chunk / (BK / 8);
      const int ki = (chunk % (BK / 8)) * 8;
      cp_async_16(adst + mi * A_STRIDE + ki,
                  a + static_cast<size_t>(block_m + mi) * K + kt * BK + ki);
    }
    uint8_t* bdst = bstage + stage * PACKED_BYTES;
    const uint8_t* bsrc = b + (static_cast<size_t>(nt) * ktiles + kt) * PACKED_BYTES;
    for (int chunk = tid; chunk < PACKED_BYTES / 16; chunk += THREADS)
      cp_async_16(bdst + chunk * 16, bsrc + chunk * 16);
    if (tid < BN * int(sizeof(half)) / 16) {
      half* sdst = sstage + stage * BN;
      const half* ssrc = scales +
          (static_cast<size_t>(nt) * groups + (kt * BK) / GROUP) * BN;
      cp_async_16(sdst + tid * 8, ssrc + tid * 8);
    }
    cp_commit();
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
    cp_wait<STAGES - 1>();
    __syncthreads();
    const int stage = STAGES == 2 ? (kt & 1) : (kt % STAGES);
    const half* atile = astage + stage * A_HALF;
    const uint16_t* btile = reinterpret_cast<const uint16_t*>(
        bstage + stage * PACKED_BYTES);
    const half* stile = sstage + stage * BN;
    #pragma unroll
    for (int k16 = 0; k16 < 2; ++k16) {
      AFragment af[WARP_M / 16];
      #pragma unroll
      for (int i = 0; i < WARP_M / 16; ++i)
        af[i] = load_a_fragment(atile, wm * WARP_M + i * 16,
                                k16 * 16, lane);
      #pragma unroll
      for (int j = 0; j < WARP_N / 8; ++j) {
        const int n_base = wn * WARP_N + j * 8;
        const int n = n_base + (lane >> 2);
        uint16_t q;
        if constexpr (WARP_N == 64) {
          q = btile[mma_fragment_word(k16, wn, j, lane)];
        } else {
          const int segment64 = wn >> 1;
          const int n8_in_segment = (wn & 1) * 4 + j;
          q = btile[mma_fragment_word(k16, segment64, n8_in_segment, lane)];
        }
        const BFragment bf = decode_b_fragment(q, stile[n]);
        #pragma unroll
        for (int i = 0; i < WARP_M / 16; ++i)
          mma_m16n8k16(acc[i][j], af[i], bf);
      }
    }
    __syncthreads();
    if (kt + STAGES < ktiles) issue_tile(stage, kt + STAGES);
  }
  asm volatile("cp.async.wait_all;");

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

template <int GROUP, int WARP_M, int WARP_N, int STAGES>
KernelResources resources() {
  constexpr int threads = (BM / WARP_M) * (BN / WARP_N) * 32;
  constexpr int smem = STAGES *
      (A_HALF * int(sizeof(half)) + PACKED_BYTES + BN * int(sizeof(half)));
  auto fn = register_mma_kernel<GROUP, WARP_M, WARP_N, STAGES>;
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr, fn));
  int blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, fn, threads, smem));
  return {attr.numRegs, int(attr.sharedSizeBytes), attr.maxThreadsPerBlock,
          blocks, blocks * threads / 1536.0f};
}

template <int WARP_M, int WARP_N, int STAGES>
void launch_register(const half* a, const uint8_t* b, const half* scales,
                     half* c, int M, int N, int K, int group,
                     cudaStream_t stream) {
  constexpr int threads = (BM / WARP_M) * (BN / WARP_N) * 32;
  constexpr int smem = STAGES *
      (A_HALF * int(sizeof(half)) + PACKED_BYTES + BN * int(sizeof(half)));
  auto launch = [&](auto group_tag) {
    constexpr int G = decltype(group_tag)::value;
    register_mma_kernel<G, WARP_M, WARP_N, STAGES>
        <<<dim3(N / BN, M / BM), threads, smem, stream>>>(
            a, b, scales, c, M, N, K);
  };
  if (group == 32) launch(std::integral_constant<int, 32>{});
  else if (group == 64) launch(std::integral_constant<int, 64>{});
  else launch(std::integral_constant<int, 128>{});
  CUDA_CHECK(cudaGetLastError());
}

template <int WARP_M, int WARP_N, int STAGES>
KernelResources pick_resources(int group) {
  if (group == 32) return resources<32, WARP_M, WARP_N, STAGES>();
  if (group == 64) return resources<64, WARP_M, WARP_N, STAGES>();
  return resources<128, WARP_M, WARP_N, STAGES>();
}

}  // namespace

void launch_v16b_register_mma(const half* a, const uint8_t* b, const half* scales,
                              half* c, int M, int N, int K, int group,
                              cudaStream_t stream) {
  launch_register<32, 64, 2>(a, b, scales, c, M, N, K, group, stream);
}

KernelResources query_v16b_resources(int group) {
  return pick_resources<32, 64, 2>(group);
}

void launch_v16d0_warp32x32(const half* a, const uint8_t* b, const half* scales,
                            half* c, int M, int N, int K, int group,
                            cudaStream_t stream) {
  launch_register<32, 32, 2>(a, b, scales, c, M, N, K, group, stream);
}

void launch_v16d1_warp16x64(const half* a, const uint8_t* b, const half* scales,
                            half* c, int M, int N, int K, int group,
                            cudaStream_t stream) {
  launch_register<16, 64, 2>(a, b, scales, c, M, N, K, group, stream);
}

void launch_v16d2_warp64x32(const half* a, const uint8_t* b, const half* scales,
                            half* c, int M, int N, int K, int group,
                            cudaStream_t stream) {
  launch_register<64, 32, 2>(a, b, scales, c, M, N, K, group, stream);
}

KernelResources query_v16d_resources(const std::string& version, int group) {
  if (version == "v16d0") return pick_resources<32, 32, 2>(group);
  if (version == "v16d1") return pick_resources<16, 64, 2>(group);
  if (version == "v16d2") return pick_resources<64, 32, 2>(group);
  throw std::runtime_error("unknown V16d kernel: " + version);
}

}  // namespace int4gemm
