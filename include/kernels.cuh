#pragma once
#include "common.cuh"

namespace int4gemm {

struct KernelMetadata {
  const char* version;
  const char* description;
  int BM, BN, BK;
  int warp_m, warp_n;
  int stages;
  int threads;
  int smem_bytes;
};

struct KernelResources {
  int registers_per_thread = -1;
  int static_smem_bytes = -1;
  int max_threads_per_block = -1;
  int active_blocks_per_sm = -1;
  float occupancy = -1.0f;
};

using KernelLaunch = void (*)(const half*, const uint8_t*, const half*, half*,
                              int, int, int, int, cudaStream_t);

void launch_v00(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v01(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v02(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v03(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v04_64x128(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v04_128x64(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v04_128x128(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v05_bk64(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v05_bk128(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v06_warp64(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v07_b_reuse(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v08_128x64_bk32(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v08_128x64_bk64(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v09_stage2(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v09_stage3(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v09_stage4(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v10_apad(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v10_bpad(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v10_abpad(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v11_a8b16(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v11_a16b8(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v11_a16b16(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v11_a8b24(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v12_a8b32(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v12_a8b40(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v12_a16b24(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v13_persistent(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v14_warp64x32(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v14_warp64x64(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v15_tma_b(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v15_tma_ab(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v15_tma_ab_stage3(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v15_tma_ab_stage4(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v15_tma_ab_leader_wait(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v16a_warp_epilogue(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v16b_register_mma(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v16c0_tma(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v16c1_tma_swizzle(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v16c2_tma_swizzle_stage3(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v16c3_tma_dma_warp(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v16c4_tma_stage3(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v16c5_tma_stage4(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v16d0_warp32x32(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v16d1_warp16x64(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v16d2_warp64x32(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v17a_tma_warp64x32(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v17b_tma_ldmatrix(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v17c_tma_fragment_pair(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v17d_tma_full_empty(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v17e_tma_bfe_decode(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v18a_tma_compact_pair(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v18b_tma_compact_full_empty(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
void launch_v18c_tma_compact_bfe(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);
#if !defined(INT4_GEMM_ENABLE_FP32_OUTPUT) || INT4_GEMM_ENABLE_FP32_OUTPUT
void launch_final_fp32(const half*, const uint8_t*, const half*, float*, int, int, int, int, cudaStream_t);
#endif
void launch_final(const half*, const uint8_t*, const half*, half*, int, int, int, int, cudaStream_t);

std::vector<std::pair<KernelMetadata, KernelLaunch>> kernel_registry();
KernelResources query_v00_resources();
KernelResources query_v01_resources();
KernelResources query_v02_resources();
KernelResources query_v03_resources(int group);
KernelResources query_v04_resources(const std::string& version, int group);
KernelResources query_final_resources(int group);
KernelResources query_tuned_resources(const std::string& version, int group);
KernelResources query_v09_resources(const std::string& version, int group);
KernelResources query_v10_resources(const std::string& version, int group);
KernelResources query_v11_resources(const std::string& version, int group);
KernelResources query_v12_resources(const std::string& version, int group);
KernelResources query_v13_resources(int group);
KernelResources query_v14_resources(const std::string& version, int group);
KernelResources query_v15_resources(const std::string& version, int group);
KernelResources query_v16_resources(const std::string& version, int group);
KernelResources query_v16b_resources(int group);
KernelResources query_v16c_resources(const std::string& version, int group);
KernelResources query_v16d_resources(const std::string& version, int group);
KernelResources query_v17_resources(const std::string& version, int group);
KernelResources query_resources(const std::string& version, int group);

}  // namespace int4gemm
