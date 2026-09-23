#include "kernels.cuh"
#include "tensorcore_impl.cuh"

namespace int4gemm {
void launch_final(const half* a, const uint8_t* b, const half* s, half* c,
                  int M, int N, int K, int group, cudaStream_t stream) {
  if (group == 32)
    launch_v18a_tma_compact_pair(a, b, s, c, M, N, K, group, stream);
  else if (group == 64)
    launch_v17b_tma_ldmatrix(a, b, s, c, M, N, K, group, stream);
  else
    launch_v18b_tma_compact_full_empty(a, b, s, c, M, N, K, group, stream);
}

void launch_v05_bk64(const half* a, const uint8_t* b, const half* s, half* c,
                     int M, int N, int K, int group, cudaStream_t stream) {
  tc::launch<64, 128, 64, 32, true, true>(a,b,s,c,M,N,K,group,stream);
}
void launch_v05_bk128(const half* a, const uint8_t* b, const half* s, half* c,
                      int M, int N, int K, int group, cudaStream_t stream) {
  tc::launch<64, 128, 128, 32, true, true>(a,b,s,c,M,N,K,group,stream);
}
void launch_v06_warp64(const half* a, const uint8_t* b, const half* s, half* c,
                       int M, int N, int K, int group, cudaStream_t stream) {
  tc::launch<64, 128, 64, 64, true, true>(a,b,s,c,M,N,K,group,stream);
}
void launch_v07_b_reuse(const half* a, const uint8_t* b, const half* s, half* c,
                        int M, int N, int K, int group, cudaStream_t stream) {
  tc::launch<64, 128, 64, 64, true, true, true>(a,b,s,c,M,N,K,group,stream);
}
void launch_v08_128x64_bk32(const half* a, const uint8_t* b, const half* s, half* c,
                            int M, int N, int K, int group, cudaStream_t stream) {
  tc::launch<128, 64, 32, 64, true, false>(a,b,s,c,M,N,K,group,stream);
}
void launch_v08_128x64_bk64(const half* a, const uint8_t* b, const half* s, half* c,
                            int M, int N, int K, int group, cudaStream_t stream) {
  tc::launch<128, 64, 64, 64, true, false>(a,b,s,c,M,N,K,group,stream);
}

KernelResources query_final_resources(int group) {
  if (group == 32) return query_v17_resources("v18a", group);
  if (group == 64) return query_v17_resources("v17b", group);
  return query_v17_resources("v18b", group);
}

KernelResources query_tuned_resources(const std::string& version, int group) {
#define RES(BK, WN, G, MAP) tc::resources<64,128,BK,WN,G,true,true,MAP>()
  if (version == "v05a") {
    if(group==32)return RES(64,32,32,false); if(group==64)return RES(64,32,64,false); return RES(64,32,128,false);
  }
  if (version == "v05b") {
    if(group==32)return RES(128,32,32,false); if(group==64)return RES(128,32,64,false); return RES(128,32,128,false);
  }
  if (version == "v06") {
    if(group==32)return RES(64,64,32,false); if(group==64)return RES(64,64,64,false); return RES(64,64,128,false);
  }
  if (version == "v08a") {
    if(group==32)return tc::resources<128,64,32,64,32,true,false>(); if(group==64)return tc::resources<128,64,32,64,64,true,false>(); return tc::resources<128,64,32,64,128,true,false>();
  }
  if (version == "v08b") {
    if(group==32)return tc::resources<128,64,64,64,32,true,false>(); if(group==64)return tc::resources<128,64,64,64,64,true,false>(); return tc::resources<128,64,64,64,128,true,false>();
  }
  if(group==32)return RES(64,64,32,true); if(group==64)return RES(64,64,64,true); return RES(64,64,128,true);
#undef RES
}

std::vector<std::pair<KernelMetadata, KernelLaunch>> kernel_registry() {
  return {
    {{"v00", "scalar fused row-packed", 16, 16, 1, 1, 1, 1, 256, 0}, launch_v00},
    {{"v01", "pairwise INT4 load + half2 store", 1, 2, 1, 1, 2, 1, 256, 0}, launch_v01},
    {{"v02", "SIMT CTA-tile-repacked", 1, 2, 32, 1, 2, 1, 256, 0}, launch_v02},
    {{"v03", "WMMA Tensor Core, row-packed B", 64, 128, 32, 32, 32, 1, 256, 32768}, launch_v03},
    {{"v04a", "WMMA tile-repacked 64x128", 64, 128, 32, 32, 32, 1, 256, 32768}, launch_v04_64x128},
    {{"v04b", "WMMA tile-repacked 128x64", 128, 64, 32, 32, 32, 1, 256, 32768}, launch_v04_128x64},
    {{"v04c", "WMMA tile-repacked 128x128", 128, 128, 32, 32, 32, 1, 512, 65536}, launch_v04_128x128},
    {{"v05a", "vector B + BK64 barrier reduction", 64, 128, 64, 32, 32, 1, 256, 32768}, launch_v05_bk64},
    {{"v05b", "vector B + BK128", 64, 128, 128, 32, 32, 1, 256, 49152}, launch_v05_bk128},
    {{"v06", "BK64 + 32x64 warp tile", 64, 128, 64, 32, 64, 1, 128, 32768}, launch_v06_warp64},
    {{"v07", "BK64 + 32x64 warp + B-reuse grid", 64, 128, 64, 32, 64, 1, 128, 32768}, launch_v07_b_reuse},
    {{"v08a", "128x64 CTA + BK32 + 32x64 warp (scalar tile load)", 128, 64, 32, 32, 64, 1, 128, 32768}, launch_v08_128x64_bk32},
    {{"v08b", "128x64 CTA + BK64 + 32x64 warp (scalar tile load)", 128, 64, 64, 32, 64, 1, 128, 32768}, launch_v08_128x64_bk64},
    {{"v09s2", "cp.async 2-stage packed-B pipeline", 64, 128, 32, 32, 64, 2, 128, 32768}, launch_v09_stage2},
    {{"v09s3", "cp.async 3-stage packed-B pipeline", 64, 128, 32, 32, 64, 3, 128, 32768}, launch_v09_stage3},
    {{"v09s4", "cp.async 4-stage packed-B pipeline", 64, 128, 32, 32, 64, 4, 128, 32768}, launch_v09_stage4},
    {{"v10a", "stage2 + A shared stride padding", 64, 128, 32, 32, 64, 2, 128, 32768}, launch_v10_apad},
    {{"v10b", "stage2 + B shared stride padding", 64, 128, 32, 32, 64, 2, 128, 32768}, launch_v10_bpad},
    {{"v10c", "stage2 + A/B shared stride padding", 64, 128, 32, 32, 64, 2, 128, 32768}, launch_v10_abpad},
    {{"v11a", "padding sweep A+8 B+16", 64, 128, 32, 32, 64, 2, 128, 32768}, launch_v11_a8b16},
    {{"v11b", "padding sweep A+16 B+8", 64, 128, 32, 32, 64, 2, 128, 32768}, launch_v11_a16b8},
    {{"v11c", "padding sweep A+16 B+16", 64, 128, 32, 32, 64, 2, 128, 32768}, launch_v11_a16b16},
    {{"v11d", "padding sweep A+8 B+24", 64, 128, 32, 32, 64, 2, 128, 32768}, launch_v11_a8b24},
    {{"v12a", "padding sweep A+8 B+32", 64, 128, 32, 32, 64, 2, 128, 32768}, launch_v12_a8b32},
    {{"v12b", "padding sweep A+8 B+40", 64, 128, 32, 32, 64, 2, 128, 32768}, launch_v12_a8b40},
    {{"v12c", "padding sweep A+16 B+24", 64, 128, 32, 32, 64, 2, 128, 32768}, launch_v12_a16b24},
    {{"v13p", "persistent 128-CTA grid-stride scheduling", 64, 128, 32, 32, 64, 2, 128, 32768}, launch_v13_persistent},
    {{"v14a", "final pipeline with 64x32 warp tile", 64, 128, 32, 64, 32, 2, 128, 32768}, launch_v14_warp64x32},
    {{"v14b", "final pipeline with 64x64 warp tile", 64, 128, 32, 64, 64, 2, 64, 32768}, launch_v14_warp64x64},
    {{"v15a", "TMA packed-B + cp.async padded-A", 64, 128, 32, 32, 64, 2, 128, 32784}, launch_v15_tma_b},
    {{"v15b", "TMA A/B with compact A shared tile", 64, 128, 32, 32, 64, 2, 128, 32784}, launch_v15_tma_ab},
    {{"v15c", "three-stage TMA A/B pipeline", 64, 128, 32, 32, 64, 3, 128, 32792}, launch_v15_tma_ab_stage3},
    {{"v15d", "four-stage TMA A/B pipeline, compact B", 64, 128, 32, 32, 64, 4, 128, 32800}, launch_v15_tma_ab_stage4},
    {{"v15e", "three-stage TMA A/B, leader mbarrier wait", 64, 128, 32, 32, 64, 3, 128, 32792}, launch_v15_tma_ab_leader_wait},
    {{"v16a", "warp-streamed 16x16 FP32 scratch epilogue", 64, 128, 32, 32, 64, 2, 128, 24064}, launch_v16a_warp_epilogue},
    {{"v16b", "lane-native INT4 decode + explicit register MMA", 64, 128, 32, 32, 64, 2, 128, 14848}, launch_v16b_register_mma},
    {{"v16c0", "register MMA + 2-stage TMA, no swizzle", 64, 128, 32, 32, 64, 2, 128, 12816}, launch_v16c0_tma},
    {{"v16c1", "register MMA + 2-stage TMA + shared swizzle", 64, 128, 32, 32, 64, 2, 128, 12816}, launch_v16c1_tma_swizzle},
    {{"v16c2", "register MMA + 3-stage TMA + shared swizzle", 64, 128, 32, 32, 64, 3, 128, 19224}, launch_v16c2_tma_swizzle_stage3},
    {{"v16c3", "register MMA + 3-stage swizzled TMA + DMA warp", 64, 128, 32, 32, 64, 3, 160, 19224}, launch_v16c3_tma_dma_warp},
    {{"v16c4", "register MMA + 3-stage TMA, no swizzle", 64, 128, 32, 32, 64, 3, 128, 19224}, launch_v16c4_tma_stage3},
    {{"v16c5", "register MMA + 4-stage TMA, no swizzle", 64, 128, 32, 32, 64, 4, 128, 25632}, launch_v16c5_tma_stage4},
    {{"v16d1", "register MMA warp tile 16x64", 64, 128, 32, 16, 64, 2, 256, 14848}, launch_v16d1_warp16x64},
    {{"v16d2", "register MMA warp tile 64x32", 64, 128, 32, 64, 32, 2, 128, 14848}, launch_v16d2_warp64x32},
    {{"v17a", "TMA baseline with 64x32 warp tile", 64, 128, 32, 64, 32, 2, 128, 12816}, launch_v17a_tma_warp64x32},
    {{"v17b", "64x32 TMA + ldmatrix.x4 A fragments", 64, 128, 32, 64, 32, 2, 128, 12816}, launch_v17b_tma_ldmatrix},
    {{"v17c", "ldmatrix + group32 k16 fragment-pair repack", 64, 128, 32, 64, 32, 2, 128, 12816}, launch_v17c_tma_fragment_pair},
    {{"v17d", "fragment-pair TMA + full/empty mbarrier handoff", 64, 128, 32, 64, 32, 2, 128, 12832}, launch_v17d_tma_full_empty},
    {{"v17e", "full/empty TMA + BFE/PRMT INT4 decode", 64, 128, 32, 64, 32, 2, 128, 12832}, launch_v17e_tma_bfe_decode},
    {{"v18a", "fragment-pair TMA + two packed scale registers", 64, 128, 32, 64, 32, 2, 128, 12816}, launch_v18a_tma_compact_pair},
    {{"v18b", "compact pair/scale + computed-phase full/empty TMA", 64, 128, 32, 64, 32, 2, 128, 12832}, launch_v18b_tma_compact_full_empty},
    {{"v18c", "compact full/empty TMA + serialized BFE/PRMT decode", 64, 128, 32, 64, 32, 2, 128, 12832}, launch_v18c_tma_compact_bfe},
    {{"final", "group-aware V18/V17: compact pair g32, ldmatrix g64, full-empty g128", 64, 128, 32, 64, 32, 2, 128, 12816}, launch_final},
  };
}

KernelResources query_resources(const std::string& version, int group) {
  if (version == "v00") return query_v00_resources();
  if (version == "v01") return query_v01_resources();
  if (version == "v02") return query_v02_resources();
  if (version == "v03") return query_v03_resources(group);
  if (version.rfind("v04", 0) == 0) return query_v04_resources(version, group);
  if (version.rfind("v05", 0) == 0 || version == "v06" || version == "v07" || version.rfind("v08",0)==0) return query_tuned_resources(version, group);
  if (version.rfind("v09",0)==0) return query_v09_resources(version,group);
  if (version.rfind("v10",0)==0) return query_v10_resources(version,group);
  if (version.rfind("v11",0)==0) return query_v11_resources(version,group);
  if (version.rfind("v12",0)==0) return query_v12_resources(version,group);
  if (version == "v13p") return query_v13_resources(group);
  if (version.rfind("v14",0)==0) return query_v14_resources(version,group);
  if (version.rfind("v15",0)==0) return query_v15_resources(version,group);
  if (version.rfind("v16",0)==0) return query_v16_resources(version,group);
  if (version.rfind("v17",0)==0) return query_v17_resources(version,group);
  if (version.rfind("v18",0)==0) return query_v17_resources(version,group);
  return query_final_resources(group);
}
}  // namespace int4gemm
