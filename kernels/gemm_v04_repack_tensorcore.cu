#include "kernels.cuh"
#include "tensorcore_impl.cuh"

namespace int4gemm {
void launch_v04_64x128(const half* a, const uint8_t* b, const half* s, half* c,
                       int M, int N, int K, int group, cudaStream_t stream) {
  tc::launch<64, 128, 32, 32, true, false>(a, b, s, c, M, N, K, group, stream);
}
void launch_v04_128x64(const half* a, const uint8_t* b, const half* s, half* c,
                       int M, int N, int K, int group, cudaStream_t stream) {
  tc::launch<128, 64, 32, 32, true, false>(a, b, s, c, M, N, K, group, stream);
}
void launch_v04_128x128(const half* a, const uint8_t* b, const half* s, half* c,
                        int M, int N, int K, int group, cudaStream_t stream) {
  tc::launch<128, 128, 32, 32, true, false>(a, b, s, c, M, N, K, group, stream);
}
KernelResources query_v04_resources(const std::string& version, int group) {
#define PICK(BM, BN, G) tc::resources<BM, BN, 32, 32, G, true, false>()
  if (version == "v04a") {
    if (group == 32) return PICK(64, 128, 32); if (group == 64) return PICK(64, 128, 64); return PICK(64, 128, 128);
  }
  if (version == "v04b") {
    if (group == 32) return PICK(128, 64, 32); if (group == 64) return PICK(128, 64, 64); return PICK(128, 64, 128);
  }
  if (group == 32) return PICK(128, 128, 32); if (group == 64) return PICK(128, 128, 64); return PICK(128, 128, 128);
#undef PICK
}
}  // namespace int4gemm
