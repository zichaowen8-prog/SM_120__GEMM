#include "kernels.cuh"
#include "tensorcore_impl.cuh"

namespace int4gemm {
void launch_v03(const half* a, const uint8_t* b, const half* s, half* c,
                int M, int N, int K, int group, cudaStream_t stream) {
  tc::launch<64, 128, 32, 32, false, false>(a, b, s, c, M, N, K, group, stream);
}
KernelResources query_v03_resources(int group) {
  if (group == 32) return tc::resources<64, 128, 32, 32, 32, false, false>();
  if (group == 64) return tc::resources<64, 128, 32, 32, 64, false, false>();
  return tc::resources<64, 128, 32, 32, 128, false, false>();
}
}  // namespace int4gemm
