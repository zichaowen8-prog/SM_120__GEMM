#include "kernels.cuh"
#include "async_tensorcore_impl.cuh"

namespace int4gemm {

void launch_v16a_warp_epilogue(const half* a, const uint8_t* b, const half* s,
                               half* c, int M, int N, int K, int group,
                               cudaStream_t stream) {
  async_tc::launch_warp_epilogue<2, 8, 24>(a, b, s, c, M, N, K, group, stream);
}

KernelResources query_v16_resources(const std::string& version, int group) {
  if (version == "v16b") return query_v16b_resources(group);
  if (version.rfind("v16c", 0) == 0) return query_v16c_resources(version, group);
  if (version.rfind("v16d", 0) == 0) return query_v16d_resources(version, group);
  if (version != "v16a") throw std::runtime_error("unknown v16 kernel: " + version);
  if (group == 32) return async_tc::warp_epilogue_resources<2, 32, 8, 24>();
  if (group == 64) return async_tc::warp_epilogue_resources<2, 64, 8, 24>();
  return async_tc::warp_epilogue_resources<2, 128, 8, 24>();
}

}  // namespace int4gemm
