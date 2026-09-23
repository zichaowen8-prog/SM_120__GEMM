#pragma once
#include "common.cuh"
#include <numeric>

namespace int4gemm {

struct TimingStats {
  float median_ms = 0, mean_ms = 0, min_ms = 0, p95_ms = 0;
};

template <class F>
TimingStats benchmark_cuda(F&& fn, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  std::vector<float> times(iterations);
  for (int i = 0; i < iterations; ++i) {
    CUDA_CHECK(cudaEventRecord(begin));
    fn();
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&times[i], begin, end));
  }
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  std::sort(times.begin(), times.end());
  TimingStats s;
  s.min_ms = times.front();
  s.median_ms = times[times.size() / 2];
  s.p95_ms = times[std::min(times.size() - 1, size_t(std::ceil(times.size() * .95) - 1))];
  s.mean_ms = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
  return s;
}

inline double tflops(int M, int N, int K, double ms) {
  return 2.0 * M * N * K / (ms * 1.0e9);
}

}  // namespace int4gemm

