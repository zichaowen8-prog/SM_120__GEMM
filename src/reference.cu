#include "quant.cuh"
#include <limits>

namespace int4gemm {

void quantize_cpu(const std::vector<half>& b, std::vector<uint8_t>& packed,
                  std::vector<half>& scales, int K, int N, int group_size) {
  const int groups = (K + group_size - 1) / group_size;
  scales.resize(static_cast<size_t>(groups) * N);
  packed.assign(static_cast<size_t>(K) * N / 2, 0);
  for (int g = 0; g < groups; ++g) {
    for (int n = 0; n < N; ++n) {
      float vmax = 0;
      for (int k = g * group_size; k < std::min(K, (g + 1) * group_size); ++k)
        vmax = std::max(vmax, std::fabs(__half2float(b[static_cast<size_t>(k) * N + n])));
      scales[static_cast<size_t>(g) * N + n] = __float2half(std::max(vmax / 7.0f, 1.0e-8f));
    }
  }
  for (size_t e = 0; e < static_cast<size_t>(K) * N; ++e) {
    int k = int(e / N), n = int(e % N);
    float s = __half2float(scales[static_cast<size_t>(k / group_size) * N + n]);
    int q = std::max(-8, std::min(7, int(std::nearbyint(__half2float(b[e]) / s))));
    if (e & 1) packed[e >> 1] |= uint8_t((q & 15) << 4);
    else packed[e >> 1] = uint8_t(q & 15);
  }
}

void dequantize_cpu(const std::vector<uint8_t>& packed, const std::vector<half>& scales,
                    std::vector<half>& b, int K, int N, int group_size) {
  b.resize(static_cast<size_t>(K) * N);
  for (size_t e = 0; e < b.size(); ++e) {
    const int k = int(e / N), n = int(e % N);
    const uint8_t x = packed[e >> 1];
    const int q = (e & 1) ? sign_extend_int4(x >> 4) : sign_extend_int4(x);
    b[e] = __float2half(float(q) * __half2float(scales[static_cast<size_t>(k / group_size) * N + n]));
  }
}

ErrorMetrics compare_half(const std::vector<half>& got, const std::vector<half>& ref) {
  if (got.size() != ref.size()) throw std::runtime_error("compare size mismatch");
  long double abs_sum = 0, sq_sum = 0, ref_sq = 0;
  ErrorMetrics m;
  for (size_t i = 0; i < got.size(); ++i) {
    double g = __half2float(got[i]), r = __half2float(ref[i]);
    double d = std::fabs(g - r);
    m.max_abs = std::max(m.max_abs, d);
    abs_sum += d; sq_sum += d * d; ref_sq += r * r;
  }
  m.mean_abs = double(abs_sum / got.size());
  m.rmse = std::sqrt(double(sq_sum / got.size()));
  m.relative_l2 = std::sqrt(double(sq_sum / std::max(ref_sq, (long double)1.0e-30)));
  return m;
}

void cpu_gemm_fp32(const std::vector<half>& a, const std::vector<half>& b,
                   std::vector<float>& c, int M, int N, int K) {
  c.assign(static_cast<size_t>(M) * N, 0.0f);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k) {
      float av = __half2float(a[static_cast<size_t>(m) * K + k]);
      for (int n = 0; n < N; ++n)
        c[static_cast<size_t>(m) * N + n] += av * __half2float(b[static_cast<size_t>(k) * N + n]);
    }
}

}  // namespace int4gemm

