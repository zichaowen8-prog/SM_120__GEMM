#pragma once
#include "common.cuh"

namespace int4gemm {

void launch_quantize(const half* b, uint8_t* packed, half* scales,
                     int K, int N, int group_size, cudaStream_t stream = nullptr);
void launch_dequantize(const uint8_t* packed, const half* scales, half* b,
                       int K, int N, int group_size, cudaStream_t stream = nullptr);
void launch_repack_128x32(const uint8_t* packed, const half* scales,
                          uint8_t* repacked, half* repacked_scales,
                          int K, int N, int group_size, cudaStream_t stream = nullptr);
void launch_inverse_repack_128x32(const uint8_t* repacked, uint8_t* packed,
                                  int K, int N, cudaStream_t stream = nullptr);
void launch_repack_mma_128x32(const uint8_t* packed, uint8_t* repacked,
                              int K, int N, cudaStream_t stream = nullptr);
void launch_repack_mma_pair_128x32(const uint8_t* packed, uint8_t* repacked,
                                   int K, int N, cudaStream_t stream = nullptr);
void launch_repack_scales_128(const half* scales, half* repacked_scales,
                              int K, int N, int group_size,
                              cudaStream_t stream = nullptr);

void quantize_cpu(const std::vector<half>& b, std::vector<uint8_t>& packed,
                  std::vector<half>& scales, int K, int N, int group_size);
void dequantize_cpu(const std::vector<uint8_t>& packed, const std::vector<half>& scales,
                    std::vector<half>& b, int K, int N, int group_size);
ErrorMetrics compare_half(const std::vector<half>& got, const std::vector<half>& ref);
void cpu_gemm_fp32(const std::vector<half>& a, const std::vector<half>& b,
                   std::vector<float>& c, int M, int N, int K);

}  // namespace int4gemm
