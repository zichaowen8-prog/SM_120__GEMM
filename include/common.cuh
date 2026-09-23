#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#define CUDA_CHECK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) { \
  char _buf[512]; std::snprintf(_buf, sizeof(_buf), "CUDA error %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(_e)); \
  throw std::runtime_error(_buf); } } while (0)

#define CUBLAS_CHECK(expr) do { cublasStatus_t _s = (expr); if (_s != CUBLAS_STATUS_SUCCESS) { \
  char _buf[512]; std::snprintf(_buf, sizeof(_buf), "cuBLAS error %s:%d: status=%d", __FILE__, __LINE__, int(_s)); \
  throw std::runtime_error(_buf); } } while (0)

namespace int4gemm {

constexpr int kTargetM = 4096;
constexpr int kTargetN = 4096;
constexpr int kTargetK = 4096;

template <class T> class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count_) CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
  }
  ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), count_(o.count_) { o.ptr_ = nullptr; o.count_ = 0; }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) { if (ptr_) cudaFree(ptr_); ptr_ = o.ptr_; count_ = o.count_; o.ptr_ = nullptr; o.count_ = 0; }
    return *this;
  }
  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  size_t size() const { return count_; }
 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

struct ErrorMetrics {
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double rmse = 0.0;
  double relative_l2 = 0.0;
};

inline int8_t sign_extend_int4(uint8_t nibble) {
  nibble &= 0x0f;
  return static_cast<int8_t>((nibble ^ 0x08u) - 0x08u);
}

__device__ __forceinline__ int unpack_low(uint8_t x) {
  return int(static_cast<int8_t>(x << 4) >> 4);
}

__device__ __forceinline__ int unpack_high(uint8_t x) {
  return int(static_cast<int8_t>(x) >> 4);
}

}  // namespace int4gemm
