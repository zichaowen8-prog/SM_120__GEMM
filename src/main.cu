#include "benchmark.cuh"
#include "kernels.cuh"
#include "layout.cuh"
#include "quant.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

namespace int4gemm {
namespace {

struct CublasHandle {
  cublasHandle_t h{};
  CublasHandle() { CUBLAS_CHECK(cublasCreate(&h)); CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH)); }
  ~CublasHandle() { if (h) cublasDestroy(h); }
};

void cublas_row_gemm(cublasHandle_t h, const half* a, const half* b, half* c,
                     int M, int N, int K) {
  const float alpha = 1.0f, beta = 0.0f;
  // Row-major C=A*B is column-major C^T=B^T*A^T.
  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                            &alpha, b, CUDA_R_16F, N, a, CUDA_R_16F, K,
                            &beta, c, CUDA_R_16F, N,
                            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void cublas_row_gemm_fp32(cublasHandle_t h, const half* a, const half* b, float* c,
                          int M, int N, int K) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                            &alpha, b, CUDA_R_16F, N, a, CUDA_R_16F, K,
                            &beta, c, CUDA_R_32F, N,
                            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

__global__ void disturb_kernel(uint32_t* p, size_t count) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) p[i] = p[i] * 1664525u + uint32_t(i) + 1013904223u;
}

TimingStats benchmark_cold(KernelLaunch fn, const half* a, const uint8_t* b, const half* s,
                           half* c, int M, int N, int K, int group,
                           uint32_t* trash, size_t trash_count, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    disturb_kernel<<<(trash_count + 255) / 256, 256>>>(trash, trash_count);
    fn(a, b, s, c, M, N, K, group, nullptr);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t begin, end; CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end));
  std::vector<float> times(iters);
  for (int i = 0; i < iters; ++i) {
    disturb_kernel<<<(trash_count + 255) / 256, 256>>>(trash, trash_count);
    CUDA_CHECK(cudaEventRecord(begin));
    fn(a, b, s, c, M, N, K, group, nullptr);
    CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&times[i], begin, end));
  }
  CUDA_CHECK(cudaEventDestroy(begin)); CUDA_CHECK(cudaEventDestroy(end));
  std::sort(times.begin(), times.end());
  TimingStats x;
  x.min_ms = times.front(); x.median_ms = times[times.size()/2];
  x.p95_ms = times[std::min(times.size()-1, size_t(std::ceil(times.size()*.95)-1))];
  x.mean_ms = std::accumulate(times.begin(), times.end(), 0.0f)/times.size();
  return x;
}

std::vector<half> random_half(size_t count, uint64_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<half> x(count);
  for (half& v : x) v = __float2half(dist(gen));
  return x;
}

void print_hardware() {
  int runtime = 0, driver = 0, count = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime)); CUDA_CHECK(cudaDriverGetVersion(&driver));
  CUDA_CHECK(cudaGetDeviceCount(&count));
  std::cout << "CUDA runtime: " << runtime << "\nCUDA driver API: " << driver << "\nDevice count: " << count << "\n";
  for (int d = 0; d < count; ++d) {
    cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p, d));
    int core_clock = 0, memory_clock = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&core_clock, cudaDevAttrClockRate, d));
    CUDA_CHECK(cudaDeviceGetAttribute(&memory_clock, cudaDevAttrMemoryClockRate, d));
    const double theoretical_bandwidth_gbs =
        2.0 * double(memory_clock) * 1000.0 * double(p.memoryBusWidth) / 8.0 / 1.0e9;
    std::cout << "GPU " << d << ": " << p.name
              << "\ncompute capability: " << p.major << '.' << p.minor
              << "\nSM count: " << p.multiProcessorCount
              << "\nwarp size: " << p.warpSize
              << "\nmax threads/SM: " << p.maxThreadsPerMultiProcessor
              << "\nmax threads/block: " << p.maxThreadsPerBlock
              << "\nregisters/block: " << p.regsPerBlock
              << "\nregisters/SM: " << p.regsPerMultiprocessor
              << "\nshared memory/block: " << p.sharedMemPerBlock
              << "\nshared memory/block opt-in: " << p.sharedMemPerBlockOptin
              << "\nshared memory/SM: " << p.sharedMemPerMultiprocessor
              << "\nL2 bytes: " << p.l2CacheSize
              << "\ncore clock kHz: " << core_clock
              << "\nmemory clock kHz: " << memory_clock
              << "\nmemory bus bits: " << p.memoryBusWidth
              << "\ntheoretical DDR memory bandwidth GB/s: " << theoretical_bandwidth_gbs
              << "\nglobal memory bytes: " << p.totalGlobalMem
              << "\nasync engines: " << p.asyncEngineCount
              << "\nconcurrent kernels: " << p.concurrentKernels << "\n";
  }
}

struct CaseBuffers {
  int M, N, K, group;
  std::vector<half> h_a, h_b;
  DeviceBuffer<half> a, b_original, scales, repacked_scales, dequant, c, reference;
  DeviceBuffer<uint8_t> packed, repacked, mma_repacked, mma_pair_repacked, inverse;
  CaseBuffers(int m, int n, int k, int g)
      : M(m), N(n), K(k), group(g), h_a(random_half(size_t(m)*k, 1234)),
        h_b(random_half(size_t(k)*n, 5678)), a(size_t(m)*k), b_original(size_t(k)*n),
        scales(size_t(k/g)*n), repacked_scales(size_t(k/g)*n), dequant(size_t(k)*n),
        c(size_t(m)*n), reference(size_t(m)*n), packed(size_t(k)*n/2),
        repacked(size_t(k)*n/2), mma_repacked(size_t(k)*n/2),
        mma_pair_repacked(size_t(k)*n/2), inverse(size_t(k)*n/2) {
    CUDA_CHECK(cudaMemcpy(a.get(), h_a.data(), a.size()*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_original.get(), h_b.data(), b_original.size()*sizeof(half), cudaMemcpyHostToDevice));
    launch_quantize(b_original.get(), packed.get(), scales.get(), K, N, group);
    launch_repack_128x32(packed.get(), scales.get(), repacked.get(), repacked_scales.get(), K, N, group);
    launch_repack_mma_128x32(packed.get(), mma_repacked.get(), K, N);
    launch_repack_mma_pair_128x32(packed.get(), mma_pair_repacked.get(), K, N);
    launch_dequantize(packed.get(), scales.get(), dequant.get(), K, N, group);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
};

void validate_repack(CaseBuffers& x) {
  launch_inverse_repack_128x32(x.repacked.get(), x.inverse.get(), x.K, x.N);
  std::vector<uint8_t> src(x.packed.size()), inv(x.inverse.size()), rep(x.repacked.size());
  CUDA_CHECK(cudaMemcpy(src.data(), x.packed.get(), src.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(inv.data(), x.inverse.get(), inv.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(rep.data(), x.repacked.get(), rep.size(), cudaMemcpyDeviceToHost));
  if (src != inv) throw std::runtime_error("GPU repack inverse test failed");
  // Independent CPU mapping reference for the GPU repacker.
  std::vector<uint8_t> cpu(rep.size(), 0);
  for (int k = 0; k < x.K; ++k) for (int n = 0; n < x.N; ++n) {
    const size_t logical = static_cast<size_t>(k)*x.N+n;
    int q = (logical & 1) ? sign_extend_int4(src[logical>>1]>>4) : sign_extend_int4(src[logical>>1]);
    size_t dst = repacked_element<128,32>(k,n,x.N,x.K);
    if (dst & 1) cpu[dst>>1] |= uint8_t((q&15)<<4); else cpu[dst>>1] = uint8_t(q&15);
  }
  if (cpu != rep) throw std::runtime_error("GPU repack != independent CPU mapping");

  std::vector<uint8_t> mma(x.mma_repacked.size()), mma_cpu(mma.size(), 0);
  CUDA_CHECK(cudaMemcpy(mma.data(), x.mma_repacked.get(), mma.size(), cudaMemcpyDeviceToHost));
  constexpr int words_per_tile = 32 * 128 / 4;
  const int ktiles = x.K / 32;
  const int ntiles = x.N / 128;
  auto q4 = [&](int k, int n) {
    const size_t logical = static_cast<size_t>(k) * x.N + n;
    return int((logical & 1) ? (src[logical >> 1] >> 4) : (src[logical >> 1] & 15));
  };
  for (int nt = 0; nt < ntiles; ++nt) for (int kt = 0; kt < ktiles; ++kt)
    for (int k16 = 0; k16 < 2; ++k16) for (int wn = 0; wn < 2; ++wn)
      for (int j = 0; j < 8; ++j) for (int lane = 0; lane < 32; ++lane) {
        const int gid = lane >> 2, t = lane & 3;
        const int n = nt * 128 + wn * 64 + j * 8 + gid;
        const int k0 = kt * 32 + k16 * 16 + t * 2;
        const uint16_t word = uint16_t(q4(k0, n) | (q4(k0 + 1, n) << 4) |
                                       (q4(k0 + 8, n) << 8) | (q4(k0 + 9, n) << 12));
        const size_t wi = (static_cast<size_t>(nt) * ktiles + kt) * words_per_tile
                        + mma_fragment_word(k16, wn, j, lane);
        mma_cpu[wi * 2] = uint8_t(word);
        mma_cpu[wi * 2 + 1] = uint8_t(word >> 8);
      }
  if (mma_cpu != mma) throw std::runtime_error("GPU MMA-fragment repack != CPU reference");

  std::vector<uint8_t> pair(x.mma_pair_repacked.size());
  std::vector<uint8_t> pair_cpu(pair.size(), 0);
  CUDA_CHECK(cudaMemcpy(pair.data(), x.mma_pair_repacked.get(), pair.size(),
                        cudaMemcpyDeviceToHost));
  constexpr int pairs_per_tile = 32 * 128 / 8;
  for (int nt = 0; nt < ntiles; ++nt) for (int kt = 0; kt < ktiles; ++kt)
    for (int wn = 0; wn < 2; ++wn) for (int j = 0; j < 8; ++j)
      for (int lane = 0; lane < 32; ++lane) {
        const int gid = lane >> 2, t = lane & 3;
        const int n = nt * 128 + wn * 64 + j * 8 + gid;
        uint32_t pair_word = 0;
        for (int k16 = 0; k16 < 2; ++k16) {
          const int k0 = kt * 32 + k16 * 16 + t * 2;
          const uint16_t fragment = uint16_t(
              q4(k0, n) | (q4(k0 + 1, n) << 4) |
              (q4(k0 + 8, n) << 8) | (q4(k0 + 9, n) << 12));
          pair_word |= uint32_t(fragment) << (k16 * 16);
        }
        const size_t wi = (static_cast<size_t>(nt) * ktiles + kt) * pairs_per_tile
                        + mma_fragment_pair_word(wn, j, lane);
        pair_cpu[wi * 4] = uint8_t(pair_word);
        pair_cpu[wi * 4 + 1] = uint8_t(pair_word >> 8);
        pair_cpu[wi * 4 + 2] = uint8_t(pair_word >> 16);
        pair_cpu[wi * 4 + 3] = uint8_t(pair_word >> 24);
      }
  if (pair_cpu != pair)
    throw std::runtime_error("GPU MMA fragment-pair repack != CPU reference");
}

bool uses_mma_repack(const std::string& version) {
  return version.rfind("v16b", 0) == 0 || version.rfind("v16c", 0) == 0 ||
         version.rfind("v16d", 0) == 0 || version == "v17a" ||
         version == "v17b" || version == "final";
}

bool uses_mma_pair_repack(const std::string& version, int group) {
  return version == "v17c" || version == "v17d" || version == "v17e" ||
         version.rfind("v18", 0) == 0 ||
         (version == "final" && (group == 32 || group == 128));
}

std::vector<half> copy_half(const half* p, size_t count) {
  std::vector<half> h(count); CUDA_CHECK(cudaMemcpy(h.data(), p, count*sizeof(half), cudaMemcpyDeviceToHost)); return h;
}

void correctness(int size, int group) {
  std::cout << "correctness M=N=K=" << size << " group=" << group << "\n";
  CaseBuffers x(size, size, size, group);
  validate_repack(x);
  CublasHandle blas;
  cublas_row_gemm(blas.h, x.a.get(), x.dequant.get(), x.reference.get(), x.M, x.N, x.K);
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto href = copy_half(x.reference.get(), size_t(size)*size);

  if (size <= 256) {
    std::vector<uint8_t> hp(x.packed.size()); std::vector<half> hs(x.scales.size()), hbq;
    CUDA_CHECK(cudaMemcpy(hp.data(), x.packed.get(), hp.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hs.data(), x.scales.get(), hs.size()*sizeof(half), cudaMemcpyDeviceToHost));
    dequantize_cpu(hp, hs, hbq, size, size, group);
    std::vector<float> cref; cpu_gemm_fp32(x.h_a, hbq, cref, size, size, size);
    std::vector<half> cref_half(cref.size()); for (size_t i=0;i<cref.size();++i) cref_half[i]=__float2half(cref[i]);
    ErrorMetrics cm = compare_half(href, cref_half);
    std::cout << "cpu_fp32_vs_cublas max_abs=" << cm.max_abs << " mean_abs=" << cm.mean_abs
              << " rmse=" << cm.rmse << " rel_l2=" << cm.relative_l2 << "\n";
  }

  for (auto& entry : kernel_registry()) {
    const auto& meta = entry.first; auto fn = entry.second;
    const std::string version = meta.version;
    const bool uses_repack = version != "v00" && version != "v01" && version != "v03";
    const uint8_t* b_arg = uses_mma_pair_repack(version, group) ? x.mma_pair_repacked.get()
                         : uses_mma_repack(version) ? x.mma_repacked.get()
                                                    : (uses_repack ? x.repacked.get() : x.packed.get());
    fn(x.a.get(), b_arg,
       uses_repack?x.repacked_scales.get():x.scales.get(), x.c.get(), size,size,size,group,nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    ErrorMetrics e = compare_half(copy_half(x.c.get(), size_t(size)*size), href);
    const bool pass = e.relative_l2 < 0.02 && e.max_abs < 2.0;
    std::cout << meta.version << " max_abs=" << e.max_abs << " mean_abs=" << e.mean_abs
              << " rmse=" << e.rmse << " rel_l2=" << e.relative_l2 << " " << (pass?"PASS":"FAIL") << "\n";
    if (!pass) throw std::runtime_error(std::string(meta.version)+" correctness failed");
  }
}

#if !defined(INT4_GEMM_ENABLE_FP32_OUTPUT) || INT4_GEMM_ENABLE_FP32_OUTPUT
void fp32_output_check(int size, int group) {
  CaseBuffers x(size,size,size,group); validate_repack(x); CublasHandle blas;
  DeviceBuffer<float> got(size_t(size)*size), ref(size_t(size)*size);
  cublas_row_gemm_fp32(blas.h,x.a.get(),x.dequant.get(),ref.get(),size,size,size);
  launch_final_fp32(x.a.get(),x.repacked.get(),x.repacked_scales.get(),got.get(),size,size,size,group,nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> hg(got.size()),hr(ref.size());
  CUDA_CHECK(cudaMemcpy(hg.data(),got.get(),hg.size()*sizeof(float),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hr.data(),ref.get(),hr.size()*sizeof(float),cudaMemcpyDeviceToHost));
  long double ae=0,se=0,rs=0;double ma=0;
  for(size_t i=0;i<hg.size();++i){double d=std::fabs(double(hg[i])-hr[i]);ma=std::max(ma,d);ae+=d;se+=d*d;rs+=double(hr[i])*hr[i];}
  double mean=double(ae/hg.size()),rmse=std::sqrt(double(se/hg.size())),rel=std::sqrt(double(se/std::max(rs,(long double)1e-30)));
  std::cout<<"fp32_output size="<<size<<" group="<<group<<" max_abs="<<ma<<" mean_abs="<<mean<<" rmse="<<rmse<<" rel_l2="<<rel<<"\n";
  if(rel>=0.02||ma>=2.0)throw std::runtime_error("FP32 output correctness failed");
}
#endif

struct CsvRow {
  std::string version, description, cache, notes;
  int M=0,N=0,K=0,group=0,BM=0,BN=0,BK=0,wm=0,wn=0,stages=0,threads=0,regs=-1,smem=0;
  float occupancy=-1; TimingStats t; double tf=0, prev=0, base=0, max_abs=0, rmse=0;
};

void write_csv(const std::string& path, const std::vector<CsvRow>& rows) {
  std::ofstream f(path); if (!f) throw std::runtime_error("cannot open "+path);
  f << "version,description,M,N,K,group_size,BM,BN,BK,warp_tile_m,warp_tile_n,stages,threads,registers_per_thread,smem_bytes,occupancy,cache_mode,median_ms,mean_ms,min_ms,p95_ms,effective_tflops,speedup_vs_previous,speedup_vs_baseline,max_abs_error,rmse,notes\n";
  f << std::setprecision(8);
  for (const auto& r:rows)
    f<<r.version<<",\""<<r.description<<"\","<<r.M<<','<<r.N<<','<<r.K<<','<<r.group<<','<<r.BM<<','<<r.BN<<','<<r.BK<<','<<r.wm<<','<<r.wn<<','<<r.stages<<','<<r.threads<<','<<r.regs<<','<<r.smem<<','<<r.occupancy<<','<<r.cache<<','<<r.t.median_ms<<','<<r.t.mean_ms<<','<<r.t.min_ms<<','<<r.t.p95_ms<<','<<r.tf<<','<<r.prev<<','<<r.base<<','<<r.max_abs<<','<<r.rmse<<",\""<<r.notes<<"\"\n";
}

void benchmark_all(int M, int N, int K, const std::vector<int>& groups, int warmup, int iters,
                   const std::string& csv_path, const std::string& only) {
  std::vector<CsvRow> rows;
  cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  CublasHandle blas;
  constexpr size_t trash_count = 128ull*1024*1024/sizeof(uint32_t);
  DeviceBuffer<uint32_t> trash(trash_count); CUDA_CHECK(cudaMemset(trash.get(), 1, trash.size()*sizeof(uint32_t)));
  for (int group:groups) {
    CaseBuffers x(M,N,K,group);
    validate_repack(x);
    TimingStats tq = benchmark_cuda([&]{launch_quantize(x.b_original.get(),x.packed.get(),x.scales.get(),K,N,group);},warmup,iters);
    TimingStats tr = benchmark_cuda([&]{launch_repack_128x32(x.packed.get(),x.scales.get(),x.repacked.get(),x.repacked_scales.get(),K,N,group);},warmup,iters);
    TimingStats tr_mma = benchmark_cuda([&]{
      launch_repack_mma_128x32(x.packed.get(),x.mma_repacked.get(),K,N);
      launch_repack_scales_128(x.scales.get(),x.repacked_scales.get(),K,N,group);
    },warmup,iters);
    TimingStats tr_pair = benchmark_cuda([&]{
      launch_repack_mma_pair_128x32(x.packed.get(),x.mma_pair_repacked.get(),K,N);
      launch_repack_scales_128(x.scales.get(),x.repacked_scales.get(),K,N,group);
    },warmup,iters);
    TimingStats td = benchmark_cuda([&]{launch_dequantize(x.packed.get(),x.scales.get(),x.dequant.get(),K,N,group);},warmup,iters);
    TimingStats tg = benchmark_cuda([&]{cublas_row_gemm(blas.h,x.a.get(),x.dequant.get(),x.c.get(),M,N,K);},warmup,iters);
    TimingStats ttotal = benchmark_cuda([&]{launch_dequantize(x.packed.get(),x.scales.get(),x.dequant.get(),K,N,group); cublas_row_gemm(blas.h,x.a.get(),x.dequant.get(),x.c.get(),M,N,K);},warmup,iters);
    TimingStats te2e = benchmark_cuda([&]{
      launch_quantize(x.b_original.get(),x.packed.get(),x.scales.get(),K,N,group);
      if (group == 32 || group == 128)
        launch_repack_mma_pair_128x32(x.packed.get(),x.mma_pair_repacked.get(),K,N);
      else
        launch_repack_mma_128x32(x.packed.get(),x.mma_repacked.get(),K,N);
      launch_repack_scales_128(x.scales.get(),x.repacked_scales.get(),K,N,group);
      const uint8_t* final_b = (group == 32 || group == 128)
                                   ? x.mma_pair_repacked.get()
                                   : x.mma_repacked.get();
      launch_final(x.a.get(),final_b,x.repacked_scales.get(),x.c.get(),M,N,K,group,nullptr);
    },warmup,iters);
    std::cout << "group="<<group<<" quant="<<tq.median_ms<<"ms repack="<<tr.median_ms
              <<"ms mma_repack="<<tr_mma.median_ms<<"ms pair_repack="<<tr_pair.median_ms
              <<"ms dequant="<<td.median_ms
              <<"ms cublas="<<tg.median_ms<<"ms separate_total="<<ttotal.median_ms
              <<"ms end_to_end="<<te2e.median_ms<<"ms\n";
    rows.push_back({"quantize","GPU symmetric group quantization","preprocess","",M,N,K,group,0,0,0,0,0,0,256,-1,0,-1,tq,0});
    rows.push_back({"repack","CTA tile-contiguous BN128 BK32","preprocess","",M,N,K,group,0,128,32,0,0,0,256,-1,0,-1,tr,0});
    rows.push_back({"repack_mma","lane-native MMA fragments + tile-contiguous scales","preprocess","weights and scales; no allocation in timed region",M,N,K,group,0,128,32,0,0,0,256,-1,0,-1,tr_mma,0});
    rows.push_back({"repack_mma_pair","two k16 fragments per uint32 + tile-contiguous scales","preprocess","weights and scales; no allocation in timed region",M,N,K,group,0,128,32,0,0,0,256,-1,0,-1,tr_pair,0});
    rows.push_back({"end_to_end","quantize + final-layout repack + fused final GEMM","one-shot","preprocessing is normally amortized for inference weights",M,N,K,group,64,128,32,64,32,2,128,-1,0,-1,te2e,tflops(M,N,K,te2e.median_ms)});
    rows.push_back({"dequant","materialize packed INT4 B as FP16","baseline-component","separately timed component of dequant + cuBLAS",M,N,K,group,0,0,0,0,0,0,256,-1,0,-1,td,0});
    rows.push_back({"cublas_fp16","cuBLAS FP16 Tensor Core on dequantized B","steady","not workload-equivalent: B already materialized",M,N,K,group,0,0,0,0,0,0,0,-1,0,-1,tg,tflops(M,N,K,tg.median_ms)});
    rows.push_back({"separate","dequant kernel + cuBLAS","steady","includes dequant and GEMM",M,N,K,group,0,0,0,0,0,0,0,-1,0,-1,ttotal,tflops(M,N,K,ttotal.median_ms)});

    cublas_row_gemm(blas.h,x.a.get(),x.dequant.get(),x.reference.get(),M,N,K); CUDA_CHECK(cudaDeviceSynchronize());
    const auto href=copy_half(x.reference.get(),size_t(M)*N);
    double baseline_ms=0, previous_ms=0;
    for (auto& entry:kernel_registry()) {
      const auto& md=entry.first; auto fn=entry.second;
      const std::string ver=md.version;
      if (only=="padding" && ver.rfind("v10",0)!=0 && ver.rfind("v11",0)!=0 && ver.rfind("v12",0)!=0) continue;
      if (only=="scheduling" && ver!="v13p" && ver!="final") continue;
      if (only=="warp_shapes" && ver!="v14a" && ver!="v14b" && ver!="final") continue;
      if (only=="tma" && ver.rfind("v15",0)!=0 && ver!="final") continue;
      if (only=="v16" && ver.rfind("v16",0)!=0 && ver!="final") continue;
      if (only=="v16_best" && ver!="v16b" && ver!="v16c0" && ver!="final") continue;
      if (only=="v16_tile" && ver!="v16b" && ver!="v16c0" &&
          ver.rfind("v16d",0)!=0 && ver!="final") continue;
      if (only=="v16_finalists" && ver!="v16b" && ver!="v16c0" &&
          ver!="v16d2" && ver!="final") continue;
      if (only=="v17" && ver.rfind("v17",0)!=0 && ver!="final") continue;
      if (only=="v18" && ver.rfind("v18",0)!=0 && ver!="final") continue;
      if (only=="selected" && ver!="v11d" && ver!="final") continue;
      if (!only.empty() && only!="padding" && only!="scheduling" && only!="warp_shapes"
          && only!="tma" && only!="v16" && only!="v16_best" && only!="v16_tile"
          && only!="v16_finalists" && only!="v17" && only!="v18"
          && only!="selected" && ver!=only) continue;
      bool rp=ver!="v00"&&ver!="v01"&&ver!="v03";
      const uint8_t* bp=uses_mma_pair_repack(ver,group)?x.mma_pair_repacked.get():
                        (uses_mma_repack(ver)?x.mma_repacked.get():(rp?x.repacked.get():x.packed.get())); const half* sp=rp?x.repacked_scales.get():x.scales.get();
      fn(x.a.get(),bp,sp,x.c.get(),M,N,K,group,nullptr); CUDA_CHECK(cudaDeviceSynchronize());
      ErrorMetrics err=compare_half(copy_half(x.c.get(),size_t(M)*N),href);
      if (err.relative_l2>=0.02||err.max_abs>=2.0) throw std::runtime_error(ver+" benchmark correctness failed");
      TimingStats ts=benchmark_cuda([&]{fn(x.a.get(),bp,sp,x.c.get(),M,N,K,group,nullptr);},warmup,iters);
      KernelResources kr=query_resources(ver,group);
      kr.occupancy=float(kr.active_blocks_per_sm*md.threads)/prop.maxThreadsPerMultiProcessor;
      if (!baseline_ms) baseline_ms=ts.median_ms;
      const int reported_smem = ver == "final"
                                    ? (group == 128 ? 12832 : 12816)
                                    : md.smem_bytes;
      CsvRow row{ver,md.description,"steady","",M,N,K,group,md.BM,md.BN,md.BK,md.warp_m,md.warp_n,md.stages,md.threads,kr.registers_per_thread,reported_smem,kr.occupancy,ts,tflops(M,N,K,ts.median_ms),previous_ms?previous_ms/ts.median_ms:1.0,baseline_ms/ts.median_ms,err.max_abs,err.rmse};
      rows.push_back(row); previous_ms=ts.median_ms;
      std::cout<<ver<<" median="<<ts.median_ms<<"ms mean="<<ts.mean_ms<<" min="<<ts.min_ms
               <<" p95="<<ts.p95_ms<<" TFLOP/s="<<row.tf<<" regs="<<kr.registers_per_thread
               <<" smem="<<md.smem_bytes<<" occupancy="<<kr.occupancy<<" error="<<err.max_abs<<"\n";
      if (ver=="final") {
        TimingStats cold=benchmark_cold(fn,x.a.get(),bp,sp,x.c.get(),M,N,K,group,trash.get(),trash.size(),warmup,iters);
        CsvRow cr=row; cr.version="final_cold"; cr.cache="cold-ish"; cr.t=cold; cr.tf=tflops(M,N,K,cold.median_ms); cr.prev=ts.median_ms/cold.median_ms; cr.base=baseline_ms/cold.median_ms; cr.notes="128 MiB cache-disturb kernel before each untimed GEMM event"; rows.push_back(cr);
        std::cout<<"final_cold median="<<cold.median_ms<<"ms TFLOP/s="<<cr.tf<<"\n";
      }
    }
  }
  write_csv(csv_path,rows);
}

void profile_one(int size, int group, const std::string& wanted) {
  CaseBuffers x(size,size,size,group);
  for (auto& entry:kernel_registry()) if (wanted==entry.first.version) {
    bool rp=wanted!="v00"&&wanted!="v01"&&wanted!="v03";
    const uint8_t* bp=uses_mma_pair_repack(wanted,group)?x.mma_pair_repacked.get():
                      (uses_mma_repack(wanted)?x.mma_repacked.get():(rp?x.repacked.get():x.packed.get())); const half* sp=rp?x.repacked_scales.get():x.scales.get();
    for(int i=0;i<10;++i)entry.second(x.a.get(),bp,sp,x.c.get(),size,size,size,group,nullptr);
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaProfilerStart());
    entry.second(x.a.get(),bp,sp,x.c.get(),size,size,size,group,nullptr);
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaProfilerStop());
    std::cout<<"profiled "<<wanted<<" size="<<size<<" group="<<group<<"\n"; return;
  }
  throw std::runtime_error("unknown profile kernel: "+wanted);
}

}  // namespace
}  // namespace int4gemm

int main(int argc, char** argv) {
  using namespace int4gemm;
  try {
    std::string mode="hardware", csv="results/benchmark.csv", profile_version, only; int size=256, group=32, warmup=20, iters=100; bool all_groups=false;
    for(int i=1;i<argc;++i){std::string a=argv[i];
      if(a=="--hardware")mode="hardware"; else if(a=="--correctness")mode="correctness"; else if(a=="--benchmark")mode="benchmark";
#if !defined(INT4_GEMM_ENABLE_FP32_OUTPUT) || INT4_GEMM_ENABLE_FP32_OUTPUT
      else if(a=="--fp32-output-check")mode="fp32_output";
#endif
      else if(a=="--profile"&&i+1<argc){mode="profile";profile_version=argv[++i];}
      else if(a=="--size"&&i+1<argc)size=std::stoi(argv[++i]); else if(a=="--group"&&i+1<argc){std::string g=argv[++i];if(g=="all")all_groups=true;else group=std::stoi(g);}
      else if(a=="--warmup"&&i+1<argc)warmup=std::stoi(argv[++i]); else if(a=="--iters"&&i+1<argc)iters=std::stoi(argv[++i]); else if(a=="--csv"&&i+1<argc)csv=argv[++i];
      else if(a=="--only"&&i+1<argc)only=argv[++i];
      else throw std::runtime_error("unknown argument: "+a);
    }
    if(mode=="hardware") print_hardware();
    else if(mode=="correctness") correctness(size,group);
#if !defined(INT4_GEMM_ENABLE_FP32_OUTPUT) || INT4_GEMM_ENABLE_FP32_OUTPUT
    else if(mode=="fp32_output") fp32_output_check(size,group);
#endif
    else if(mode=="benchmark") benchmark_all(size,size,size,all_groups?std::vector<int>{32,64,128}:std::vector<int>{group},warmup,iters,csv,only);
    else profile_one(size,group,profile_version);
  } catch(const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<"\n"; return 1; }
  return 0;
}
