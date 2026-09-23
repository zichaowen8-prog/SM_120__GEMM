#pragma once
#include "common.cuh"
#include <mma.h>

namespace int4gemm::async_tc {
namespace wmma = nvcuda::wmma;

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(addr), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template<int N> __device__ __forceinline__ void cp_wait() {
  if constexpr (N == 1) asm volatile("cp.async.wait_group 1;");
  else if constexpr (N == 2) asm volatile("cp.async.wait_group 2;");
  else if constexpr (N == 3) asm volatile("cp.async.wait_group 3;");
}

template<int STAGES, int GROUP, int APAD=0, int BPAD=0,
         bool FP32_OUT=false, bool PERSISTENT=false,
         int WARP_M=32, int WARP_N=64,
         bool WARP_EPILOGUE=false>
__global__ void kernel(const half* __restrict__ a, const uint8_t* __restrict__ b,
                       const half* __restrict__ scales, void* __restrict__ c_raw,
                       int M, int N, int K) {
  constexpr int BM=64, BN=128, BK=32;
  constexpr int WARPS_M=BM/WARP_M, WARPS_N=BN/WARP_N;
  constexpr int THREADS=WARPS_M*WARPS_N*32;
  static_assert(BM%WARP_M==0 && BN%WARP_N==0 && WARP_M%16==0 && WARP_N%16==0);
  constexpr int A_STRIDE=BK+APAD, B_STRIDE=BN+BPAD;
  constexpr int A_HALF=BM*A_STRIDE, PACKED_BYTES=BK*BN/2;
  extern __shared__ __align__(16) unsigned char raw[];
  half* astage=reinterpret_cast<half*>(raw);
  uint8_t* bpack=raw + STAGES*A_HALF*sizeof(half);
  half* bs=reinterpret_cast<half*>(bpack + STAGES*PACKED_BYTES);
  float* out=reinterpret_cast<float*>(raw);
  const int tid=threadIdx.x, warp=tid/32, lane=tid&31;
  const int ktiles=K/BK, groups=K/GROUP, ntiles=N/BN;

  const int total_tiles=(M/BM)*ntiles;
  const int first_tile=PERSISTENT?int(blockIdx.x):int(blockIdx.y)*ntiles+int(blockIdx.x);
  const int tile_stride=PERSISTENT?int(gridDim.x):total_tiles;
  for(int tile_id=first_tile;tile_id<total_tiles;tile_id+=tile_stride) {
  const int tile_m=tile_id/ntiles;
  const int block_m=tile_m*BM;
  const int nt=tile_id-tile_m*ntiles;
  const int block_n=nt*BN;

  auto issue_tile = [&](int stage, int kt) {
    half* adst=astage+stage*A_HALF;
    for(int chunk=tid;chunk<BM*(BK/8);chunk+=THREADS) {
      int mi=chunk/(BK/8), ki=(chunk%(BK/8))*8;
      cp_async_16(adst+mi*A_STRIDE+ki,a+static_cast<size_t>(block_m+mi)*K+kt*BK+ki);
    }
    uint8_t* bdst=bpack+stage*PACKED_BYTES;
    const uint8_t* bsrc=b+(static_cast<size_t>(nt)*ktiles+kt)*PACKED_BYTES;
    for(int chunk=tid;chunk<PACKED_BYTES/16;chunk+=THREADS)
      cp_async_16(bdst+chunk*16,bsrc+chunk*16);
    cp_commit();
  };

  wmma::fragment<wmma::accumulator,16,16,16,float> acc[WARP_M/16][WARP_N/16];
  #pragma unroll
  for(int i=0;i<WARP_M/16;++i) for(int j=0;j<WARP_N/16;++j) wmma::fill_fragment(acc[i][j],0.0f);

  #pragma unroll
  for(int s=0;s<STAGES;++s) issue_tile(s,s);
  for(int kt=0;kt<ktiles;++kt) {
    cp_wait<STAGES-1>();
    __syncthreads();
    const int stage=kt%STAGES;
    const uint8_t* packed=bpack+stage*PACKED_BYTES;
    for(int pair=tid;pair<PACKED_BYTES;pair+=THREADS) {
      int e=pair*2, ki=e/BN, ni=e%BN;
      uint8_t x=packed[pair];
      const half* sp=scales+(static_cast<size_t>(nt)*groups+(kt*BK+ki)/GROUP)*BN+ni;
      reinterpret_cast<half2*>(bs+ki*B_STRIDE+ni)[0]=__floats2half2_rn(unpack_low(x)*__half2float(sp[0]),unpack_high(x)*__half2float(sp[1]));
    }
    __syncthreads();
    const half* as=astage+stage*A_HALF;
    const int wm=warp/WARPS_N, wn=warp%WARPS_N;
    #pragma unroll
    for(int kk=0;kk<BK;kk+=16) {
      wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af[WARP_M/16];
      wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf[WARP_N/16];
      #pragma unroll
      for(int i=0;i<WARP_M/16;++i) wmma::load_matrix_sync(af[i],as+(wm*WARP_M+i*16)*A_STRIDE+kk,A_STRIDE);
      #pragma unroll
      for(int j=0;j<WARP_N/16;++j) wmma::load_matrix_sync(bf[j],bs+kk*B_STRIDE+wn*WARP_N+j*16,B_STRIDE);
      #pragma unroll
      for(int i=0;i<WARP_M/16;++i) for(int j=0;j<WARP_N/16;++j) wmma::mma_sync(acc[i][j],af[i],bf[j],acc[i][j]);
    }
    __syncthreads();
    if(kt+STAGES<ktiles) issue_tile(stage,kt+STAGES);
  }
  asm volatile("cp.async.wait_all;");
  __syncthreads();
  const int wm=warp/WARPS_N,wn=warp%WARPS_N;
  if constexpr (WARP_EPILOGUE) {
    static_assert(!FP32_OUT, "warp-streamed epilogue is specialized for FP16 output");
    // WMMA accumulator lane ownership is intentionally opaque.  Reuse one
    // 16x16 FP32 scratch tile per warp, then immediately convert/store it.
    // This preserves the documented WMMA store mapping while reducing the
    // CTA epilogue footprint from BM*BN floats (32 KiB) to 4*16*16 floats
    // (4 KiB).  Mainloop staging is therefore the shared-memory high-water
    // mark instead of the epilogue.
    float* warp_out=out+warp*16*16;
    half* c=static_cast<half*>(c_raw);
    #pragma unroll
    for(int i=0;i<WARP_M/16;++i) {
      #pragma unroll
      for(int j=0;j<WARP_N/16;++j) {
        wmma::store_matrix_sync(warp_out,acc[i][j],16,wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for(int p=lane;p<16*16/2;p+=32) {
          const int e=p*2,mi=e/16,ni=e-mi*16;
          half2 value=__floats2half2_rn(warp_out[e],warp_out[e+1]);
          half* dst=c+static_cast<size_t>(block_m+wm*WARP_M+i*16+mi)*N
                     +block_n+wn*WARP_N+j*16+ni;
          reinterpret_cast<half2*>(dst)[0]=value;
        }
        __syncwarp();
      }
    }
  } else {
    #pragma unroll
    for(int i=0;i<WARP_M/16;++i) for(int j=0;j<WARP_N/16;++j)
      wmma::store_matrix_sync(out+(wm*WARP_M+i*16)*BN+wn*WARP_N+j*16,acc[i][j],BN,wmma::mem_row_major);
    __syncthreads();
    for(int p=tid;p<BM*BN/2;p+=THREADS) {
      int e=p*2,mi=e/BN,ni=e%BN;
      if constexpr (FP32_OUT) {
        float* c=static_cast<float*>(c_raw);
        reinterpret_cast<float2*>(c+static_cast<size_t>(block_m+mi)*N+block_n+ni)[0]=make_float2(out[e],out[e+1]);
      } else {
        half* c=static_cast<half*>(c_raw);
        reinterpret_cast<half2*>(c+static_cast<size_t>(block_m+mi)*N+block_n+ni)[0]=__floats2half2_rn(out[e],out[e+1]);
      }
    }
  }
  if constexpr (PERSISTENT) {
    __syncthreads();
  }
  }
}

template<int STAGES, int APAD=0, int BPAD=0, int WARP_M=32, int WARP_N=64>
void launch_warp_epilogue(const half* a,const uint8_t* b,const half* s,half* c,
                          int M,int N,int K,int group,cudaStream_t stream) {
  constexpr int threads=(64/WARP_M)*(128/WARP_N)*32;
  constexpr int warps=threads/32;
  constexpr int input=STAGES*(64*(32+APAD)*2+32*128/2)+32*(128+BPAD)*2;
  constexpr int output=warps*16*16*4;
  constexpr int smem=input>output?input:output;
  auto f=[&](auto g){constexpr int G=decltype(g)::value;auto fn=kernel<STAGES,G,APAD,BPAD,false,false,WARP_M,WARP_N,true>;fn<<<dim3(N/128,M/64),threads,smem,stream>>>(a,b,s,c,M,N,K);};
  if(group==32)f(std::integral_constant<int,32>{});else if(group==64)f(std::integral_constant<int,64>{});else f(std::integral_constant<int,128>{});
  CUDA_CHECK(cudaGetLastError());
}

template<int STAGES,int GROUP,int APAD=0,int BPAD=0,int WARP_M=32,int WARP_N=64>
KernelResources warp_epilogue_resources() {
  constexpr int threads=(64/WARP_M)*(128/WARP_N)*32;
  constexpr int warps=threads/32;
  constexpr int input=STAGES*(64*(32+APAD)*2+32*128/2)+32*(128+BPAD)*2;
  constexpr int output=warps*16*16*4;
  constexpr int smem=input>output?input:output;
  auto fn=kernel<STAGES,GROUP,APAD,BPAD,false,false,WARP_M,WARP_N,true>;
  cudaFuncAttributes a{};CUDA_CHECK(cudaFuncGetAttributes(&a,fn));int blocks=0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fn,threads,smem));
  return {a.numRegs,int(a.sharedSizeBytes),a.maxThreadsPerBlock,blocks,blocks*threads/1536.0f};
}

template<int STAGES, int APAD=0, int BPAD=0, int WARP_M=32, int WARP_N=64>
void launch(const half* a,const uint8_t* b,const half* s,half* c,int M,int N,int K,int group,cudaStream_t stream) {
  constexpr int threads=(64/WARP_M)*(128/WARP_N)*32;
  constexpr int input=STAGES*(64*(32+APAD)*2+32*128/2)+32*(128+BPAD)*2;
  constexpr int output=64*128*4;
  constexpr int smem=input>output?input:output;
  auto f=[&](auto g){constexpr int G=decltype(g)::value;auto fn=kernel<STAGES,G,APAD,BPAD,false,false,WARP_M,WARP_N>;fn<<<dim3(N/128,M/64),threads,smem,stream>>>(a,b,s,c,M,N,K);};
  if(group==32)f(std::integral_constant<int,32>{});else if(group==64)f(std::integral_constant<int,64>{});else f(std::integral_constant<int,128>{});
  CUDA_CHECK(cudaGetLastError());
}

// Fixed-target persistent comparison.  128 CTAs divide the 2048 output tiles
// exactly at 4096^3; each CTA processes 16 tiles with a grid-stride loop.
// Smaller correctness shapes clamp the grid to the available tile count.
template<int STAGES, int APAD=0, int BPAD=0>
void launch_persistent(const half* a,const uint8_t* b,const half* s,half* c,
                       int M,int N,int K,int group,cudaStream_t stream) {
  constexpr int input=STAGES*(64*(32+APAD)*2+32*128/2)+32*(128+BPAD)*2;
  constexpr int output=64*128*4;
  constexpr int smem=input>output?input:output;
  const int tiles=(M/64)*(N/128);
  const int blocks=std::min(tiles,128);
  auto f=[&](auto g){constexpr int G=decltype(g)::value;auto fn=kernel<STAGES,G,APAD,BPAD,false,true>;fn<<<blocks,128,smem,stream>>>(a,b,s,c,M,N,K);};
  if(group==32)f(std::integral_constant<int,32>{});else if(group==64)f(std::integral_constant<int,64>{});else f(std::integral_constant<int,128>{});
  CUDA_CHECK(cudaGetLastError());
}

template<int STAGES,int GROUP,int APAD=0,int BPAD=0,int WARP_M=32,int WARP_N=64>
KernelResources resources() {
  constexpr int threads=(64/WARP_M)*(128/WARP_N)*32;
  constexpr int input=STAGES*(64*(32+APAD)*2+32*128/2)+32*(128+BPAD)*2;
  constexpr int output=64*128*4;
  constexpr int smem=input>output?input:output;
  auto fn=kernel<STAGES,GROUP,APAD,BPAD,false,false,WARP_M,WARP_N>;cudaFuncAttributes a{};CUDA_CHECK(cudaFuncGetAttributes(&a,fn));int blocks=0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fn,threads,smem));
  return {a.numRegs,int(a.sharedSizeBytes),a.maxThreadsPerBlock,blocks,blocks*threads/1536.0f};
}

template<int STAGES,int GROUP,int APAD=0,int BPAD=0>
KernelResources persistent_resources() {
  constexpr int input=STAGES*(64*(32+APAD)*2+32*128/2)+32*(128+BPAD)*2;
  constexpr int output=64*128*4;
  constexpr int smem=input>output?input:output;
  auto fn=kernel<STAGES,GROUP,APAD,BPAD,false,true>;cudaFuncAttributes a{};CUDA_CHECK(cudaFuncGetAttributes(&a,fn));int blocks=0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fn,128,smem));
  return {a.numRegs,int(a.sharedSizeBytes),a.maxThreadsPerBlock,blocks,blocks*128/1536.0f};
}

#if !defined(INT4_GEMM_ENABLE_FP32_OUTPUT) || INT4_GEMM_ENABLE_FP32_OUTPUT
template<int STAGES, int APAD=0, int BPAD=0>
void launch_fp32(const half* a,const uint8_t* b,const half* s,float* c,int M,int N,int K,int group,cudaStream_t stream) {
  constexpr int input=STAGES*(64*(32+APAD)*2+32*128/2)+32*(128+BPAD)*2;
  constexpr int output=64*128*4;
  constexpr int smem=input>output?input:output;
  auto f=[&](auto g){constexpr int G=decltype(g)::value;auto fn=kernel<STAGES,G,APAD,BPAD,true>;fn<<<dim3(N/128,M/64),128,smem,stream>>>(a,b,s,c,M,N,K);};
  if(group==32)f(std::integral_constant<int,32>{});else if(group==64)f(std::integral_constant<int,64>{});else f(std::integral_constant<int,128>{});
  CUDA_CHECK(cudaGetLastError());
}
#endif
} // namespace int4gemm::async_tc
