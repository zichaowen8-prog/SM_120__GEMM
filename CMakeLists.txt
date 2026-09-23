cmake_minimum_required(VERSION 3.24)
project(int4_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo --use_fast_math -Xptxas=-v,-warn-spills")

find_package(CUDAToolkit REQUIRED)
option(INT4_GEMM_ENABLE_FP32_OUTPUT "Build the FP32-output correctness path" ON)

add_executable(int4_gemm
  src/main.cu
  src/quantize.cu
  src/repack.cu
  src/reference.cu
  kernels/gemm_v00_reference.cu
  kernels/gemm_v01_pairwise.cu
  kernels/gemm_v02_repack_simt.cu
  kernels/gemm_v03_tensorcore.cu
  kernels/gemm_v04_repack_tensorcore.cu
  kernels/gemm_v09_async.cu
  kernels/gemm_v15_tma.cu
  kernels/gemm_v16_epilogue.cu
  kernels/gemm_v16_register.cu
  kernels/gemm_v16_tma.cu
  kernels/gemm_v17_instruction.cu
  kernels/gemm_final.cu)

target_include_directories(int4_gemm PRIVATE include)
target_compile_definitions(int4_gemm PRIVATE INT4_GEMM_ENABLE_FP32_OUTPUT=$<BOOL:${INT4_GEMM_ENABLE_FP32_OUTPUT}>)
target_link_libraries(int4_gemm PRIVATE CUDA::cudart CUDA::cublas CUDA::cuda_driver)
set_target_properties(int4_gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
