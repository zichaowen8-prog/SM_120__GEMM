#include "kernels.cuh"
#include "async_tensorcore_impl.cuh"
namespace int4gemm {
void launch_v09_stage2(const half*a,const uint8_t*b,const half*s,half*c,int M,int N,int K,int g,cudaStream_t st){async_tc::launch<2>(a,b,s,c,M,N,K,g,st);}
void launch_v09_stage3(const half*a,const uint8_t*b,const half*s,half*c,int M,int N,int K,int g,cudaStream_t st){async_tc::launch<3>(a,b,s,c,M,N,K,g,st);}
void launch_v09_stage4(const half*a,const uint8_t*b,const half*s,half*c,int M,int N,int K,int g,cudaStream_t st){async_tc::launch<4>(a,b,s,c,M,N,K,g,st);}
KernelResources query_v09_resources(const std::string&v,int g){
#define PICK(S,G) async_tc::resources<S,G>()
 int s=v=="v09s2"?2:v=="v09s3"?3:4;
 if(s==2){if(g==32)return PICK(2,32);if(g==64)return PICK(2,64);return PICK(2,128);}
 if(s==3){if(g==32)return PICK(3,32);if(g==64)return PICK(3,64);return PICK(3,128);}
 if(g==32)return PICK(4,32);if(g==64)return PICK(4,64);return PICK(4,128);
#undef PICK
}
void launch_v10_apad(const half*a,const uint8_t*b,const half*s,half*c,int M,int N,int K,int g,cudaStream_t st){async_tc::launch<2,8,0>(a,b,s,c,M,N,K,g,st);}
void launch_v10_bpad(const half*a,const uint8_t*b,const half*s,half*c,int M,int N,int K,int g,cudaStream_t st){async_tc::launch<2,0,8>(a,b,s,c,M,N,K,g,st);}
void launch_v10_abpad(const half*a,const uint8_t*b,const half*s,half*c,int M,int N,int K,int g,cudaStream_t st){async_tc::launch<2,8,8>(a,b,s,c,M,N,K,g,st);}
KernelResources query_v10_resources(const std::string&v,int g){
#define PICK(A,B,G) async_tc::resources<2,G,A,B>()
 if(v=="v10a"){if(g==32)return PICK(8,0,32);if(g==64)return PICK(8,0,64);return PICK(8,0,128);}
 if(v=="v10b"){if(g==32)return PICK(0,8,32);if(g==64)return PICK(0,8,64);return PICK(0,8,128);}
 if(g==32)return PICK(8,8,32);if(g==64)return PICK(8,8,64);return PICK(8,8,128);
#undef PICK
}
void launch_v11_a8b16(const half*a,const uint8_t*b,const half*s,half*c,int M,int N,int K,int g,cudaStream_t st){async_tc::launch<2,8,16>(a,b,s,c,M,N,K,g,st);}
void launch_v11_a16b8(const half*a,const uint8_t*b,const half*s,half*c,int M,int N,int K,int g,cudaStream_t st){async_tc::launch<2,16,8>(a,b,s,c,M,N,K,g,st);}
void launch_v11_a16b16(const half*a,const uint8_t*b,const half*s,half*c,int M,int N,int K,int g,cudaStream_t st){async_tc::launch<2,16,16>(a,b,s,c,M,N,K,g,st);}
void launch_v11_a8b24(const half*a,const uint8_t*b,const half*s,half*c,int M,int N,int K,int g,cudaStream_t st){async_tc::launch<2,8,24>(a,b,s,c,M,N,K,g,st);}
KernelResources query_v11_resources(const std::string&v,int g){
#define PICK(A,B,G) async_tc::resources<2,G,A,B>()
#define GROUPS(A,B) if(g==32)return PICK(A,B,32);if(g==64)return PICK(A,B,64);return PICK(A,B,128)
 if(v=="v11a"){GROUPS(8,16);} if(v=="v11b"){GROUPS(16,8);} if(v=="v11c"){GROUPS(16,16);} GROUPS(8,24);
#undef GROUPS
#undef PICK
}
void launch_v12_a8b32(const half*a,const uint8_t*b,const half*s,half*c,int M,int N,int K,int g,cudaStream_t st){async_tc::launch<2,8,32>(a,b,s,c,M,N,K,g,st);}
void launch_v12_a8b40(const half*a,const uint8_t*b,const half*s,half*c,int M,int N,int K,int g,cudaStream_t st){async_tc::launch<2,8,40>(a,b,s,c,M,N,K,g,st);}
void launch_v12_a16b24(const half*a,const uint8_t*b,const half*s,half*c,int M,int N,int K,int g,cudaStream_t st){async_tc::launch<2,16,24>(a,b,s,c,M,N,K,g,st);}
KernelResources query_v12_resources(const std::string&v,int g){
#define PICK(A,B,G) async_tc::resources<2,G,A,B>()
#define GROUPS(A,B) if(g==32)return PICK(A,B,32);if(g==64)return PICK(A,B,64);return PICK(A,B,128)
 if(v=="v12a"){GROUPS(8,32);} if(v=="v12b"){GROUPS(8,40);} GROUPS(16,24);
#undef GROUPS
#undef PICK
}
void launch_v13_persistent(const half*a,const uint8_t*b,const half*s,half*c,int M,int N,int K,int g,cudaStream_t st){async_tc::launch_persistent<2,8,24>(a,b,s,c,M,N,K,g,st);}
KernelResources query_v13_resources(int g){
 if(g==32)return async_tc::persistent_resources<2,32,8,24>();
 if(g==64)return async_tc::persistent_resources<2,64,8,24>();
 return async_tc::persistent_resources<2,128,8,24>();
}
void launch_v14_warp64x32(const half*a,const uint8_t*b,const half*s,half*c,int M,int N,int K,int g,cudaStream_t st){async_tc::launch<2,8,24,64,32>(a,b,s,c,M,N,K,g,st);}
void launch_v14_warp64x64(const half*a,const uint8_t*b,const half*s,half*c,int M,int N,int K,int g,cudaStream_t st){async_tc::launch<2,8,24,64,64>(a,b,s,c,M,N,K,g,st);}
KernelResources query_v14_resources(const std::string&v,int g){
#define PICK(WM,WN,G) async_tc::resources<2,G,8,24,WM,WN>()
 if(v=="v14a"){if(g==32)return PICK(64,32,32);if(g==64)return PICK(64,32,64);return PICK(64,32,128);}
 if(g==32)return PICK(64,64,32);if(g==64)return PICK(64,64,64);return PICK(64,64,128);
#undef PICK
}
#if !defined(INT4_GEMM_ENABLE_FP32_OUTPUT) || INT4_GEMM_ENABLE_FP32_OUTPUT
void launch_final_fp32(const half*a,const uint8_t*b,const half*s,float*c,int M,int N,int K,int g,cudaStream_t st){async_tc::launch_fp32<2,8,24>(a,b,s,c,M,N,K,g,st);}
#endif
}
