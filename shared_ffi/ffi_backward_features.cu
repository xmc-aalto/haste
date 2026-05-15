// Copyright (c) 2025-2026, Aalto University, developed by Nasib Ullah
// All rights reserved.
//
// SPDX-License-Identifier: MIT


#include<torch/extension.h>
#include<cuda_runtime.h>
#include<cuda_fp16.h>
#include<cuda_bf16.h>
#include<mma.h>

using namespace nvcuda;

#define CEIL_DIV(A,B) (((A)+(B)-1)/(B))
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16



// ------------------- Type conversion helper --------------------------------------------------------------//
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(at::Half x) { return __half2float(static_cast<__half>(x)); }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
template<typename T> __device__ __forceinline__ T from_float(float x) { return static_cast<T>(x); }
template<> __device__ __forceinline__ __half from_float<__half>(float x) { return __float2half_rn(x); }

// add for bf16:
__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);   // CUDA bf16 scalar→float intrinsic
}
template<>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}



// ---------------- Stochastic rounding kernel (for bfloat16 is useful)-------------------------------//
//TODO: For bf16 and maybe fp16 return C matrix by SR(C_fp32) matrix.  


// ----------------- v1 kernels (when B and FANIN small) -----------------------------------------------//


template<typename T, const int BN,const int BL, const int BK, const int WN>
__global__ void group_shared_fanin_bf_v1(const int* location_ptr,
    const T* weights_ptr, const T* grad_output_ptr,
    float* feature_grad_ptr,const int L, const int B, const int H, const int FANIN, 
    const int GSIZE,const int BM, const int WM)
    
{
    // strategy=1
    // each block computes a partial result along FANIN,B,BL(out of L) and atomic write to output

    // offset along K dimension (this is row for A tile and column for B tile)
    //int k_offset = blockIdx.x * BL; 

    if(blockIdx.x * BL>=L) return;

    int tid = threadIdx.x;
    //int lane = tid & 31;
    int warp_idx = tid/32;
    int warp_m = warp_idx / WN;
    int warp_n = warp_idx % WN;

    extern __shared__ __align__(16) unsigned char smem[];
    int* location_s = reinterpret_cast<int*>(smem);
    T* A_s = reinterpret_cast<T*>(smem+FANIN*sizeof(int));
    T* B_s = reinterpret_cast<T*>(smem+FANIN*sizeof(int)+BM*BK*sizeof(T));
    float* C_s = reinterpret_cast<float*>(smem+FANIN*sizeof(int)+(BM*BK+BK*BN)*sizeof(T));

    int elemA = BM*BK;
    int elemB = BK*BN;
    int elemC = BM*BN;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c0;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, nvcuda::wmma::col_major> a0,a1;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, nvcuda::wmma::col_major> b0,b1;

    // Read locations
    for(int v = tid; v<FANIN/4; v+=blockDim.x){
        int h = v*4;
        int c = h % FANIN;
        const int4* src = reinterpret_cast<const int4*>(&location_ptr[blockIdx.x * FANIN+c]);
        int4* dst = reinterpret_cast<int4*>(&location_s[c]);
        *dst = *src;
    }


    int k_base = blockIdx.x * BL;
    int k_max = min(k_base+BK,L);
    int valid_k = k_max - k_base;
    bool full_tile = (valid_k==BK);
    T zero_T = from_float<T>(0.0f);

    // read A tile to shared memory
    if(full_tile){

        for(int v=tid; v<elemA/8;v+=blockDim.x){
            int h = v*8;
            int r = h / FANIN;
            int c = h % FANIN;
            const uint4* src = reinterpret_cast<const uint4*>(&weights_ptr[(k_base+r)*FANIN+c]);
            uint4* dst = reinterpret_cast<uint4*>(&A_s[r * FANIN + c]);
            *dst = *src;

        }
    }
    else{
        // Tail tile: scalar guarded loads
        for (int v = tid; v < elemA; v += blockDim.x) {
            int r = v / FANIN;  
            int c = v % FANIN;
            if (r < valid_k && c < FANIN) {
                T val = weights_ptr[(k_base+r) * FANIN + c];
                A_s[r * FANIN + c] = val;

            }
            else{
                A_s[r * FANIN + c] = zero_T;
            }
            
            }
        }

    __syncthreads();

    // loop over B/BN
    for(int bn=0;bn<B;bn+=BN)
    {
        int remaining_B = (B > bn) ? (B - bn) : 0;
        int valid_bRows = remaining_B > BN ? BN : remaining_B;
        nvcuda::wmma::fill_fragment(c0,0.0f);
        // initialize result shared memory to 0
        for(int v=tid;v<elemC/4;v+=blockDim.x){
            int h = v*4;
            int r = h / BN;
            int c = h % BN;
            *reinterpret_cast<float4*>(&C_s[r*BN+c]) = make_float4(0, 0, 0, 0);
        }

        // read B tile to shared memory (vectorized where possible, scalar otherwise)
        for (int v = tid; v < elemB / 8; v += blockDim.x) {
            int h = v * 8;
            int r = h / BK;   // row index in this BN-tile [0..BN-1]
            int c = h % BK;   // local K index [0..BK-1]
        
            bool all_valid_k = (c + 7 < valid_k);
            bool row_valid   = (r < valid_bRows);           // bn + r < B
        
            if (all_valid_k && row_valid) {
                int idx = (bn + r) * L + k_base + c;
                if ((idx & 7) == 0) {                       // 8 bf16 = 16B
                    const uint4* src =
                        reinterpret_cast<const uint4*>(&grad_output_ptr[idx]);
                    uint4* dst =
                        reinterpret_cast<uint4*>(&B_s[r * BK + c]);
                    *dst = *src;
                    continue;
                }
            }
        
            // Scalar fallback (+ tail handling)
            for (int i = 0; i < 8; ++i) {
                int kk = c + i;
                int gk = k_base + kk;
                int gb = bn + r;                           // global batch index
        
                if (kk < valid_k && r < valid_bRows && gb < B) {
                    T val = grad_output_ptr[gb * L + gk];
                    B_s[r * BK + kk] = val;
                } else {
                    B_s[r * BK + kk] = zero_T;
                }
            }
        }

        __syncthreads(); // sync barrier
        for(unsigned int ks=0;ks<BK;ks+=WMMA_K)
        {

        T* As_warp = A_s + 0*FANIN + warp_m*WMMA_M + ks*FANIN;
        nvcuda::wmma::load_matrix_sync(a0,As_warp,FANIN);
        T* Bs_warp = B_s + warp_n*WMMA_N*BK + ks;
        nvcuda::wmma::load_matrix_sync(b0,Bs_warp, BK);

        nvcuda::wmma::mma_sync(c0,a0,b0,c0);
        }

        // atomic storage to C (use location_s for selecting feature row)
        float* Cs_warp = C_s + warp_m*WMMA_M*BN + warp_n*WMMA_N;
        // every warp write in its 16x16 portion in FANINxBN block tile. need barrier after it
        nvcuda::wmma::store_matrix_sync(Cs_warp, c0, BN, nvcuda::wmma::mem_row_major);
        __syncthreads();

        //SMEM to GMEM (non vectorized version) 
        for (int v = tid; v < elemC; v += blockDim.x) {
            int r   = v / BN;
            int c   = v % BN;
            int row = location_s[r];
            int gb  = bn + c;                    // global batch index
        
            if (gb < B) {                        // guard tail tile
                float val = C_s[r * BN + c];
                atomicAdd(&feature_grad_ptr[row * B + gb], val);
            }
        }

        __syncthreads();
    }

}





// This kernel works but doesn't respect all shapes. add memory guards for all shapes
template<const int BN, const int BL, const int BK, const int WN>
__global__ void group_shared_fanin_bf_v1_fp32(const int* location_ptr,
    const float* weights_ptr, const float* grad_output_ptr,
    float* feature_grad_ptr,const int L, const int B, const int H, const int FANIN, 
    const int GSIZE,const int BM,const int WM)
    
{
    // strategy=1
    // each block computes a partial result along FANIN,B,BL(out of L) and atomic write to output

    // offset along K dimension (this is row for A tile and column for B tile)
    unsigned int k_offset = blockIdx.x * BL; 

    if(k_offset>=L) return;

    unsigned int tid = threadIdx.x;
    //unsigned int lane = tid & 31;
    unsigned int warp_idx = tid/32;
    unsigned int warp_m = warp_idx / WN;
    unsigned int warp_n = warp_idx % WN;

    extern __shared__ __align__(16) unsigned char smem[];
    int* location_s = reinterpret_cast<int*>(smem);
    __half* A_s = reinterpret_cast<__half*>(smem+FANIN*sizeof(int));
    __half* B_s = reinterpret_cast<__half*>(smem+FANIN*sizeof(int)+BM*BK*sizeof(__half));
    float* C_s = reinterpret_cast<float*>(smem+FANIN*sizeof(int)+(BM*BK+BK*BN)*sizeof(__half));

    unsigned int elemLoc = FANIN;
    unsigned int elemA = BM*BK;
    unsigned int elemB = BK*BN;
    unsigned int elemC = BM*BN;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c0;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::col_major> a0;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::col_major> b0;

    // Read locations
    for(unsigned int v = tid; v<elemLoc/4; v+=blockDim.x){
        unsigned int h = v*4;
        unsigned int r = h / FANIN;
        unsigned int c = h % FANIN;
        const int4* src = reinterpret_cast<const int4*>(&location_ptr[(blockIdx.x+r)*FANIN+c]);
        int4* dst = reinterpret_cast<int4*>(&location_s[r*FANIN+c]);
        *dst = *src;
    }

    unsigned int k_base = k_offset;
    unsigned int k_max = min(k_base+BK,L);
    unsigned int valid_k = k_max - k_base;
    bool full_tile = (valid_k==BK);

    // read A tile to shared memory
    if(full_tile){

        for(unsigned int v=tid; v<elemA/4;v+=blockDim.x){
            unsigned int h = v*4;
            unsigned int r = h / FANIN;
            unsigned int c = h % FANIN;
            const float4* src = reinterpret_cast<const float4*>(&weights_ptr[(k_base+r)*FANIN+c]);

            float4 f = *src;

            // Shared store: convert to 4 halves
            __half* dst = reinterpret_cast<__half*>(&A_s[r * FANIN + c]);
            dst[0] = __float2half_rn(f.x);
            dst[1] = __float2half_rn(f.y);
            dst[2] = __float2half_rn(f.z);
            dst[3] = __float2half_rn(f.w);
        }
    }
    else{
        // Tail tile: scalar guarded loads
        for (unsigned int v = tid; v < elemA; v += blockDim.x) {
            unsigned int r = v / FANIN;  
            unsigned int c = v % FANIN;

            __half val_h = __float2half_rn(0.0f);
            if (r < valid_k && c < FANIN) {
                float val = weights_ptr[(k_base+r) * FANIN + c];
                val_h = __float2half_rn(val);
            }

            A_s[r * FANIN + c] = val_h;
        }
    }

    __syncthreads();

    //loop over (B/BN)
    for(int bn=0;bn<B;bn+=BN)
    {

        unsigned int remaining_B = (B > bn) ? (B - bn) : 0;
        unsigned int valid_bRows = remaining_B > BN ? BN : remaining_B;

        nvcuda::wmma::fill_fragment(c0,0.0f);
        // initialize result shared memory to 0
        for(unsigned int v=tid;v<elemC/4;v+=blockDim.x){
            unsigned int h = v*4;
            unsigned int r = h / BN;
            unsigned int c = h % BN;
            *reinterpret_cast<float4*>(&C_s[r*BN+c]) = make_float4(0.0f,0.0f,0.0f,0.0f);
        }

        // read B tile to shared memory (vectorized where possible, scalar otherwise)
        for (unsigned int v = tid; v < elemB / 4; v += blockDim.x) {
            unsigned int h = v * 4;
            unsigned int r = h / BK;   // batch index in tile: 0..BN-1
            unsigned int c = h % BK;   // K index in tile: 0..BK-1, multiple of 4

            // We handle 4 K positions: k = c, c+1, c+2, c+3
            unsigned int gk0 = k_base + c;

            // Are all 4 ks inside the valid_k range?
            bool all_valid_k = (c + 3 < valid_k);
            bool row_valid   = (r < valid_bRows);   // bn + r < B

            if (all_valid_k && row_valid) {
                // Global base index in elements
                unsigned int idx = (r+bn) * L + gk0;

                // Check 16-byte alignment for float4
                if ((idx & 3u) == 0u) {
                    // Aligned vector load
                    const float4* src = reinterpret_cast<const float4*>(&grad_output_ptr[idx]);
                    float4 f = *src;

                    __half* dst = &B_s[r * BK + c];
                    dst[0] = __float2half_rn(f.x);
                    dst[1] = __float2half_rn(f.y);
                    dst[2] = __float2half_rn(f.z);
                    dst[3] = __float2half_rn(f.w);
                    continue;  // done with this quad
                }
            }

            // Fallback: scalar loads for this quad (handles tail, misalignment, etc.)
            for (int i = 0; i < 4; ++i) {
                unsigned int kk = c + i;           // local K in tile
                unsigned int gb = bn + r;          // global batch index
                __half val_h = __float2half_rn(0.0f);
            
                if (kk < valid_k && r < valid_bRows && gb < B) {
                    float val = grad_output_ptr[gb * L + (k_base + kk)];
                    val_h = __float2half_rn(val);
                }
            
                B_s[r * BK + kk] = val_h;
            }
        }

        __syncthreads(); // sync barrier

        for(unsigned int ks=0;ks<BK;ks+=WMMA_K)
        {

        __half* Bs_warp = B_s + warp_n*WMMA_N*BK + ks ;
        nvcuda::wmma::load_matrix_sync(b0,Bs_warp, BK);
        __half* As_warp = A_s + warp_m*WMMA_M + ks*FANIN;
        nvcuda::wmma::load_matrix_sync(a0,As_warp,FANIN);

        nvcuda::wmma::mma_sync(c0,a0,b0,c0);

        }

        // atomic storage to C (use location_s for selecting feature row)
        float* Cs_warp = C_s + warp_m*WMMA_M*BN + warp_n*WMMA_N;
        nvcuda::wmma::store_matrix_sync(Cs_warp, c0, BN, nvcuda::wmma::mem_row_major);
        __syncthreads();

        //SMEM to GMEM (non vectorized version) 
        for (unsigned int v = tid; v < elemC; v += blockDim.x) {
            unsigned int r   = v / BN;
            unsigned int c   = v % BN;
            unsigned int row = location_s[r];
            unsigned int gb  = bn + c;
        
            if (gb < B) {
                float val = C_s[r * BN + c];
                atomicAdd(&feature_grad_ptr[row * B + gb], val);
            }
        }
        __syncthreads();

    }


}



//--------------------------------- Interface CPP host function ---------------------------------//

torch::Tensor ffi_backward_features_cuda(torch::Tensor features, 
    torch::Tensor weights, torch::Tensor locations, torch::Tensor grad_outputs,
    const int B, const int H, const int FANIN, const int L,int gsize) 
    {

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(features.device());
    torch::Tensor features_grad = torch::zeros({H, B}, opts);

    switch(gsize)
    {
        case 32:
                {

            const int BM = FANIN;      // tile dimension along FANIN (M), currently uses full M (no split)
            const int BN = 64;      // tile dimension along BATCH (N), currently uses full N (no split)
            const int BL = 32;      // tile dimension along LABEL (L),chunksize is BL
            const int BK = 32;      // BK rows/cols are processed at once out of BL
            const int WM = BM / WMMA_M;       // number of warps along BM 
            const int WN = 4;       // number of warps along BN  
            const int THREADS = WM*WN*32; // total number of threads (WM*WN*32)
            const int GSIZE = 32;
            const int NGROUPS = 1;

            TORCH_CHECK(THREADS<=1024, "Too many threads to be launch. check your B and FANIN size");

            // this strategy uses many blocks as NGROUPS=1 so k splits chunk size = 32
            // easier to implement but inefficient: 1. more blocks require more shared memory, 2. more atomic operations

            dim3 blocks(THREADS,1,1);
            dim3 grids(CEIL_DIV(L,BL)); 

            TORCH_CHECK(BM == FANIN, "BM should be equal to FANIN");
            TORCH_CHECK(BK==BL,"BK should be equal to BL for this kernel");


            TORCH_CHECK(FANIN % WMMA_M == 0,"FANIN must be multiple of 16 for WMMA");

            if (features.dtype()== torch::kFloat16) {
                size_t smem_bytes = ( BM*BK + BN*BK) * sizeof(__half) + FANIN*sizeof(int) + BM*BN * sizeof(float);

                group_shared_fanin_bf_v1<__half,BN,BL,BK,WN><<< grids,blocks,smem_bytes>>>(
                locations.data_ptr<int>(),
                reinterpret_cast<const __half*>(weights.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(grad_outputs.data_ptr<at::Half>()),
                features_grad.data_ptr<float>(),
                L,B,H,FANIN,GSIZE, BM, WM);

            }
            else if(features.dtype()== torch::kFloat32){

                size_t smem_bytes = ( BM*BK + BN*BK) * sizeof(__half) + FANIN*sizeof(int) + BM*BN * sizeof(float);

                group_shared_fanin_bf_v1_fp32<BN,BL,BK,WN><<< grids,blocks,smem_bytes>>>(
                locations.data_ptr<int>(),weights.data_ptr<float>(),
                grad_outputs.data_ptr<float>(),features_grad.data_ptr<float>(),L,B,H,FANIN,GSIZE, BM, WM);


            }
            else if(features.dtype()== torch::kBFloat16)
            {
                size_t smem_bytes = ( BM*BK + BN*BK) * sizeof(__nv_bfloat16) + FANIN*sizeof(int) + BM*BN * sizeof(float);

                group_shared_fanin_bf_v1<__nv_bfloat16,BN,BL,BK,WN><<< grids,blocks,smem_bytes>>>(
                locations.data_ptr<int>(),
                reinterpret_cast<const __nv_bfloat16*>(weights.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(grad_outputs.data_ptr<at::BFloat16>()),
                features_grad.data_ptr<float>(),
                L,B,H,FANIN,GSIZE, BM, WM);
                
            }

            break;


        }

        case 16:
        {

            const int BM = FANIN;      // tile dimension along FANIN (M), currently uses full M (no split)
            const int BN = 64;      // tile dimension along BATCH (N), currently uses full N (no split)
            const int BL = 16;      // tile dimension along LABEL (L),chunksize is BL
            const int BK = 16;      // BK rows/cols are processed at once out of BL
            const int WM = BM / WMMA_M;       // number of warps along BM 
            const int WN = 4;       // number of warps along BN  
            const int THREADS = WM*WN*32; // total number of threads (WM*WN*32)
            const int GSIZE = 16;
            const int NGROUPS = 1;

            TORCH_CHECK(THREADS<=1024, "Too many threads to be launch. check your B and FANIN size");

            // this strategy uses many blocks as NGROUPS=1 so k splits chunk size = 32
            // easier to implement but inefficient: 1. more blocks require more shared memory, 2. more atomic operations

            dim3 blocks(THREADS,1,1);
            dim3 grids(CEIL_DIV(L,BL)); 

            TORCH_CHECK(BM == FANIN, "BM should be equal to FANIN");
            TORCH_CHECK(BK==BL,"BK should be equal to BL for this kernel");


            TORCH_CHECK(FANIN % WMMA_M == 0,"FANIN must be multiple of 16 for WMMA");

            if (features.dtype()== torch::kFloat16) {
            size_t smem_bytes = ( BM*BK + BN*BK) * sizeof(__half) + FANIN*sizeof(int) + BM*BN * sizeof(float);

            group_shared_fanin_bf_v1<__half,BN,BL,BK,WN><<< grids,blocks,smem_bytes>>>(
            locations.data_ptr<int>(),
            reinterpret_cast<const __half*>(weights.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(grad_outputs.data_ptr<at::Half>()),
            features_grad.data_ptr<float>(),
            L,B,H,FANIN,GSIZE, BM, WM);

            }
            else if(features.dtype()== torch::kFloat32){

            size_t smem_bytes = ( BM*BK + BN*BK) * sizeof(__half) + FANIN*sizeof(int) + BM*BN * sizeof(float);

            group_shared_fanin_bf_v1_fp32<BN,BL,BK,WN><<< grids,blocks,smem_bytes>>>(
            locations.data_ptr<int>(),weights.data_ptr<float>(),
            grad_outputs.data_ptr<float>(),features_grad.data_ptr<float>(),L,B,H,FANIN,GSIZE, BM, WM);


            }
            else if(features.dtype()== torch::kBFloat16)
            {
            size_t smem_bytes = ( BM*BK + BN*BK) * sizeof(__nv_bfloat16) + FANIN*sizeof(int) + BM*BN * sizeof(float);

            group_shared_fanin_bf_v1<__nv_bfloat16,BN,BL,BK,WN><<< grids,blocks,smem_bytes>>>(
            locations.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(weights.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(grad_outputs.data_ptr<at::BFloat16>()),
            features_grad.data_ptr<float>(),
            L,B,H,FANIN,GSIZE, BM, WM);

            }

            break;


        }

        case 64:
        {

            const int BM = FANIN;      // tile dimension along FANIN (M), currently uses full M (no split)
            const int BN = 64;      // tile dimension along BATCH (N), currently uses full N (no split)
            const int BL = 64;      // tile dimension along LABEL (L),chunksize is BL
            const int BK = 64;      // BK rows/cols are processed at once out of BL
            const int WM = BM / WMMA_M;       // number of warps along BM 
            const int WN = 4;       // number of warps along BN  
            const int THREADS = WM*WN*32; // total number of threads (WM*WN*32)
            const int GSIZE = 64;
            const int NGROUPS = 1;

            TORCH_CHECK(THREADS<=1024, "Too many threads to be launch. check your B and FANIN size");

            // this strategy uses many blocks as NGROUPS=1 so k splits chunk size = 32
            // easier to implement but inefficient: 1. more blocks require more shared memory, 2. more atomic operations

            dim3 blocks(THREADS,1,1);
            dim3 grids(CEIL_DIV(L,BL)); 

            TORCH_CHECK(BM == FANIN, "BM should be equal to FANIN");
            TORCH_CHECK(BK==BL,"BK should be equal to BL for this kernel");


            TORCH_CHECK(FANIN % WMMA_M == 0,"FANIN must be multiple of 16 for WMMA");

            if (features.dtype()== torch::kFloat16) {
            size_t smem_bytes = ( BM*BK + BN*BK) * sizeof(__half) + FANIN*sizeof(int) + BM*BN * sizeof(float);

            group_shared_fanin_bf_v1<__half,BN,BL,BK,WN><<< grids,blocks,smem_bytes>>>(
            locations.data_ptr<int>(),
            reinterpret_cast<const __half*>(weights.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(grad_outputs.data_ptr<at::Half>()),
            features_grad.data_ptr<float>(),
            L,B,H,FANIN,GSIZE, BM, WM);

            }
            else if(features.dtype()== torch::kFloat32){

            size_t smem_bytes = ( BM*BK + BN*BK) * sizeof(__half) + FANIN*sizeof(int) + BM*BN * sizeof(float);

            group_shared_fanin_bf_v1_fp32<BN,BL,BK,WN><<< grids,blocks,smem_bytes>>>(
            locations.data_ptr<int>(),weights.data_ptr<float>(),
            grad_outputs.data_ptr<float>(),features_grad.data_ptr<float>(),L,B,H,FANIN,GSIZE, BM, WM);


            }
            else if(features.dtype()== torch::kBFloat16)
            {
            size_t smem_bytes = ( BM*BK + BN*BK) * sizeof(__nv_bfloat16) + FANIN*sizeof(int) + BM*BN * sizeof(float);

            group_shared_fanin_bf_v1<__nv_bfloat16,BN,BL,BK,WN><<< grids,blocks,smem_bytes>>>(
            locations.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(weights.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(grad_outputs.data_ptr<at::BFloat16>()),
            features_grad.data_ptr<float>(),
            L,B,H,FANIN,GSIZE, BM, WM);

            }

            break;


        }

    default:
    {
        throw std::runtime_error("Invalid group size. Choose group size from 16,32,64");
    }

}

    return features_grad;
}