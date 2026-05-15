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

// ------------------- Type conversion helper --------------------------------------------------------------//
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(at::Half x) { return __half2float(static_cast<__half>(x)); }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
template<typename T> __device__ __forceinline__ T from_float(float x) { return static_cast<T>(x); }
template<> __device__ __forceinline__ __half from_float<__half>(float x) { return __float2half_rn(x); }
template<typename T> __device__ __forceinline__ T zero_of() {    return from_float<T>(0.0f);}

// add for bf16:
__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);   // CUDA bf16 scalar→float intrinsic
}
template<>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}


// ---------------------------- RF -> SMEM --> GMEM kernels-----------------------------------------------------------------//
template<typename T>
__device__ __forceinline__
void dump_and_store_tile_grad(
    const nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float>& cf,
    float* smDump, unsigned lane,
    T* C, int M, int N,   // C is [M, N] = [L, FANIN]
    unsigned out_row_offset, unsigned out_col_offset,
    unsigned warp_m, unsigned warp_n,
    unsigned row_off, unsigned col_off)
{
    static_assert(sizeof(T) == 2, "Expected 2-byte type (half/bfloat16).");

    // Dump 16x16 tile to shared as float
    nvcuda::wmma::store_matrix_sync(smDump, cf, 16, nvcuda::wmma::mem_row_major);

    const unsigned tileElems  = 16 * 16;    // 256
    const unsigned chunkSize  = 8;           // 8 elements per chunk
    const unsigned tileChunks = tileElems / chunkSize;  // 32

    #pragma unroll
    for (unsigned idx = lane; idx < tileChunks; idx += 32) {
        unsigned h = idx * chunkSize;  // 0..248
        unsigned r = h / 16;          // 0..15
        unsigned c = h % 16;          // 0,8 (with chunkSize=8 we hit 0,8,...)

        // 8 floats from shared (2 x float4)
        float4 f4a = *reinterpret_cast<const float4*>(&smDump[h]);
        float4 f4b = *reinterpret_cast<const float4*>(&smDump[h + 4]);

        float vals_f[8] = {
            f4a.x, f4a.y, f4a.z, f4a.w,
            f4b.x, f4b.y, f4b.z, f4b.w
        };

        unsigned gr = out_row_offset + warp_m * WMMA_M + row_off + r;
        unsigned gc = out_col_offset + warp_n * WMMA_N + col_off + c;

        unsigned idx_elem   = gr * N + gc;  // index in T
        bool   in_row      = gr < M;
        bool   full_in_col = (gc + (chunkSize - 1)) < N;  // gc+7 < N
        bool   aligned16   = ((idx_elem & 7) == 0);  // 8 * 2B = 16B

        if (in_row && full_in_col && aligned16) {
            // Fast path: convert 8 floats to T and do a uint4 store
            T vals_t[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                vals_t[j] = from_float<T>(vals_f[j]);
            }
            uint4 packed = *reinterpret_cast<const uint4*>(vals_t);
            *reinterpret_cast<uint4*>(&C[idx_elem]) = packed;
        } else {
            // Tail / misaligned: scalar with guards
            if (gr < M) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    unsigned gcj = gc + j;
                    if (gcj < N) {
                        C[gr * N + gcj] = from_float<T>(vals_f[j]);
                    }
                }
            }
        }
    }
}




// ---------------------- one tile handle one group (BM=32 fixed)--------------------------------------------------

template<typename T, const int BM, const int BK, const int WM, const int NGROUPS>
__global__ void group_shared_fanin_bw_v1(
    const int* location_ptr,
    const T* feature_ptr,
    const T* grad_output_ptr,
    T* weight_grad_ptr,
    const int L, const int B, const int H,
    const int FANIN, const int GSIZE, const int BN, const int WN)
{

    unsigned int cRow_offset = blockIdx.x * BM;
    unsigned int cCol_offset = blockIdx.y * BN;  // 0

    unsigned int tid = threadIdx.x;
    unsigned int lane = tid & 31;
    unsigned int warp_idx = tid / 32;
    unsigned int warp_m = warp_idx / WN;
    unsigned int warp_n = warp_idx % WN;

    // if this block is completely out of range, early exit
    if (cRow_offset >= L) return;
    unsigned int valid_cRows = min(cRow_offset+BM,L) - cRow_offset;
    bool all_valid_rows = (valid_cRows==BM); 

    unsigned int group_idx = cRow_offset / GSIZE;
    T zero_T = zero_of<T>();

    extern __shared__ __align__(16) unsigned char smem[];
    int* location_s = reinterpret_cast<int*>(smem);
    T* grad_output_s = reinterpret_cast<T*>(smem + NGROUPS * FANIN * sizeof(int));
    T* feature_s = reinterpret_cast<T*>(smem + NGROUPS * FANIN * sizeof(int) + BM * BK * sizeof(T));
    float* smDump = reinterpret_cast<float*>(smem + NGROUPS * FANIN * sizeof(int) + (BM * BK + BK * BN) * sizeof(T));
    float* smDump_warp = smDump + warp_idx * (WMMA_M * WMMA_N);

    unsigned int elemLoc = NGROUPS * FANIN;
    unsigned int elemA = BM*BK;
    unsigned int elemB = BK*BN;

    // load locations
    for(unsigned int v = tid;v<elemLoc/4;v+= blockDim.x){
        unsigned int h = v*4;
        unsigned int r = h / FANIN;
        unsigned int c = h % FANIN;
        const int4* src = reinterpret_cast<const int4*>(&location_ptr[(group_idx+r)*FANIN+c]);
        int4* dst = reinterpret_cast<int4*>(&location_s[r*FANIN+c]);
        *dst = *src;
    }

    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c0;
    nvcuda::wmma::fill_fragment(c0,0.0f);
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, nvcuda::wmma::col_major> a0,a1;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, nvcuda::wmma::col_major> b0,b1;
       
    // inner loop over K dimension (batch here)
    for(unsigned ks=0; ks<B; ks+=BK)
    {
        unsigned int remaining_k = (B > ks) ? (B - ks) : 0;
        unsigned int valid_kRows = remaining_k > BK ? BK : remaining_k;
        
        // load A tile (need to take care of non aligned address)
        for (unsigned int v = tid; v < elemA / 8; v += blockDim.x) {
            unsigned int h = v * 8;
            unsigned int r = h / BM;   // 0..BK-1
            unsigned int c = h % BM;   // 0..BM-1
        
            unsigned int row_k = ks + r;  // batch index
            bool row_valid = row_k < B;
            bool cols_full = (c + 7) < valid_cRows;
            
            unsigned int idx_elem = row_k * L + (cRow_offset + c);
            bool is_aligned = ((idx_elem & 7u) == 0u);  // 8 * 2B = 16B
            
            T* dst_base = &grad_output_s[r * BM + c];
            
            if (row_valid && cols_full && is_aligned) {
                const uint4* src = reinterpret_cast<const uint4*>(&grad_output_ptr[idx_elem]);
                uint4* dst = reinterpret_cast<uint4*>(dst_base);
                *dst = *src;
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    unsigned int kk = c + i;
                    unsigned int gc = cRow_offset + kk;
            
                    if (row_valid && kk < valid_cRows) {
                        dst_base[i] = grad_output_ptr[row_k * L + gc];
                    } else {
                        dst_base[i] = zero_T;
                    }
                }
            }
        }

        // Load B tile
        for (unsigned int v = tid; v < elemB / 8; v += blockDim.x) {
            unsigned int h = v * 8;
            unsigned int r = h / BK;  // fanin index (0..BK-1)
            unsigned int c = h % BK;  // batch index within this K-tile
        
            unsigned int row = location_s[r];      // which feature row
            unsigned int col_k = ks + c;           // batch index
        
            bool cols_full = (c + 7) < valid_kRows;

            unsigned int idx_elem  = row * B + col_k;
            bool   aligned16 = ((idx_elem & 7) == 0);  // 8 * 2B = 16B
            
            if (cols_full && aligned16) {
                const uint4* src = reinterpret_cast<const uint4*>(&feature_ptr[idx_elem]);
                uint4* dst = reinterpret_cast<uint4*>(&feature_s[r * BK + c]);
                *dst = *src;
            } else {
                // scalar with guard on B, zero-pad
                T* dst = &feature_s[r * BK + c];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    unsigned int kk = c + i;
                    unsigned int gk = ks + kk;
            
                    if (kk < valid_kRows && gk < B) {
                        dst[i] = feature_ptr[row * B + gk];
                    } else {
                        dst[i] = zero_T;
                    }
                }
            }
        }

        __syncthreads();

        // using a bit of ILP (BK==32 for this is work)

        T* As_warp = grad_output_s + (warp_m * WMMA_M);
        T* As_warp2 = grad_output_s + (warp_m * WMMA_M) + 16 * BM;
        nvcuda::wmma::load_matrix_sync(a0, As_warp, BM);
        nvcuda::wmma::load_matrix_sync(a1, As_warp2, BM);

        T* Bs_warp = feature_s + (warp_n * WMMA_N) * BK;
        T* Bs_warp2 = feature_s + (warp_n * WMMA_N) * BK + 16;
        nvcuda::wmma::load_matrix_sync(b0, Bs_warp, BK);
        nvcuda::wmma::load_matrix_sync(b1, Bs_warp2, BK);
        nvcuda::wmma::mma_sync(c0, a0, b0, c0);
        nvcuda::wmma::mma_sync(c0, a1, b1, c0);

        __syncthreads();

    }

    unsigned base_row = cRow_offset + warp_m * WMMA_M;
    unsigned base_col = cCol_offset + warp_n * WMMA_N;
    
    if (base_row < L && base_col < FANIN) {
        dump_and_store_tile_grad<T>(c0,smDump_warp,lane,weight_grad_ptr,
            L,FANIN,cRow_offset,cCol_offset,warp_m,warp_n,0, 0);
    }

}



template<const int BM, const int BK, const int WM, const int NGROUPS>
__global__ void group_shared_fanin_bw_v1_fp32(const int* location_ptr,
    const float* feature_ptr, const float* grad_output_ptr,
    float* weight_grad_ptr,const int L, const int B, const int H,
     const int FANIN, const int GSIZE, const int BN, const int WN)
    
{
    unsigned int cRow_offset = blockIdx.x * BM;
    unsigned int cCol_offset = blockIdx.y * BN;  // 0

    unsigned int tid = threadIdx.x;
    unsigned int lane = tid & 31;
    unsigned int warp_idx = tid / 32;
    unsigned int warp_m = warp_idx / WN;
    unsigned int warp_n = warp_idx % WN;

    // if this block is completely out of range, early exit
    if (cRow_offset >= L) return;
    unsigned int valid_cRows = min(cRow_offset+BM,L) - cRow_offset;
    bool all_valid_rows = (valid_cRows==BM);  // (still unused, but kept for style)

    unsigned int group_idx = cRow_offset / GSIZE;
    __half zero_half = __float2half_rn(0.0f);

    extern __shared__ __align__(16) unsigned char smem[];
    int* location_s = reinterpret_cast<int*>(smem);
    __half* grad_output_s = reinterpret_cast<__half*>(smem+NGROUPS*FANIN*sizeof(int));
    __half* feature_s = reinterpret_cast<__half*>(smem+NGROUPS*FANIN*sizeof(int)+BM*BK*sizeof(__half));
    float* smDump = reinterpret_cast<float*>(smem+NGROUPS*FANIN*sizeof(int)+(BM*BK+BK*BN)*sizeof(__half));
    float* smDump_warp = smDump + warp_idx * (WMMA_M*WMMA_N); 

    unsigned int elemLoc = NGROUPS * FANIN;
    unsigned int elemA = BM*BK;
    unsigned int elemB = BK*BN;

    // load locations
    for(unsigned int v = tid;v<elemLoc/4;v+= blockDim.x){
        unsigned int h = v*4;
        unsigned int r = h / FANIN;
        unsigned int c = h % FANIN;
        const int4* src = reinterpret_cast<const int4*>(&location_ptr[(group_idx+r)*FANIN+c]);
        int4* dst = reinterpret_cast<int4*>(&location_s[r*FANIN+c]);
        *dst = *src;
    }

    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c0;
    nvcuda::wmma::fill_fragment(c0,0.0f);
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::col_major> a0,a1;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::col_major> b0,b1;

       
    // inner loop over K dimension (batch here)
    for(unsigned ks=0; ks< B; ks+=BK)
    {
        // NEW: how many K-rows (batch) are valid in this BK tile
        unsigned int remaining_k = (B > ks) ? (B - ks) : 0;
        unsigned int valid_kRows = remaining_k > BK ? BK : remaining_k;

        // load A tile (take care of non aligned locations when L is of any shape)
        for(unsigned int v = tid; v<elemA/4;v+= blockDim.x){
            unsigned int h = v*4;
            unsigned int r = h / BM;
            unsigned int c = h % BM;

            unsigned int row_k    = ks + r;              // batch index
            bool         row_valid = row_k < B;
            bool         all_valid = row_valid && (c+3 < valid_cRows);

            unsigned int idx_float = row_k * L + (cRow_offset + c);
            bool   is_aligned = ((idx_float & 3) == 0);  // 4 floats = 16B

            if(all_valid && is_aligned)
            {
                const float4* src = reinterpret_cast<const float4*>(&grad_output_ptr[idx_float]);
                float4 f = *src;

                // Shared store: convert to 4 halves
                __half* dst = reinterpret_cast<__half*>(&grad_output_s[r * BM + c]);
                dst[0] = __float2half_rn(f.x);
                dst[1] = __float2half_rn(f.y);
                dst[2] = __float2half_rn(f.z);
                dst[3] = __float2half_rn(f.w);

            }
            else{

                for (int i = 0; i < 4; ++i) {
                    int kk = c + i;      
                    
                    float val = 0.0f;
                    if (row_valid && kk < (int)valid_cRows) {
                        unsigned int gc = cRow_offset + kk;
                        val = grad_output_ptr[row_k * L + gc];
                    }
                    grad_output_s[r * BM + kk] = __float2half_rn(val);
                }
            }
        }

        // Load B tile (no longer assuming B is multiple of 64)
        for(unsigned int v=tid;v<elemB/4;v+= blockDim.x){
            unsigned int h = v*4;
            unsigned int r = h / BK;  // fanin index inside this group
            unsigned int c = h % BK;  // local K index in tile

            unsigned int row   = location_s[r]; // which specific feature row in H
            unsigned int col_k = ks + c;        // batch index

            bool cols_full = (c + 3) < valid_kRows;

            unsigned int idx_float = row * B + col_k;
            bool aligned16 = ((idx_float & 3) == 0); // 4 floats = 16B

            __half* dst = reinterpret_cast<__half*>(&feature_s[r * BK + c]);

            if (cols_full && aligned16) {
                const float4* src = reinterpret_cast<const float4*>(&feature_ptr[idx_float]);
                float4 f = *src;

                dst[0] = __float2half_rn(f.x);
                dst[1] = __float2half_rn(f.y);
                dst[2] = __float2half_rn(f.z);
                dst[3] = __float2half_rn(f.w);
            } else {
                // scalar tail: guard on B and zero-pad
                for (int i = 0; i < 4; ++i) {
                    unsigned int kk = c + i;
                    unsigned int gk = ks + kk;

                    float val = 0.0f;
                    if (kk < valid_kRows && gk < B) {
                        val = feature_ptr[row * B + gk];
                    }
                    dst[i] = __float2half_rn(val);
                }
            }
        }

        __syncthreads();

        __half* As_warp = grad_output_s + (warp_m * WMMA_M);
        __half* As_warp2 = grad_output_s + (warp_m * WMMA_M) + 16 * BM;
        nvcuda::wmma::load_matrix_sync(a0, As_warp, BM);
        nvcuda::wmma::load_matrix_sync(a1, As_warp2, BM);


        __half* Bs_warp = feature_s + (warp_n * WMMA_N) * BK;
        __half* Bs_warp2 = feature_s + (warp_n * WMMA_N) * BK + 16;
        nvcuda::wmma::load_matrix_sync(b0, Bs_warp, BK);
        nvcuda::wmma::load_matrix_sync(b1, Bs_warp2, BK);


        nvcuda::wmma::mma_sync(c0, a0, b0, c0);
        nvcuda::wmma::mma_sync(c0, a1, b1, c0);

        __syncthreads();

    }

    unsigned base_row = cRow_offset + warp_m * WMMA_M;
    unsigned base_col = cCol_offset + warp_n * WMMA_N;

    if (base_row < L && base_col < FANIN) {

        nvcuda::wmma::store_matrix_sync(smDump_warp, c0, 16, nvcuda::wmma::mem_row_major);

        // We’ll process 4 floats at a time (float4)
        const unsigned tileElems  = 16 * 16;  // 256
        const unsigned chunkSize  = 4;
        const unsigned tileChunks = tileElems / chunkSize; // 64 chunks
    
        #pragma unroll
        for (unsigned idx = lane; idx < tileChunks; idx += 32) {
            unsigned h = idx * chunkSize;  // element index in tile
            unsigned r = h / 16;          // 0..15
            unsigned c = h % 16;          // 0,4,8,12,...
    
            // Load 4 floats from shared
            float4 f4 = *reinterpret_cast<const float4*>(&smDump_warp[h]);
    
            unsigned gr = base_row + r;
            unsigned gc = base_col + c;
    
            // global index in float elements
            unsigned g_idx = gr * FANIN + gc;
    
            bool in_row      = (gr < L);
            bool full_in_col = (gc + (chunkSize - 1)) < FANIN;  // gc+3 < N
            bool aligned16   = ((g_idx & 3) == 0); // 4 floats = 16B
    
            if (in_row && full_in_col && aligned16) {
                // fast path: vectorized float4 store
                *reinterpret_cast<float4*>(&weight_grad_ptr[g_idx]) = f4;
            } else {
                // tail / misaligned path: scalar stores with guards
                if (gr < L) {
                    if (gc + 0 < FANIN) weight_grad_ptr[gr * FANIN + gc + 0u] = f4.x;
                    if (gc + 1 < FANIN) weight_grad_ptr[gr * FANIN + gc + 1u] = f4.y;
                    if (gc + 2 < FANIN) weight_grad_ptr[gr * FANIN + gc + 2u] = f4.z;
                    if (gc + 3 < FANIN) weight_grad_ptr[gr * FANIN + gc + 3u] = f4.w;
                }
            }
        }
    }

}



//--------------------------------- Interface CPP host function ---------------------------------//


torch::Tensor ffi_backward_weights_cuda(torch::Tensor features, 
    torch::Tensor weights, torch::Tensor locations, torch::Tensor grad_outputs,
    const int B, const int H, const int FANIN, const int L,int gsize) 
    {


    torch::Tensor weights_grad = torch::empty({L, FANIN}, features.options());

    switch(gsize)
    {

        case 32:
        {

            const int BM = 32;      // label dimension each block handles
            const int BN = FANIN;      // label dimension each blocks handles
            const int BK = 32;      // used only if FANIN is large
            const int WM = 2;       // number of warps along BM 
            const int WN = BN/WMMA_N;       // number of warps along BN  
            const int THREADS = WM*WN*32; // total number of threads (WM*WN*32)
            const int GSIZE = gsize;
            const int NGROUPS = 1;

            dim3 blocks(THREADS,1,1);
            // gridDim.x limit 2 billion, gridDim.y, gridDim.z limits 65535 
            // so choose x dimension for L 
            dim3 grids(CEIL_DIV(L,BM),CEIL_DIV(FANIN,BN)); 

            TORCH_CHECK(FANIN % 16 == 0, "FANIN size should be multiple of 16");
            TORCH_CHECK(FANIN<=128,"FANIN should be <=128 for this kernel to efficiently working");


            if (features.dtype()== torch::kFloat16) 
            {
            size_t smem_bytes = ( BM*BK + BN*BK) * sizeof(__half) + NGROUPS*FANIN*sizeof(int) + BM*BN * sizeof(float);

            group_shared_fanin_bw_v1<__half,BM,BK,WM,NGROUPS><<< grids,blocks,smem_bytes>>>(
            locations.data_ptr<int>(),
            reinterpret_cast<const __half*>(features.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(grad_outputs.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(weights_grad.data_ptr<at::Half>()),
            L,B,H,FANIN,GSIZE, BN, WN);

            }
            else if(features.dtype()== torch::kBFloat16){

            size_t smem_bytes = ( BM*BK + BN*BK) * sizeof(__nv_bfloat16) + NGROUPS*FANIN*sizeof(int) + BM*BN * sizeof(float);

            group_shared_fanin_bw_v1<__nv_bfloat16,BM,BK,WM,NGROUPS><<< grids,blocks,smem_bytes>>>(
            locations.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(features.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(grad_outputs.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(weights_grad.data_ptr<at::BFloat16>()),
            L,B,H,FANIN,GSIZE, BN, WN);

            }

            else if(features.dtype()== torch::kFloat32){

            size_t smem_bytes = ( BM*BK + BN*BK) * sizeof(__half) + NGROUPS*FANIN*sizeof(int) + BM*BN * sizeof(float);

            group_shared_fanin_bw_v1_fp32<BM,BK,WM,NGROUPS><<< grids,blocks,smem_bytes>>>(
            locations.data_ptr<int>(),features.data_ptr<float>(),
            grad_outputs.data_ptr<float>(),weights_grad.data_ptr<float>(),
            L,B,H,FANIN,GSIZE, BN, WN);
            
            }

            break;
        }


        case 16:
        {

            const int BM = 16;      // label dimension each block handles
            const int BN = FANIN;      // label dimension each blocks handles
            const int BK = 32;      // used only if FANIN is large
            const int WM = 1;       // number of warps along BM 
            const int WN = BN/WMMA_N;       // number of warps along BN  
            const int THREADS = WM*WN*32; // total number of threads (WM*WN*32)
            const int GSIZE = gsize;
            const int NGROUPS = 1;

            dim3 blocks(THREADS,1,1);
            // gridDim.x limit 2 billion, gridDim.y, gridDim.z limits 65535 
            // so choose x dimension for L 
            dim3 grids(CEIL_DIV(L,BM),CEIL_DIV(FANIN,BN)); 

            TORCH_CHECK(FANIN % 16 == 0, "FANIN size should be multiple of 16");
            TORCH_CHECK(FANIN<=256,"FANIN should be <=128 for this kernel to efficiently working");


            if (features.dtype()== torch::kFloat16) 
            {
            size_t smem_bytes = ( BM*BK + BN*BK) * sizeof(__half) + NGROUPS*FANIN*sizeof(int) + BM*BN * sizeof(float);

            group_shared_fanin_bw_v1<__half,BM,BK,WM,NGROUPS><<< grids,blocks,smem_bytes>>>(
            locations.data_ptr<int>(),
            reinterpret_cast<const __half*>(features.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(grad_outputs.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(weights_grad.data_ptr<at::Half>()),
            L,B,H,FANIN,GSIZE, BN, WN);

            }
            else if(features.dtype()== torch::kBFloat16){

            size_t smem_bytes = ( BM*BK + BN*BK) * sizeof(__nv_bfloat16) + NGROUPS*FANIN*sizeof(int) + BM*BN * sizeof(float);

            group_shared_fanin_bw_v1<__nv_bfloat16,BM,BK,WM,NGROUPS><<< grids,blocks,smem_bytes>>>(
            locations.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(features.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(grad_outputs.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(weights_grad.data_ptr<at::BFloat16>()),
            L,B,H,FANIN,GSIZE, BN, WN);

            }

            else if(features.dtype()== torch::kFloat32){

            size_t smem_bytes = ( BM*BK + BN*BK) * sizeof(__half) + NGROUPS*FANIN*sizeof(int) + BM*BN * sizeof(float);

            group_shared_fanin_bw_v1_fp32<BM,BK,WM,NGROUPS><<< grids,blocks,smem_bytes>>>(
            locations.data_ptr<int>(),features.data_ptr<float>(),
            grad_outputs.data_ptr<float>(),weights_grad.data_ptr<float>(),
            L,B,H,FANIN,GSIZE, BN, WN);
            
            }

            break;
        }



        case 64:
        {

            const int BM = 64;      // label dimension each block handles
            const int BN = FANIN;      // label dimension each blocks handles
            const int BK = 32;      // used only if FANIN is large
            const int WM = 4;       // number of warps along BM 
            const int WN = BN/WMMA_N;       // number of warps along BN  
            const int THREADS = WM*WN*32; // total number of threads (WM*WN*32)
            const int GSIZE = gsize;
            const int NGROUPS = 1;

            dim3 blocks(THREADS,1,1);
            // gridDim.x limit 2 billion, gridDim.y, gridDim.z limits 65535 
            // so choose x dimension for L 
            dim3 grids(CEIL_DIV(L,BM),CEIL_DIV(FANIN,BN)); 

            TORCH_CHECK(FANIN % 16 == 0, "FANIN size should be multiple of 16");
            TORCH_CHECK(FANIN<=128,"FANIN should be <=128 for this kernel to efficiently working");


            if (features.dtype()== torch::kFloat16) 
            {
            size_t smem_bytes = ( BM*BK + BN*BK) * sizeof(__half) + NGROUPS*FANIN*sizeof(int) + BM*BN * sizeof(float);

            group_shared_fanin_bw_v1<__half,BM,BK,WM,NGROUPS><<< grids,blocks,smem_bytes>>>(
            locations.data_ptr<int>(),
            reinterpret_cast<const __half*>(features.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(grad_outputs.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(weights_grad.data_ptr<at::Half>()),
            L,B,H,FANIN,GSIZE, BN, WN);

            }
            else if(features.dtype()== torch::kBFloat16){

            size_t smem_bytes = ( BM*BK + BN*BK) * sizeof(__nv_bfloat16) + NGROUPS*FANIN*sizeof(int) + BM*BN * sizeof(float);

            group_shared_fanin_bw_v1<__nv_bfloat16,BM,BK,WM,NGROUPS><<< grids,blocks,smem_bytes>>>(
            locations.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(features.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(grad_outputs.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(weights_grad.data_ptr<at::BFloat16>()),
            L,B,H,FANIN,GSIZE, BN, WN);

            }

            else if(features.dtype()== torch::kFloat32){

            size_t smem_bytes = ( BM*BK + BN*BK) * sizeof(__half) + NGROUPS*FANIN*sizeof(int) + BM*BN * sizeof(float);

            group_shared_fanin_bw_v1_fp32<BM,BK,WM,NGROUPS><<< grids,blocks,smem_bytes>>>(
            locations.data_ptr<int>(),features.data_ptr<float>(),
            grad_outputs.data_ptr<float>(),weights_grad.data_ptr<float>(),
            L,B,H,FANIN,GSIZE, BN, WN);
            
            }

            break;
        }


    }

    return weights_grad;
}






