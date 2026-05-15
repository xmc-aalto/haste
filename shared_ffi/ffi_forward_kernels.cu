// Copyright (c) 2025-2026, Aalto University, developed by Nasib Ullah
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For half precision support
#include <cuda_bf16.h>
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

// add for bf16:
__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);   // CUDA bf16 scalar→float intrinsic
}
template<>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}



template<typename T>
__device__ __forceinline__
void dump_and_store_tile(
    const nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float>& cf,
    float* smDump, unsigned lane,
    T* C,                    // [M, N] output
    const T* bias,           // [N] or nullptr
    int M, int N,
    int N_eff,            // M = B (#rows), N = L (#cols)
    unsigned out_row_offset, unsigned out_col_offset,
    unsigned warp_m, unsigned warp_n,
    unsigned row_off, unsigned col_off)
{
    static_assert(sizeof(T) == 2, "This dump_and_store_tile expects 2-byte T (half/bfloat16).");

    // Dump 16x16 FP32 tile to shared (row-major, ld=16)
    nvcuda::wmma::store_matrix_sync(smDump, cf, 16, nvcuda::wmma::mem_row_major);

    const unsigned tileElems  = 16u * 16u;    // 256
    const unsigned chunkSize  = 8u;           // process 8 elements per chunk
    const unsigned tileChunks = tileElems / chunkSize;  // 32 chunks

    #pragma unroll
    for (unsigned idx = lane; idx < tileChunks; idx += 32) {
        unsigned h = idx * chunkSize;  // element index in tile [0..255]
        unsigned r = h / 16u;          // row in tile [0..15]
        unsigned c = h % 16u;          // col in tile [0,8]

        // Load 8 floats from smem: 2 x float4
        float4 f4a = *reinterpret_cast<const float4*>(&smDump[h]);
        float4 f4b = *reinterpret_cast<const float4*>(&smDump[h + 4]);

        float vals_f[8] = { f4a.x, f4a.y, f4a.z, f4a.w,
                            f4b.x, f4b.y, f4b.z, f4b.w };

        // Global coords for first element of this 8-wide chunk
        unsigned gr = out_row_offset + warp_m * WMMA_M + row_off + r;
        unsigned gc = out_col_offset + warp_n * WMMA_N + col_off + c;

        size_t idx_elem   = (size_t)gr * (size_t)N + (size_t)gc;  // index in T-elements
        bool   in_row      = gr < (unsigned)M;
        bool   full_in_col = (gc + (chunkSize - 1u)) < (unsigned)N;  // gc+7 < N
        bool   aligned16   = ((idx_elem & 7u) == 0u);  // 8 * sizeof(T)=16 bytes

        // Add bias (per column) in float
        if (bias != nullptr && in_row) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                unsigned gcj = gc + j;
                if (gcj < N_eff) {
                    vals_f[j] += to_float(bias[gcj]);
                }
            }
        }

        if (in_row && full_in_col && aligned16) {
            // Fast path: vectorized uint4 store
            T vals_t[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                vals_t[j] = from_float<T>(vals_f[j]);
            }

            uint4 packed = *reinterpret_cast<const uint4*>(vals_t);
            *reinterpret_cast<uint4*>(&C[idx_elem]) = packed;
        } else {
            // Tail / misaligned path: scalar stores with bounds checks
            if (gr < (unsigned)M) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    unsigned gcj = gc + j;
                    if (gcj < (unsigned)N) {
                        C[gr * N + gcj] = from_float<T>(vals_f[j]);
                    }
                }
            }
        }
    }
}




// ------------------------ Group shared Fan-in Kernel ---------------------------------------------------//

template<typename T,const int BM, const int BN, const int BK, const int WM, const int WN, const int NGROUPS>
__global__ void group_shared_fanin_v1(
    const int* location_ptr,const T* feature_ptr,
    const T* weights_ptr,const T* bias_ptr, T* output_ptr,
    const int L, const int B, const int H, const int FANIN, const int GSIZE,const int L_eff)
{

        // one output tile computes exactly one group in L dimension.
        unsigned int cRow_offset = blockIdx.y * BM;
        unsigned int cCol_offset = blockIdx.x * BN;

        if(cRow_offset>=B || cCol_offset>=L ) return;

        unsigned int tid = threadIdx.x;
        unsigned int lane = tid & 31;
        unsigned int warp_idx = tid / 32;
        unsigned int warp_m = warp_idx / WN;
        unsigned int warp_n  = warp_idx % WN;

        // block fanin location
        unsigned int group_idx = blockIdx.x;

        // shared memory
        extern __shared__ __align__(16) unsigned char smem[];
        int* location_s = reinterpret_cast<int*>(smem);
        T* feature_s = reinterpret_cast<T*>(smem+NGROUPS*FANIN*sizeof(int));
        T* weight_s = reinterpret_cast<T*>(smem+NGROUPS*FANIN*sizeof(int)+BM*FANIN*sizeof(T));
        float* smDump = reinterpret_cast<float*>(smem+NGROUPS*FANIN*sizeof(int)+(BM+BN)*FANIN*sizeof(T));
        float* smDump_warp = smDump + warp_idx * (WMMA_M * WMMA_N);
        // load location for NGROUPS

        unsigned int elemLoc = NGROUPS * FANIN;
        unsigned int elemB = BN * FANIN;
        unsigned int elemA = BM*FANIN;

        // How many columns in this BM-wide tile are actually valid in B?
        // B can be arbitrary (not multiple of BM), last tile may be partial.
        unsigned int remaining = (B - cRow_offset);
        unsigned int valid_cRows = remaining > 0 ?(remaining < BM ? remaining : BM) :0;
        T zero_T = from_float<T>(0.0f);

        // loading location for this tile
        for(unsigned int v=tid; v< elemLoc/4; v += blockDim.x){
            unsigned int h = v*4;
            unsigned int r = h / FANIN;
            unsigned int c = h % FANIN;
            const int4* src = reinterpret_cast<const int4*>(&location_ptr[(group_idx+r)*FANIN+c]);
            int4* dst = reinterpret_cast<int4*>(&location_s[r*FANIN+c]);
            *dst = *src;  // 16B (4 integers) transfer at once

        }


        // load B tile (BN,FANIN)
        for(unsigned int v=tid;v<elemB/8;v+=blockDim.x){
            unsigned int h = v*8;
            unsigned int r = h / FANIN;
            unsigned int c = h % FANIN;

            unsigned int col = cCol_offset + r;  // global output column index (0..L-1)
            uint4* dst = reinterpret_cast<uint4*>(&weight_s[r*FANIN+c]);

            if (col < L_eff) {
                const uint4* src = reinterpret_cast<const uint4*>(
                    &weights_ptr[col * FANIN + c]);
                *dst = *src;
            } 
            else {
                uint4 z = {0, 0, 0, 0};
                *dst = z;
            }

        }
        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c00;
        nvcuda::wmma::fill_fragment(c00,0.0f);
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, nvcuda::wmma::col_major> a0;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, nvcuda::wmma::col_major> b0;


        // load A tile (BM,FANIN) the feature shape is [H,B]
        // A tile shared memory is (FANIN,BM) laid out as [r * BM + c]
        // We vectorize as 8 x T (16 bytes) where possible, and fall back to scalar otherwise.
        for (unsigned int v = tid; v < elemA/8; v += blockDim.x) {
            unsigned int h = v * 8;   // element index in tile
            unsigned int r = h / BM;              // fanin index
            unsigned int c = h % BM;              // batch index in tile

            unsigned int row = location_s[r];          // which row in H
            unsigned int local_c = c;                      // 0..BM-1 within tile
            unsigned int global_col = cRow_offset + local_c;  // 0..B-1 in batch

            // starting index in T-elements
            unsigned int idx_half = row * B + global_col;

            bool all_valid = (local_c + 7) < valid_cRows;
            bool is_aligned = ((idx_half & 7) == 0);  // 8 * 2B = 16B alignment

            T* dst_base = &feature_s[r * BM + local_c];

            if (all_valid && is_aligned) {
                // Fast path: one 16B load + one 16B store
                const uint4* src = reinterpret_cast<const uint4*>(&feature_ptr[idx_half]);
                uint4* dst = reinterpret_cast<uint4*>(dst_base);
                *dst = *src;
            } else {
                // Slow path: scalar
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    unsigned int kk = local_c + i;           // local column in [0..BM-1]
                    unsigned int gc = global_col + i;        // global column in [0..B-1]

                    if (kk < valid_cRows && gc < B) {
                        dst_base[i] = feature_ptr[row * B + gc];
                    } else {
                        dst_base[i] = zero_T;
                    }
                }
            }
        }


        __syncthreads();


        // Inner loop over FANIN. unrolled for multiple of 32
        for(unsigned int k=0; k<FANIN; k+=WMMA_K){

            // load A fragment for each warp (total 4 warps in the block)
            T* As_warp = feature_s + warp_m * WMMA_M  + k * BM;
            T* Bs_warp =  weight_s + warp_n * WMMA_N * FANIN + k;

            nvcuda::wmma::load_matrix_sync(a0,As_warp,BM);
            nvcuda::wmma::load_matrix_sync(b0,Bs_warp,FANIN);
            nvcuda::wmma::mma_sync(c00,a0,b0,c00);

        }

  
        unsigned base_row = cRow_offset + warp_m * WMMA_M;
        unsigned base_col = cCol_offset + warp_n * WMMA_N;
        
        // optional coarse check
        if (base_row < (unsigned)B && base_col < (unsigned)L) {
            dump_and_store_tile<T>(
                c00,
                smDump_warp,
                lane,
                output_ptr,
                bias_ptr,
                B, L,
                L_eff,                 // M = B, N = L
                cRow_offset,
                cCol_offset,
                warp_m,
                warp_n,
                0, 0);
        }
   

}


template<const int BM, const int BN, const int BK, const int WM, const int WN, const int NGROUPS>
__global__ void group_shared_fanin_v1_fp32(
    const int* location_ptr,const float* feature_ptr,
    const float* weights_ptr,const float* bias_ptr, float* output_ptr,
    const int L, const int B, const int H, const int FANIN, const int GSIZE,const int L_eff)
{

        // one output tile computes exactly one group in L dimension.
        unsigned int cRow_offset = blockIdx.y * BM;
        unsigned int cCol_offset = blockIdx.x * BN;

        if(cRow_offset>=B || cCol_offset>=L ) return;

        unsigned int tid = threadIdx.x;
        unsigned int lane = tid & 31;
        unsigned int warp_idx = tid / 32;
        unsigned int warp_m = warp_idx / WN;
        unsigned int warp_n  = warp_idx % WN;

        // block fanin location
        unsigned int group_idx = blockIdx.x;

        // shared memory
        extern __shared__ __align__(16) unsigned char smem[];
        int* location_s = reinterpret_cast<int*>(smem);
        __half* feature_s = reinterpret_cast<__half*>(smem+NGROUPS*FANIN*sizeof(int));
        __half* weight_s = reinterpret_cast<__half*>(smem+NGROUPS*FANIN*sizeof(int)+BM*FANIN*sizeof(__half));
        float* smDump = reinterpret_cast<float*>(smem+NGROUPS*FANIN*sizeof(int)+(BM+BN)*FANIN*sizeof(__half));
        float* smDump_warp = smDump + warp_idx * (WMMA_M * WMMA_N);
        // load location for NGROUPS

        unsigned int elemLoc = NGROUPS * FANIN;
        unsigned int elemB = BN * FANIN;
        unsigned int elemA = BM*FANIN;
        // How many columns in this BM-wide tile are actually valid in B?
        // B can be arbitrary (not multiple of BM), last tile may be partial.
        unsigned int remaining = (B - cRow_offset);
        unsigned int valid_cRows = remaining > 0? (remaining < BM ? remaining : BM): 0;
        __half zero_half = __float2half_rn(0.0f);

        // loading location for this tile
        for(unsigned int v=tid; v< elemLoc/4; v += blockDim.x){
            unsigned int h = v*4;
            unsigned int r = h / FANIN;
            unsigned int c = h % FANIN;
            const int4* src = reinterpret_cast<const int4*>(&location_ptr[(group_idx+r)*FANIN+c]);
            int4* dst = reinterpret_cast<int4*>(&location_s[r*FANIN+c]);
            *dst = *src;  // 16B (4 integers) transfer at once

        }
        
        // load B tile (BN,FANIN)
        for (unsigned int v = tid; v < elemB / 4; v += blockDim.x) {
            unsigned int h = v * 4;
            unsigned int r = h / FANIN;  // 0..BN-1
            unsigned int c = h % FANIN;  // 0..FANIN-1
    
            unsigned int col = cCol_offset + r;  // global column in [0..L-1]
            __half* dst = reinterpret_cast<__half*>(&weight_s[r * FANIN + c]);
    
            if (col < L_eff) {
                const float4* src = reinterpret_cast<const float4*>(
                    &weights_ptr[col * FANIN + c]);  // 4 floats
                float4 f = *src;
                dst[0] = __float2half_rn(f.x);
                dst[1] = __float2half_rn(f.y);
                dst[2] = __float2half_rn(f.z);
                dst[3] = __float2half_rn(f.w);
            } else {
                // out-of-bounds L tile: zero
                dst[0] = __float2half_rn(0.0f);
                dst[1] = __float2half_rn(0.0f);
                dst[2] = __float2half_rn(0.0f);
                dst[3] = __float2half_rn(0.0f);
            }
        }

        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c00;
        nvcuda::wmma::fill_fragment(c00,0.0f);
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::col_major> a0;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::col_major> b0;

        // load A tile (BM,FANIN)the feature shape is [H,B]
        // A tile in shared: (FANIN, BM), stored as half: feature_s[r * BM + c]
        // Vectorize as 4 floats (float4 = 16B) when possible, otherwise scalar + zero-pad.
        for (unsigned int v = tid; v < elemA / 4; v += blockDim.x) {
            unsigned int h = v * 4;        // element index in tile
            unsigned int r = h / BM;       // fanin index
            unsigned int c = h % BM;       // batch index within tile

            unsigned int row = location_s[r];          // which row in H
            unsigned int local_c = c;                      // 0..BM-1
            unsigned int global_col = cRow_offset + local_c;  // 0..B-1

            // starting index in float elements
            unsigned int idx_float = row * B + global_col;

            bool all_valid = (local_c + 3) < valid_cRows;    // 4-wide chunk fits inside this tile
            bool aligned   = ((idx_float & 3) == 0);        // 4 floats = 16B alignment

            __half* dst = &feature_s[r * BM + local_c];

            if (all_valid && aligned) {
                // Fast path: one 16B load + 4 float->half converts
                const float4* src = reinterpret_cast<const float4*>(&feature_ptr[idx_float]);
                float4 f = *src;
                dst[0] = __float2half_rn(f.x);
                dst[1] = __float2half_rn(f.y);
                dst[2] = __float2half_rn(f.z);
                dst[3] = __float2half_rn(f.w);
            } else {
                // Slow path: scalar with per-element bounds checks and zero-padding
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    unsigned int kk = local_c + i;        // local col in tile
                    unsigned int gc = global_col + i;     // global col in B

                    float fval = 0.0f;
                    if (kk < valid_cRows && gc < B) {
                        fval = feature_ptr[row * B + gc];
                    }
                    dst[i] = __float2half_rn(fval);
                }
            }
        }


        __syncthreads();


        // Inner loop over FANIN. unrolled for multiple of 32
        for(unsigned int k=0; k<FANIN; k+=WMMA_K){

            // load A fragment for each warp (total 4 warps in the block)
            __half* As_warp = feature_s + warp_m * WMMA_M  + k * BM;
            __half* Bs_warp =  weight_s + warp_n * WMMA_N * FANIN + k;

            nvcuda::wmma::load_matrix_sync(a0,As_warp,BM);
            nvcuda::wmma::load_matrix_sync(b0,Bs_warp,FANIN);
            nvcuda::wmma::mma_sync(c00,a0,b0,c00);

        }

  
        unsigned base_row = cRow_offset + warp_m * WMMA_M;
        unsigned base_col = cCol_offset + warp_n * WMMA_N;
                
        if (base_row < (unsigned)B && base_col < (unsigned)L) {
            nvcuda::wmma::store_matrix_sync(
                smDump_warp, c00, 16, nvcuda::wmma::mem_row_major);
        
            const unsigned tileElems  = 16 * 16;  // 256 floats
            const unsigned chunkSize  = 4;        // 4 floats per chunk
            const unsigned tileChunks = tileElems / chunkSize;  // 64
        
            #pragma unroll
            for (unsigned idx = lane; idx < tileChunks; idx += 32) {
                unsigned h = idx * chunkSize;  // 0..252
                unsigned r = h / 16u;          // 0..15
                unsigned c = h % 16u;          // 0,4,8,12,...
        
                // load 4 floats from this warp's shared tile
                float4 f4 = *reinterpret_cast<const float4*>(&smDump_warp[h]);
        
                unsigned gr = base_row + r;
                unsigned gc = base_col + c;
        
                if (gr >= (unsigned)B) continue;
        
                // Add bias (per column) if provided
                if (bias_ptr != nullptr) {
                    if (gc + 0u < L_eff) f4.x += bias_ptr[gc + 0u];
                    if (gc + 1u < L_eff) f4.y += bias_ptr[gc + 1u];
                    if (gc + 2u < L_eff) f4.z += bias_ptr[gc + 2u];
                    if (gc + 3u < L_eff) f4.w += bias_ptr[gc + 3u];
                }
        
                size_t g_idx     = (size_t)gr * (size_t)L + (size_t)gc;
                bool   full_in_c = (gc + (chunkSize - 1u)) < (unsigned)L;
                bool   aligned16 = ((g_idx & 3u) == 0u); // 4 floats = 16B
        
                if (full_in_c && aligned16) {
                    *reinterpret_cast<float4*>(&output_ptr[g_idx]) = f4;
                } else {
                    // scalar tail / misaligned path
                    if (gc + 0u < (unsigned)L) output_ptr[gr * L + gc + 0u] = f4.x;
                    if (gc + 1u < (unsigned)L) output_ptr[gr * L + gc + 1u] = f4.y;
                    if (gc + 2u < (unsigned)L) output_ptr[gr * L + gc + 2u] = f4.z;
                    if (gc + 3u < (unsigned)L) output_ptr[gr * L + gc + 3u] = f4.w;
                }
            }
        }
   

}



//--------------------------------- Interface CPP host function ---------------------------------//


torch::Tensor ffi_forward_cuda(torch::Tensor locations,
    torch::Tensor features, torch::Tensor weights,c10::optional<torch::Tensor> bias_opt, const int B, const int H, 
    const int FANIN, const int L,int gsize) {

    int L_eff = L;          
    int align = 8;         
    int L_pad = ((L_eff + align - 1) / align) * align;
    
    // allocate padded output
    torch::Tensor output_full = torch::empty({B, L_pad}, features.options());


    auto dtype = features.dtype();

    const float*  bias_f32  = nullptr;
    const __half* bias_f16  = nullptr;
    const __nv_bfloat16* bias_bf16 = nullptr;

    if (bias_opt.has_value()) {
        auto bias = bias_opt.value();
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == L, "bias must be [L]");
        TORCH_CHECK(bias.dtype() == dtype, "bias dtype must match features dtype");

        if (dtype == torch::kFloat32) {
            bias_f32 = bias.data_ptr<float>();
        } else if (dtype == torch::kFloat16) {
            bias_f16 = reinterpret_cast<const __half*>(bias.data_ptr<at::Half>());
        } else if (dtype == torch::kBFloat16) {
            bias_bf16 = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>());
        }
    }

    switch (gsize){

            case 32:
            {

            const int BM = 64;      // batch dimension each block handles
            const int BN = 32;      // label dimension each blocks handles (depends on group size)
            const int BK = 32;      // used only if FANIN is large
            const int WM = 4;       // number of warps along BM 
            const int WN = 2;       // number of warps along BN  
            const int THREADS = 256; // total number of threads (WM*WN*32)
            const int NGROUPS = 1;
            const int GSIZE = 32;
            //auto gsize = L / locations.size(0);

            TORCH_CHECK(B > 0, "batch size B must be positive");
            TORCH_CHECK(FANIN % 16 == 0,"FANIN size should be multiple of 16 and >=16");
            TORCH_CHECK(FANIN<=128,"FANIN should be <=128 for this kernel to efficiently working");

            // calculate number of threads and blocks
            dim3 threads(THREADS,1,1);
            dim3 blocks(CEIL_DIV(L_pad,BN), CEIL_DIV(B,BM));
            // shared memory size: LOCATION for group + A tile + B tile + C tile (for storing)


            if (dtype == torch::kFloat32) {
                
                size_t shmem_bytes = NGROUPS*FANIN*sizeof(int) + BM * FANIN * sizeof(__half) + BN * FANIN * sizeof(__half) + BM*BN*sizeof(float); 
        
                group_shared_fanin_v1_fp32<BM,BN,BK,WM,WN,NGROUPS><<<blocks, threads, shmem_bytes>>>(locations.data_ptr<int>(),
                    features.data_ptr<float>(),
                    weights.data_ptr<float>(),
                    bias_f32,
                    output_full.data_ptr<float>(),L_pad, B, H, FANIN, GSIZE, L_eff);
            }
            else if(dtype == torch::kFloat16)
            {
                size_t shmem_bytes = NGROUPS*FANIN*sizeof(int) + BM * FANIN * sizeof(__half) + BN * FANIN * sizeof(__half) + BM*BN*sizeof(float); 
        
                group_shared_fanin_v1<__half,BM,BN,BK,WM,WN,NGROUPS><<<blocks, threads, shmem_bytes>>>(locations.data_ptr<int>(),
                    reinterpret_cast<const __half*>(features.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(weights.data_ptr<at::Half>()),
                    bias_f16,
                    reinterpret_cast<__half*>(output_full.data_ptr<at::Half>()),L_pad, B, H, FANIN, GSIZE,L_eff);

            }
            else if(dtype == torch::kBFloat16){

                size_t shmem_bytes = NGROUPS*FANIN*sizeof(int) + BM * FANIN * sizeof(__nv_bfloat16) + BN * FANIN * sizeof(__nv_bfloat16) + BM*BN*sizeof(float); 
        
                group_shared_fanin_v1<__nv_bfloat16,BM,BN,BK,WM,WN,NGROUPS><<<blocks, threads, shmem_bytes>>>(locations.data_ptr<int>(),
                    reinterpret_cast<const __nv_bfloat16*>(features.data_ptr<at::BFloat16>()),
                    reinterpret_cast<const __nv_bfloat16*>(weights.data_ptr<at::BFloat16>()),
                    bias_bf16,
                    reinterpret_cast<__nv_bfloat16*>(output_full.data_ptr<at::BFloat16>()),L_pad, B, H, FANIN, GSIZE,L_eff);

                }

                break;
            }

            case 16:
            {

            const int BM = 64;      // batch dimension each block handles
            const int BN = 16;      // label dimension each blocks handles (depends on group size)
            const int BK = 32;      // used only if FANIN is large
            const int WM = 4;       // number of warps along BM 
            const int WN = 1;       // number of warps along BN  
            const int THREADS = 128; // total number of threads (WM*WN*32)
            const int NGROUPS = 1;
            const int GSIZE = 16;
            //auto gsize = L / locations.size(0);

            TORCH_CHECK(B > 0, "batch size B must be positive");
            TORCH_CHECK(FANIN % 16 == 0,"FANIN size should be multiple of 16 and >=16");
            TORCH_CHECK(FANIN<=256,"FANIN should be <=128 for this kernel to efficiently working");

            // calculate number of threads and blocks
            dim3 threads(THREADS,1,1);
            dim3 blocks(CEIL_DIV(L_pad,BN), CEIL_DIV(B,BM));
            // shared memory size: LOCATION for group + A tile + B tile + C tile (for storing)


            if (dtype == torch::kFloat32) {
                
                size_t shmem_bytes = NGROUPS*FANIN*sizeof(int) + BM * FANIN * sizeof(__half) + BN * FANIN * sizeof(__half) + BM*BN*sizeof(float); 
        
                group_shared_fanin_v1_fp32<BM,BN,BK,WM,WN,NGROUPS><<<blocks, threads, shmem_bytes>>>(locations.data_ptr<int>(),
                    features.data_ptr<float>(),
                    weights.data_ptr<float>(),
                    bias_f32,
                    output_full.data_ptr<float>(),L_pad, B, H, FANIN, GSIZE, L_eff);
            }
            else if(dtype == torch::kFloat16)
            {
                size_t shmem_bytes = NGROUPS*FANIN*sizeof(int) + BM * FANIN * sizeof(__half) + BN * FANIN * sizeof(__half) + BM*BN*sizeof(float); 
        
                group_shared_fanin_v1<__half,BM,BN,BK,WM,WN,NGROUPS><<<blocks, threads, shmem_bytes>>>(locations.data_ptr<int>(),
                    reinterpret_cast<const __half*>(features.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(weights.data_ptr<at::Half>()),
                    bias_f16,
                    reinterpret_cast<__half*>(output_full.data_ptr<at::Half>()),L_pad, B, H, FANIN, GSIZE,L_eff);

            }
            else if(dtype == torch::kBFloat16){

                size_t shmem_bytes = NGROUPS*FANIN*sizeof(int) + BM * FANIN * sizeof(__nv_bfloat16) + BN * FANIN * sizeof(__nv_bfloat16) + BM*BN*sizeof(float); 
        
                group_shared_fanin_v1<__nv_bfloat16,BM,BN,BK,WM,WN,NGROUPS><<<blocks, threads, shmem_bytes>>>(locations.data_ptr<int>(),
                    reinterpret_cast<const __nv_bfloat16*>(features.data_ptr<at::BFloat16>()),
                    reinterpret_cast<const __nv_bfloat16*>(weights.data_ptr<at::BFloat16>()),
                    bias_bf16,
                    reinterpret_cast<__nv_bfloat16*>(output_full.data_ptr<at::BFloat16>()),L_pad, B, H, FANIN, GSIZE,L_eff);

                }

                break;
            }

            
            case 64:
            {

            const int BM = 64;      // batch dimension each block handles
            const int BN = 64;      // label dimension each blocks handles (depends on group size)
            const int BK = 32;      // used only if FANIN is large
            const int WM = 4;       // number of warps along BM 
            const int WN = 4;       // number of warps along BN  
            const int THREADS = 512; // total number of threads (WM*WN*32)
            const int NGROUPS = 1;
            const int GSIZE = 64;
            //auto gsize = L / locations.size(0);

            TORCH_CHECK(B > 0, "batch size B must be positive");
            TORCH_CHECK(FANIN % 16 == 0,"FANIN size should be multiple of 16 and >=16");
            TORCH_CHECK(FANIN<=128,"FANIN should be <=128 for this kernel to efficiently working");

            // calculate number of threads and blocks
            dim3 threads(THREADS,1,1);
            dim3 blocks(CEIL_DIV(L_pad,BN), CEIL_DIV(B,BM));
            // shared memory size: LOCATION for group + A tile + B tile + C tile (for storing)


            if (dtype == torch::kFloat32) {
                
                size_t shmem_bytes = NGROUPS*FANIN*sizeof(int) + BM * FANIN * sizeof(__half) + BN * FANIN * sizeof(__half) + BM*BN*sizeof(float); 
        
                group_shared_fanin_v1_fp32<BM,BN,BK,WM,WN,NGROUPS><<<blocks, threads, shmem_bytes>>>(locations.data_ptr<int>(),
                    features.data_ptr<float>(),
                    weights.data_ptr<float>(),
                    bias_f32,
                    output_full.data_ptr<float>(),L_pad, B, H, FANIN, GSIZE, L_eff);
            }
            else if(dtype == torch::kFloat16)
            {
                size_t shmem_bytes = NGROUPS*FANIN*sizeof(int) + BM * FANIN * sizeof(__half) + BN * FANIN * sizeof(__half) + BM*BN*sizeof(float); 
        
                group_shared_fanin_v1<__half,BM,BN,BK,WM,WN,NGROUPS><<<blocks, threads, shmem_bytes>>>(locations.data_ptr<int>(),
                    reinterpret_cast<const __half*>(features.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(weights.data_ptr<at::Half>()),
                    bias_f16,
                    reinterpret_cast<__half*>(output_full.data_ptr<at::Half>()),L_pad, B, H, FANIN, GSIZE,L_eff);

            }
            else if(dtype == torch::kBFloat16){

                size_t shmem_bytes = NGROUPS*FANIN*sizeof(int) + BM * FANIN * sizeof(__nv_bfloat16) + BN * FANIN * sizeof(__nv_bfloat16) + BM*BN*sizeof(float); 
        
                group_shared_fanin_v1<__nv_bfloat16,BM,BN,BK,WM,WN,NGROUPS><<<blocks, threads, shmem_bytes>>>(locations.data_ptr<int>(),
                    reinterpret_cast<const __nv_bfloat16*>(features.data_ptr<at::BFloat16>()),
                    reinterpret_cast<const __nv_bfloat16*>(weights.data_ptr<at::BFloat16>()),
                    bias_bf16,
                    reinterpret_cast<__nv_bfloat16*>(output_full.data_ptr<at::BFloat16>()),L_pad, B, H, FANIN, GSIZE,L_eff);

                }

                break;
            }





        default:
            {
                throw std::runtime_error("Unsupported group size for FFI kernel. choose from 16,32,64");
            }
    }

    // if no padding needed, this is basically a no-op
    torch::Tensor output = (L_pad == L_eff)? output_full: output_full.narrow(1,0,L_eff);


    return output;
}