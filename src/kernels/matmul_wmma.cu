#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
#include <algorithm>
#include "../../include/kernels/common.cuh"

#ifndef KERNEL_DTYPE
#define KERNEL_DTYPE __nv_bfloat16
#endif

using namespace nvcuda;
using namespace yadrakova::kernels::common;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int TILE_M = 128;
constexpr int TILE_N = 64;
constexpr int TILE_K = 32;

constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int THREADS_PER_BLOCK = WARPS_M * WARPS_N * 32;

constexpr int SHMEM_PAD = 8;
constexpr int AS_STRIDE = TILE_K + SHMEM_PAD;
constexpr int BS_STRIDE = TILE_N + SHMEM_PAD;

constexpr int STAGES = 2;

__device__ __forceinline__ void copy_16bytes_async(void* smem_ptr, const void* gmem_ptr, bool valid) {
#if __CUDA_ARCH__ >= 800
    uint32_t smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    int src_size = valid ? 16 : 0;
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
        :
        : "r"(smem_int_ptr), "l"(gmem_ptr), "r"(src_size)
    );
#else
    if (valid) {
        *reinterpret_cast<uint4*>(smem_ptr) = *reinterpret_cast<const uint4*>(gmem_ptr);
    } else {
        *reinterpret_cast<uint4*>(smem_ptr) = make_uint4(0, 0, 0, 0);
    }
#endif
}

template <typename HalfT>
__device__ __forceinline__ void wmma_matmul_tile_optimized(
    const HalfT* __restrict__ A,
    const HalfT* __restrict__ B,
    HalfT* __restrict__ C,
    int64_t M, int64_t N, int64_t K)
{
    constexpr int GROUP_SIZE_M = 8;
    const int grid_m = gridDim.y;
    const int grid_n = gridDim.x;

    const int pid = blockIdx.y * grid_n + blockIdx.x;
    const int num_pid_in_group = GROUP_SIZE_M * grid_n;
    const int group_id = pid / num_pid_in_group;

    const int first_pid_m = group_id * GROUP_SIZE_M;
    const int group_size_m = min(grid_m - first_pid_m, GROUP_SIZE_M);

    const int pid_in_group = pid % num_pid_in_group;
    const int final_block_m = first_pid_m + (pid_in_group % group_size_m);
    const int final_block_n = pid_in_group / group_size_m;

    const int64_t block_row = final_block_m * TILE_M;
    const int64_t block_col = final_block_n * TILE_N;

    union SharedStorage {
        struct {
            alignas(16) HalfT As[STAGES][TILE_M][AS_STRIDE]; // 2*128*40*2 = 20480 B
            alignas(16) HalfT Bs[STAGES][TILE_K][BS_STRIDE]; // 2*32 *72*2 =  9216 B
        } inputs;
        alignas(16) float Cs[TILE_M][TILE_N];                // 128*64*4    = 32768 B
    };                                                       // Total        = 62464 B < 65536 B

    __shared__ SharedStorage shmem;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    constexpr int WARP_M_TILES = (TILE_M / WARPS_M) / WMMA_M; // 128/4 /16 = 2
    constexpr int WARP_N_TILES = (TILE_N / WARPS_N) / WMMA_N; // 64 /2 /16 = 2

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WARP_M_TILES][WARP_N_TILES];

    #pragma unroll
    for (int i = 0; i < WARP_M_TILES; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_N_TILES; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    constexpr int VEC_ELEMS = 8;
    constexpr int A_VEC_TOTAL = (TILE_M * TILE_K) / VEC_ELEMS; // 128*32/8 = 512
    constexpr int B_VEC_TOTAL = (TILE_K * TILE_N) / VEC_ELEMS; //  32*64/8 = 256

    auto load_tile_async = [&](int stage, int64_t k_offset) {
        #pragma unroll
        for (int i = tid; i < A_VEC_TOTAL; i += THREADS_PER_BLOCK) {
            int elem_idx = i * VEC_ELEMS;
            int r = elem_idx / TILE_K;
            int c = elem_idx % TILE_K;
            int64_t gr = block_row + r;
            int64_t gc = k_offset + c;

            bool valid = (gr < M && gc < K);
            const HalfT* gmem_ptr = A + gr * K + gc;
            HalfT* smem_ptr = &shmem.inputs.As[stage][r][c];

            copy_16bytes_async(smem_ptr, gmem_ptr, valid);
        }

        #pragma unroll
        for (int i = tid; i < B_VEC_TOTAL; i += THREADS_PER_BLOCK) {
            int elem_idx = i * VEC_ELEMS;
            int r = elem_idx / TILE_N;
            int c = elem_idx % TILE_N;
            int64_t gr = k_offset + r;
            int64_t gc = block_col + c;

            bool valid = (gr < K && gc < N);
            const HalfT* gmem_ptr = B + gr * N + gc;
            HalfT* smem_ptr = &shmem.inputs.Bs[stage][r][c];

            copy_16bytes_async(smem_ptr, gmem_ptr, valid);
        }

#if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group;\n" ::);
#endif
    };

    int write_stage = 0;
    int read_stage = 0;

    load_tile_async(write_stage, 0);
    write_stage ^= 1;

    for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
        int64_t next_k = k0 + TILE_K;

        if (next_k < K) {
            load_tile_async(write_stage, next_k);
        }

#if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.wait_group 0;\n" ::);
#endif
        __syncthreads();

        #pragma unroll
        for (int k_step = 0; k_step < TILE_K; k_step += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, HalfT, wmma::row_major> a_frag[WARP_M_TILES];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, HalfT, wmma::row_major> b_frag[WARP_N_TILES];

            #pragma unroll
            for (int i = 0; i < WARP_M_TILES; ++i) {
                int warp_row = warp_m * (WARP_M_TILES * WMMA_M) + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], &shmem.inputs.As[read_stage][warp_row][k_step], AS_STRIDE);
            }

            #pragma unroll
            for (int j = 0; j < WARP_N_TILES; ++j) {
                int warp_col = warp_n * (WARP_N_TILES * WMMA_N) + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j], &shmem.inputs.Bs[read_stage][k_step][warp_col], BS_STRIDE);
            }

            #pragma unroll
            for (int i = 0; i < WARP_M_TILES; ++i) {
                #pragma unroll
                for (int j = 0; j < WARP_N_TILES; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        __syncthreads();

        read_stage ^= 1;
        write_stage ^= 1;
    }

    #pragma unroll
    for (int i = 0; i < WARP_M_TILES; ++i) {
        int warp_row = warp_m * (WARP_M_TILES * WMMA_M) + i * WMMA_M;
        #pragma unroll
        for (int j = 0; j < WARP_N_TILES; ++j) {
            int warp_col = warp_n * (WARP_N_TILES * WMMA_N) + j * WMMA_N;
            wmma::store_matrix_sync(&shmem.Cs[warp_row][warp_col], c_frag[i][j],
                                    TILE_N, wmma::mem_row_major);
        }
    }

    __syncthreads();

    constexpr int C_ELEMS = TILE_M * TILE_N;
    #pragma unroll
    for (int i = tid; i < C_ELEMS; i += THREADS_PER_BLOCK) {
        int r = i / TILE_N;
        int c = i % TILE_N;
        int64_t gr = block_row + r;
        int64_t gc = block_col + c;

        if (gr < M && gc < N) {
            C[gr * N + gc] = from_float<HalfT>(shmem.Cs[r][c]);
        }
    }
}

extern "C" __global__ void matmul_wmma_kernel(
    const KERNEL_DTYPE* A, const KERNEL_DTYPE* B, KERNEL_DTYPE* C,
    int64_t M, int64_t N, int64_t K)
{
    wmma_matmul_tile_optimized<KERNEL_DTYPE>(A, B, C, M, N, K);
}