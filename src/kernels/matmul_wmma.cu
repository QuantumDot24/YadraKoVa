#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
#include "../../include/kernels/common.cuh"

#ifndef KERNEL_DTYPE
#define KERNEL_DTYPE __nv_bfloat16
#endif

using namespace nvcuda;
using namespace yadrakova::kernels::common;

// Configuración WMMA (Ampere / SM 8.6)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Block Tiling
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 32;

// 128 hilos = 4 warps
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int THREADS_PER_BLOCK = WARPS_M * WARPS_N * 32; // 128

// Padding para eliminar Bank Conflicts en Shared Memory
constexpr int SHMEM_PAD = 8;
constexpr int AS_STRIDE = TILE_K + SHMEM_PAD; // 40
constexpr int BS_STRIDE = TILE_N + SHMEM_PAD; // 72

template <typename HalfT>
__device__ __forceinline__ void wmma_matmul_tile_safe(
    const HalfT* __restrict__ A,
    const HalfT* __restrict__ B,
    HalfT* __restrict__ C,
    int64_t M, int64_t N, int64_t K)
{
    __shared__ alignas(16) HalfT As[TILE_M][AS_STRIDE];
    __shared__ alignas(16) HalfT Bs[TILE_K][BS_STRIDE];
    __shared__ alignas(16) float Cs[TILE_M][BS_STRIDE];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    const int64_t block_row = blockIdx.y * TILE_M;
    const int64_t block_col = blockIdx.x * TILE_N;

    constexpr int WARP_M_TILES = (TILE_M / WARPS_M) / WMMA_M; // 2
    constexpr int WARP_N_TILES = (TILE_N / WARPS_N) / WMMA_N; // 2

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WARP_M_TILES][WARP_N_TILES];

    #pragma unroll
    for (int i = 0; i < WARP_M_TILES; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_N_TILES; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    constexpr int A_ELEMS = TILE_M * TILE_K;
    constexpr int B_ELEMS = TILE_K * TILE_N;

    // Loop principal sobre K
    for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {

        // 1. Carga segura y coalescida de A -> Shared Memory
        #pragma unroll
        for (int i = tid; i < A_ELEMS; i += THREADS_PER_BLOCK) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int64_t gr = block_row + r;
            int64_t gc = k0 + c;

            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : HalfT(0.0f);
        }

        // 2. Carga segura y coalescida de B -> Shared Memory
        #pragma unroll
        for (int i = tid; i < B_ELEMS; i += THREADS_PER_BLOCK) {
            int r = i / TILE_N;
            int c = i % TILE_N;
            int64_t gr = k0 + r;
            int64_t gc = block_col + c;

            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : HalfT(0.0f);
        }

        __syncthreads();

        // 3. Multiplicación WMMA (2 pasos de K=16)
        #pragma unroll
        for (int k_step = 0; k_step < TILE_K; k_step += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, HalfT, wmma::row_major> a_frag[WARP_M_TILES];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, HalfT, wmma::row_major> b_frag[WARP_N_TILES];

            #pragma unroll
            for (int i = 0; i < WARP_M_TILES; ++i) {
                int warp_row = warp_m * (WARP_M_TILES * WMMA_M) + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], &As[warp_row][k_step], AS_STRIDE);
            }

            #pragma unroll
            for (int j = 0; j < WARP_N_TILES; ++j) {
                int warp_col = warp_n * (WARP_N_TILES * WMMA_N) + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j], &Bs[k_step][warp_col], BS_STRIDE);
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
    }

    // 4. Guardado en Shared Memory FP32
    #pragma unroll
    for (int i = 0; i < WARP_M_TILES; ++i) {
        int warp_row = warp_m * (WARP_M_TILES * WMMA_M) + i * WMMA_M;
        #pragma unroll
        for (int j = 0; j < WARP_N_TILES; ++j) {
            int warp_col = warp_n * (WARP_N_TILES * WMMA_N) + j * WMMA_N;
            wmma::store_matrix_sync(&Cs[warp_row][warp_col], c_frag[i][j], BS_STRIDE, wmma::mem_row_major);
        }
    }

    __syncthreads();

    // 5. Escritura final segura a Memoria Global C
    constexpr int C_ELEMS = TILE_M * TILE_N;
    #pragma unroll
    for (int i = tid; i < C_ELEMS; i += THREADS_PER_BLOCK) {
        int r = i / TILE_N;
        int c = i % TILE_N;
        int64_t gr = block_row + r;
        int64_t gc = block_col + c;

        if (gr < M && gc < N) {
            C[gr * N + gc] = from_float<HalfT>(Cs[r][c]);
        }
    }
}

extern "C" __global__ void matmul_wmma_kernel(
    const KERNEL_DTYPE* A, const KERNEL_DTYPE* B, KERNEL_DTYPE* C,
    int64_t M, int64_t N, int64_t K)
{
    wmma_matmul_tile_safe<KERNEL_DTYPE>(A, B, C, M, N, K);
}