// src/kernels/matmul_wmma.cu
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
#include "../../include/kernels/common.cuh"

#ifndef KERNEL_DTYPE
#define KERNEL_DTYPE __nv_bfloat16
#endif

struct __nv_bfloat16;
using namespace nvcuda;
using namespace yadrakova::kernels::common;

// Tiles pensados para caber en 512 threads/block (16 warps), bien
// dentro del limite de 1024. TILE_K == WMMA_K: un solo mma_sync por
// iteracion del loop de K -- mas simple de razonar que un doble loop;
// eso es la siguiente optimizacion, no la de hoy.
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int TILE_M = 64, TILE_N = 64, TILE_K = WMMA_K;
constexpr int WARPS_M = TILE_M / WMMA_M; // 4
constexpr int WARPS_N = TILE_N / WMMA_N; // 4
constexpr int THREADS_PER_BLOCK = WARPS_M * WARPS_N * 32; // 512

// bf16/fp16 con tensor cores. Solo se instancia (y solo se compila
// codigo real) cuando KERNEL_DTYPE es uno de estos dos tipos -- el
// build system ya compila un .cubin separado por dtype, asi que el
// resto del template ni siquiera se genera para float/int8.
//hola
template <typename HalfT>
__device__ __forceinline__ void wmma_matmul_tile(
    const HalfT* A, const HalfT* B, HalfT* C, int64_t M, int64_t N, int64_t K)
{
    __shared__ HalfT As[TILE_M][TILE_K];
    __shared__ HalfT Bs[TILE_K][TILE_N];
    __shared__ float Cs[TILE_M][TILE_N];

    const int tid = threadIdx.x; // block 1D de THREADS_PER_BLOCK
    const int warp_id = tid / 32;
    const int warp_row = (warp_id / WARPS_N) * WMMA_M;
    const int warp_col = (warp_id % WARPS_N) * WMMA_N;

    const int64_t tile_row0 = blockIdx.y * TILE_M;
    const int64_t tile_col0 = blockIdx.x * TILE_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, HalfT, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, HalfT, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    constexpr int A_ELEMS = TILE_M * TILE_K;
    constexpr int B_ELEMS = TILE_K * TILE_N;

    for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
        // Carga con stride: cada thread llena varios elementos si
        // A_ELEMS/B_ELEMS > THREADS_PER_BLOCK (aqui 1024 > 512 -> 2 c/u).
        for (int i = tid; i < A_ELEMS; i += THREADS_PER_BLOCK) {
            int r = i / TILE_K, c = i % TILE_K;
            int64_t gr = tile_row0 + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : HalfT(0.0f);
        }
        for (int i = tid; i < B_ELEMS; i += THREADS_PER_BLOCK) {
            int r = i / TILE_N, c = i % TILE_N;
            int64_t gr = k0 + r, gc = tile_col0 + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : HalfT(0.0f);
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, &As[warp_row][0], TILE_K);
        wmma::load_matrix_sync(b_frag, &Bs[0][warp_col], TILE_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // WMMA siempre acumula en fp32 (hardware) -- store a shared como
    // float, luego castea a T al escribir a global. Asi C queda en el
    // mismo dtype que A/B, igual que gelu/softmax.
    wmma::store_matrix_sync(&Cs[warp_row][warp_col], c_frag, TILE_N, wmma::mem_row_major);
    __syncthreads();

    constexpr int C_ELEMS = TILE_M * TILE_N;
    for (int i = tid; i < C_ELEMS; i += THREADS_PER_BLOCK) {
        int r = i / TILE_N, c = i % TILE_N;
        int64_t gr = tile_row0 + r, gc = tile_col0 + c;
        if (gr < M && gc < N) C[gr * N + gc] = from_float<HalfT>(Cs[r][c]);
    }
}

extern "C" __global__ void matmul_wmma_kernel(
    const KERNEL_DTYPE* A, const KERNEL_DTYPE* B, KERNEL_DTYPE* C,
    int64_t M, int64_t N, int64_t K)
{
    wmma_matmul_tile<KERNEL_DTYPE>(A, B, C, M, N, K);
}