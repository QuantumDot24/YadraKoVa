// Matmul naive: un thread por elemento de C, sin tiling ni shared
// memory. A proposito lento -- hoy el objetivo es CORRECTITUD, no
// velocidad. KERNEL_DTYPE se define via -D al compilar (ver
// compile_kernel.py), una compilacion separada por dtype.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifndef KERNEL_DTYPE
#define KERNEL_DTYPE float
#endif

extern "C" __global__ void matmul_kernel(
    const KERNEL_DTYPE* A, const KERNEL_DTYPE* B, KERNEL_DTYPE* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    // Acumulamos siempre en fp32 -- incluso para bf16/fp16 -- para
    // evitar perder precision en la suma, aunque el dato de entrada/
    // salida sea de menor precision. Practica estandar.
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += float(A[row * K + k]) * float(B[k * N + col]);
    }
    C[row * N + col] = KERNEL_DTYPE(acc);
}