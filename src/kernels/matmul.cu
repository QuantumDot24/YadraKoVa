// Matmul naive: un thread por elemento de C, sin tiling ni shared
// memory. A proposito lento -- hoy el objetivo es CORRECTITUD, no
// velocidad. KERNEL_DTYPE se define via -D al compilar (ver
// compile_kernel.py), una compilacion separada por dtype.
//
// CAMBIO: M, N, K ahora son int64_t (antes int). Convencion de
// proyecto: todo escalar de forma que llega a un kernel es int64_t,
// para que coincida 1:1 con core::Shape (vector<int64_t>) y con lo
// que ya declara el .yaml de dispatch (dtype: int64). Asi el
// Executor nunca tiene que truncar/castear tipos por kernel -- el
// tamano de cada argumento en args[] siempre coincide con lo que
// espera la firma real del .cubin.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#ifndef KERNEL_DTYPE
#define KERNEL_DTYPE float
#endif

extern "C" __global__ void matmul_kernel(
    const KERNEL_DTYPE* A, const KERNEL_DTYPE* B, KERNEL_DTYPE* C,
    int64_t M, int64_t N, int64_t K)
{
    int64_t row = blockIdx.y * (int64_t)blockDim.y + threadIdx.y;
    int64_t col = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    // Acumulamos siempre en fp32 -- incluso para bf16/fp16 -- para
    // evitar perder precision en la suma, aunque el dato de entrada/
    // salida sea de menor precision. Practica estandar.
    float acc = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        acc += float(A[row * K + k]) * float(B[k * N + col]);
    }
    C[row * N + col] = KERNEL_DTYPE(acc);
}
