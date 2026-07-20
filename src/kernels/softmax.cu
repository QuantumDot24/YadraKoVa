// Softmax por fila, numericamente estable. Un block por fila, un
// warp por block (asume N <= 32 por ahora -- version simple para
// probar common.cuh; softmax de N arbitrario con multiples warps
// es un paso posterior).
#include "../../include/kernels/common.cuh"

#ifndef KERNEL_DTYPE
#define KERNEL_DTYPE float
#endif

using namespace yadrakova::kernels::common;

extern "C" __global__ void softmax_kernel(
    const KERNEL_DTYPE* input, KERNEL_DTYPE* output, int64_t rows, int64_t cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x; // 0..31, un warp
    if (row >= rows) return;

    const KERNEL_DTYPE* row_in = input + row * cols;
    KERNEL_DTYPE* row_out = output + row * cols;

    // Paso 1: max de la fila (estabilidad numerica -- evita overflow
    // en expf para valores grandes).
    float local_max = -INFINITY;
    for (int c = tid; c < cols; c += 32)
        local_max = fmaxf(local_max, to_float<KERNEL_DTYPE>(row_in[c]));
    float row_max = warp_reduce_max(local_max);
    row_max = __shfl_sync(0xffffffff, row_max, 0); // broadcast del lane 0 a todo el warp

    // Paso 2: suma de exp(x - max)
    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += 32)
        local_sum += expf(to_float<KERNEL_DTYPE>(row_in[c]) - row_max);
    float row_sum = warp_reduce_sum(local_sum);
    row_sum = __shfl_sync(0xffffffff, row_sum, 0);

    // Paso 3: escribir softmax normalizado
    for (int c = tid; c < cols; c += 32) {
        float val = expf(to_float<KERNEL_DTYPE>(row_in[c]) - row_max) / row_sum;
        row_out[c] = from_float<KERNEL_DTYPE>(val);
    }
}