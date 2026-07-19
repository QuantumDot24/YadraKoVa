// GELU elementwise -- un thread por elemento. Usa to_float/from_float
// de common.cuh para no repetir la logica de conversion de dtype.
#include "kernels/common.cuh"

#ifndef KERNEL_DTYPE
#define KERNEL_DTYPE float
#endif

using namespace yadrakova::kernels::common;

extern "C" __global__ void gelu_kernel(
    const KERNEL_DTYPE* input, KERNEL_DTYPE* output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = to_float<KERNEL_DTYPE>(input[idx]);
    output[idx] = from_float<KERNEL_DTYPE>(gelu_exact(x));
}