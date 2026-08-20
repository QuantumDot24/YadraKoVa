// mul.cu -- kernel Custom elementwise: c = a * b.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

extern "C" __global__ void mul(const KERNEL_DTYPE* a, const KERNEL_DTYPE* b, KERNEL_DTYPE* c, int64_t n)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] * b[i];
    }
}