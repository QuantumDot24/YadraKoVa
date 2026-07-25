// Logica matematica compartida entre kernels. Cambios aqui invalidan
// el cache de TODOS los kernels (ver hash_shared_headers en
// compile_kernel.py) -- intencional, evita servir un .cubin
// compilado contra una version vieja de estas funciones.
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace yadrakova::kernels::common {


template <typename T>
__device__ __forceinline__ float to_float(T v) { return float(v); }

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
    return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v) { return T(v); }

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ __half from_float<__half>(float v) {
    return __float2half(v);
}

// --- Reducciones warp-level (32 threads), via shuffle. No usan
//     shared memory -- solo validas DENTRO de un mismo warp. ---

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f)); // 1/sqrt(2)
}

} // namespace yadrakova::kernels::common