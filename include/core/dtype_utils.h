#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace yadrakova::core {


enum class DType : uint8_t { BF16, FP32, FP16, INT8 };

template <typename T> struct dtype_traits;
template <> struct dtype_traits<__nv_bfloat16> { static constexpr DType value = DType::BF16; static constexpr const char* name = "bf16"; };
template <> struct dtype_traits<float>          { static constexpr DType value = DType::FP32; static constexpr const char* name = "fp32"; };
template <> struct dtype_traits<__half>         { static constexpr DType value = DType::FP16; static constexpr const char* name = "fp16"; };
template <> struct dtype_traits<int8_t>         { static constexpr DType value = DType::INT8; static constexpr const char* name = "int8"; };

// --- introspección en runtime -----------------------------------------
// dtype_traits<T> te da esto en compile-time. Estas dos son para cuando
// el DType llega como valor runtime (ej: leido de un header de
// checkpoint) y no tienes el tipo T a la mano todavia.

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::FP32: return sizeof(float);
        case DType::BF16: return sizeof(__nv_bfloat16);
        case DType::FP16: return sizeof(__half);
        case DType::INT8: return sizeof(int8_t);
    }
    throw std::runtime_error("dtype_size: DType desconocido");
}

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::FP32: return dtype_traits<float>::name;
        case DType::BF16: return dtype_traits<__nv_bfloat16>::name;
        case DType::FP16: return dtype_traits<__half>::name;
        case DType::INT8: return dtype_traits<int8_t>::name;
    }
    throw std::runtime_error("dtype_name: DType desconocido");
}

// --- conversion escalar -------------------------------------------------
// __host__ __device__: usable tanto en tests/loaders (host) como dentro
// de un .cu (ej: un kernel de cast que convierte un tensor bf16 a fp32
// elemento por elemento). Esta es la razon principal de que este header
// no dependa de tensor.h -- un .cu de kernel quiere esto sin arrastrar
// MemoryPool/Stream/HostBuffer.

__host__ __device__ inline float bf16_to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
__host__ __device__ inline __nv_bfloat16 f32_to_bf16(float v) { return __float2bfloat16(v); }
__host__ __device__ inline float f16_to_f32(__half v)  { return __half2float(v); }
__host__ __device__ inline __half f32_to_f16(float v)  { return __float2half(v); }

// bf16 <-> fp16 no son directas en hardware -- siempre pasan por fp32.
// Es intencional dejarlo explicito en vez de esconder la doble conversion.
__host__ __device__ inline __half bf16_to_f16(__nv_bfloat16 v) { return f32_to_f16(bf16_to_f32(v)); }
__host__ __device__ inline __nv_bfloat16 f16_to_bf16(__half v) { return f32_to_bf16(f16_to_f32(v)); }

// --- conversion de buffers completos (host, secuencial) -----------------
// Para cargar pesos: el checkpoint casi siempre viene en fp32 o bf16 y
// necesitas llevarlo al dtype que tu kernel espera antes de subirlo a
// device (o convertir ya en device -- eso seria un kernel .cu aparte,
// esto es el path simple de CPU para loaders/tests).

inline void convert_f32_to_bf16(const float* src, __nv_bfloat16* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = f32_to_bf16(src[i]);
}
inline void convert_bf16_to_f32(const __nv_bfloat16* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = bf16_to_f32(src[i]);
}
inline void convert_f32_to_f16(const float* src, __half* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = f32_to_f16(src[i]);
}
inline void convert_f16_to_f32(const __half* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = f16_to_f32(src[i]);
}

// --- cuantizacion int8, simetrica per-tensor -----------------------------
// Suficiente para arrancar (per-channel/per-group vendra cuando haga
// falta precision extra, tipico en pesos de attention). Simetrica =
// zero_point implicito en 0, rango [-127, 127] (se evita -128 para que
// el rango sea simetrico y -x sea representable siempre).

struct QuantParams {
    float scale = 1.0f; // valor_real = valor_int8 * scale
};

inline QuantParams compute_symmetric_scale(const float* data, size_t n) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) max_abs = std::max(max_abs, std::fabs(data[i]));
    return QuantParams{ max_abs > 0.0f ? max_abs / 127.0f : 1.0f };
}

inline void quantize_i8(const float* src, int8_t* dst, size_t n, const QuantParams& qp) {
    for (size_t i = 0; i < n; ++i) {
        float v = std::clamp(src[i] / qp.scale, -127.0f, 127.0f);
        dst[i] = static_cast<int8_t>(std::lround(v));
    }
}

inline void dequantize_i8(const int8_t* src, float* dst, size_t n, const QuantParams& qp) {
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]) * qp.scale;
}

} // namespace yadrakova::core