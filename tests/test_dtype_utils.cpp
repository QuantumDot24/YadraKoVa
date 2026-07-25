#include "core/dtype_utils.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace yadrakova::core;

void test_dtype_size_and_name()
{
    assert(dtype_size(DType::FP32) == 4);
    assert(dtype_size(DType::BF16) == 2);
    assert(dtype_size(DType::FP16) == 2);
    assert(dtype_size(DType::INT8) == 1);

    assert(std::string(dtype_name(DType::FP32)) == "fp32");
    assert(std::string(dtype_name(DType::BF16)) == "bf16");
    assert(std::string(dtype_name(DType::FP16)) == "fp16");
    assert(std::string(dtype_name(DType::INT8)) == "int8");

    static_assert(dtype_traits<float>::value == DType::FP32);
    static_assert(dtype_traits<int8_t>::value == DType::INT8);

    std::cout << "[OK] dtype_size_and_name\n";
}

void test_scalar_roundtrip_bf16()
{
    float original = 3.14159265f;
    __nv_bfloat16 as_bf16 = f32_to_bf16(original);
    float back = bf16_to_f32(as_bf16);

    float rel_err = std::fabs(back - original) / std::fabs(original);
    std::cout << "  bf16 roundtrip: " << original << " -> " << back
        << " (rel_err=" << rel_err << ")\n";
    assert(rel_err < 1e-2f);

    std::cout << "[OK] scalar_roundtrip_bf16\n";
}

void test_scalar_roundtrip_fp16()
{
    float original = 3.14159265f;
    __half as_fp16 = f32_to_f16(original);
    float back = f16_to_f32(as_fp16);

    float rel_err = std::fabs(back - original) / std::fabs(original);
    std::cout << "  fp16 roundtrip: " << original << " -> " << back
        << " (rel_err=" << rel_err << ")\n";
    assert(rel_err < 1e-3f);

    std::cout << "[OK] scalar_roundtrip_fp16\n";
}

void test_bf16_fp16_cross_conversion()
{
    float original = 2.71828f;
    __nv_bfloat16 bf = f32_to_bf16(original);
    __half via_fp32 = bf16_to_f16(bf);
    __nv_bfloat16 back = f16_to_bf16(via_fp32);

    float original_f32 = bf16_to_f32(bf);
    float roundtrip_f32 = bf16_to_f32(back);

    assert(original_f32 == roundtrip_f32);

    std::cout << "[OK] bf16_fp16_cross_conversion\n";
}

void test_buffer_conversion_f32_bf16()
{
    const size_t n = 256;
    std::vector<float> src(n);
    for (size_t i = 0; i < n; ++i) src[i] = std::sin(static_cast<float>(i) * 0.1f) * 10.0f;

    std::vector<__nv_bfloat16> mid(n);
    std::vector<float> back(n);
    convert_f32_to_bf16(src.data(), mid.data(), n);
    convert_bf16_to_f32(mid.data(), back.data(), n);

    float max_rel_err = 0.0f;
    for (size_t i = 0; i < n; ++i)
    {
        if (std::fabs(src[i]) < 1e-6f) continue;
        float rel_err = std::fabs(back[i] - src[i]) / std::fabs(src[i]);
        max_rel_err = std::max(max_rel_err, rel_err);
    }
    std::cout << "  max rel_err sobre " << n << " elementos: " << max_rel_err << "\n";
    assert(max_rel_err < 1e-2f);

    std::cout << "[OK] buffer_conversion_f32_bf16\n";
}

void test_buffer_conversion_f32_fp16()
{
    const size_t n = 256;
    std::vector<float> src(n);
    for (size_t i = 0; i < n; ++i) src[i] = std::cos(static_cast<float>(i) * 0.05f) * 5.0f;

    std::vector<__half> mid(n);
    std::vector<float> back(n);
    convert_f32_to_f16(src.data(), mid.data(), n);
    convert_f16_to_f32(mid.data(), back.data(), n);

    float max_rel_err = 0.0f;
    for (size_t i = 0; i < n; ++i)
    {
        if (std::fabs(src[i]) < 1e-6f) continue;
        float rel_err = std::fabs(back[i] - src[i]) / std::fabs(src[i]);
        max_rel_err = std::max(max_rel_err, rel_err);
    }
    assert(max_rel_err < 1e-3f);

    std::cout << "[OK] buffer_conversion_f32_fp16\n";
}

void test_quantization_roundtrip()
{
    const size_t n = 128;
    std::vector<float> src(n);
    for (size_t i = 0; i < n; ++i)
    {
        src[i] = (static_cast<float>(i) - 20.0f) * 0.37f;
    }

    QuantParams qp = compute_symmetric_scale(src.data(), n);
    assert(qp.scale > 0.0f);

    std::vector<int8_t> quantized(n);
    std::vector<float> dequantized(n);
    quantize_i8(src.data(), quantized.data(), n, qp);
    dequantize_i8(quantized.data(), dequantized.data(), n, qp);

    float max_abs_err = 0.0f;
    for (size_t i = 0; i < n; ++i)
    {
        max_abs_err = std::max(max_abs_err, std::fabs(dequantized[i] - src[i]));
    }
    std::cout << "  scale=" << qp.scale << " max_abs_err=" << max_abs_err << "\n";
    assert(max_abs_err <= qp.scale * 0.5f + 1e-6f);

    for (size_t i = 0; i < n; ++i)
    {
        assert(quantized[i] >= -127 && quantized[i] <= 127);
    }

    std::cout << "[OK] quantization_roundtrip\n";
}

void test_quantization_all_zeros_does_not_divide_by_zero()
{
    std::vector<float> src(64, 0.0f);
    QuantParams qp = compute_symmetric_scale(src.data(), src.size());
    assert(qp.scale == 1.0f);

    std::vector<int8_t> quantized(64);
    quantize_i8(src.data(), quantized.data(), 64, qp);
    for (int8_t v : quantized)
        assert(v == 0);

    std::cout << "[OK] quantization_all_zeros_does_not_divide_by_zero\n";
}

void test_precision_bf16_vs_fp16_diverge()
{
    const size_t n = 500;
    double sum_err_bf16 = 0.0;
    double sum_err_fp16 = 0.0;

    for (size_t i = 0; i < n; ++i)
    {
        float original = 0.7f + static_cast<float>(i) * 0.0137f;

        float via_bf16 = bf16_to_f32(f32_to_bf16(original));
        float via_fp16 = f16_to_f32(f32_to_f16(original));

        sum_err_bf16 += std::fabs(via_bf16 - original);
        sum_err_fp16 += std::fabs(via_fp16 - original);
    }

    double avg_err_bf16 = sum_err_bf16 / n;
    double avg_err_fp16 = sum_err_fp16 / n;

    std::cout << "  avg_err_bf16=" << avg_err_bf16
        << " avg_err_fp16=" << avg_err_fp16 << "\n";

    assert(avg_err_fp16 * 3.0 < avg_err_bf16);

    std::cout << "[OK] precision_bf16_vs_fp16_diverge\n";
}

int main()
{
    test_dtype_size_and_name();
    test_scalar_roundtrip_bf16();
    test_scalar_roundtrip_fp16();
    test_bf16_fp16_cross_conversion();
    test_buffer_conversion_f32_bf16();
    test_buffer_conversion_f32_fp16();
    test_quantization_roundtrip();
    test_quantization_all_zeros_does_not_divide_by_zero();
    test_precision_bf16_vs_fp16_diverge();
    std::cout << "Todos los tests de dtype_utils pasaron.\n";
    return 0;
}
