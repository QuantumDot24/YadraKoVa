#include "core/tensor.hpp"
#include "core/backend_dispatch.hpp"
#include <cmath>
#include <cassert>
#include <iostream>
#include <vector>

using namespace yadrakova::core;

// ---------------------------------------------------------------------------
// CPU Reference Implementations
// ---------------------------------------------------------------------------
template <typename T>
std::vector<T> cpu_add(const std::vector<T>& a, const std::vector<T>& b) {
    std::vector<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = static_cast<T>(static_cast<float>(a[i]) + static_cast<float>(b[i]));
    }
    return result;
}

template <typename T>
std::vector<T> cpu_mul(const std::vector<T>& a, const std::vector<T>& b) {
    std::vector<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = static_cast<T>(static_cast<float>(a[i]) * static_cast<float>(b[i]));
    }
    return result;
}

// ---------------------------------------------------------------------------
// Correctness Tests
// ---------------------------------------------------------------------------
template <typename T>
void test_elementwise_correctness(Op op, const char* op_name, int rows, int cols)
{
    const int64_t n = static_cast<int64_t>(rows) * cols;
    std::cout << "  Testing " << op_name << " (" << dtype_name(dtype_traits<T>::value)
              << ", " << rows << "x" << cols << ")...\n";

    Tensor<T> A = Tensor<T>::randn({rows, cols}, 42);
    Tensor<T> B = Tensor<T>::randn({rows, cols}, 43);

    // Force cuTENSOR backend
    Tensor<T> C_gpu = (op == Op::Add) ? A.add(B, Backend::CuTensor) : A.mul(B, Backend::CuTensor);

    // CPU Reference
    auto vec_a = A.to_vector();
    auto vec_b = B.to_vector();
    std::vector<T> vec_ref = (op == Op::Add) ? cpu_add(vec_a, vec_b) : cpu_mul(vec_a, vec_b);
    auto vec_gpu = C_gpu.to_vector();

    size_t mismatches = 0;
    float max_abs = 0.0f, max_rel = 0.0f;
    const float atol = 1e-2f;
    const float rtol = 1e-2f;

    for (size_t i = 0; i < vec_gpu.size(); ++i)
    {
        float a = static_cast<float>(vec_gpu[i]);
        float b = static_cast<float>(vec_ref[i]);
        float diff = std::fabs(a - b);
        float allowed = atol + rtol * std::fabs(b);

        if (diff > allowed)
        {
            ++mismatches;
            max_abs = std::max(max_abs, diff);
            max_rel = std::max(max_rel, diff / std::max(std::fabs(b), 1e-6f));
        }
    }

    if (mismatches > 0)
    {
        std::cout << "    [FAIL] Mismatches: " << mismatches
                  << " | max_abs: " << max_abs << " | max_rel: " << max_rel << "\n";
        assert(mismatches == 0 && "cuTENSOR output differs from CPU reference");
    }
    else
    {
        std::cout << "    [PASS] cuTENSOR vs CPU within tolerance.\n";
    }
}

// ---------------------------------------------------------------------------
// Dispatch Tests
// ---------------------------------------------------------------------------
void test_auto_dispatch_picks_cutensor()
{
    std::cout << "  Testing Auto Dispatch resolution...\n";

    auto res_add_bf16 = OpsDispatch::resolve(Op::Add, DType::BF16, Backend::Auto);
    assert(res_add_bf16.backend == Backend::CuTensor);

    auto res_mul_fp32 = OpsDispatch::resolve(Op::Mul, DType::FP32, Backend::Auto);
    assert(res_mul_fp32.backend == Backend::CuTensor);

    std::cout << "    [PASS] Auto dispatch correctly prioritizes cuTENSOR for elementwise ops.\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main()
{
    try
    {
        std::cout << "=== YADRAKOVA CUTENSOR BACKEND TEST SUITE ===\n\n";

        std::cout << "-- Correctness (BF16) --\n";
        test_elementwise_correctness<__nv_bfloat16>(Op::Add, "Add", 32, 32);
        test_elementwise_correctness<__nv_bfloat16>(Op::Mul, "Mul", 32, 32);
        test_elementwise_correctness<__nv_bfloat16>(Op::Add, "Add", 1024, 1024);

        std::cout << "\n-- Correctness (FP32) --\n";
        test_elementwise_correctness<float>(Op::Add, "Add", 32, 32);
        test_elementwise_correctness<float>(Op::Mul, "Mul", 32, 32);

        std::cout << "\n-- Dispatch Logic --\n";
        test_auto_dispatch_picks_cutensor();

        std::cout << "\n[PASS] All cuTENSOR tests passed successfully.\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    return 0;
}