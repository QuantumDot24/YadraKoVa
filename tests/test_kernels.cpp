#include "core/tensor.hpp"
#include "core/event.hpp"
#include "core/backend_dispatch.hpp"
#include <cmath>
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace yadrakova::core;

// ===========================================================================
// Shared helpers
// ===========================================================================

float cpu_gelu(float x)
{
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
}

std::vector<float> cpu_softmax_row(const std::vector<float>& row)
{
    float max_val = *std::max_element(row.begin(), row.end());
    std::vector<float> exp_vals(row.size());
    float sum = 0.0f;
    for (size_t i = 0; i < row.size(); ++i) {
        exp_vals[i] = std::exp(row[i] - max_val);
        sum += exp_vals[i];
    }
    for (auto& v : exp_vals) v /= sum;
    return exp_vals;
}

// BF16 tolerance check, same criterion as torch.allclose:
//   diff <= atol + rtol * |ref|
struct Bf16CompareResult {
    size_t mismatches = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
};

Bf16CompareResult compare_bf16_vectors(const std::vector<__nv_bfloat16>& a,
                                       const std::vector<__nv_bfloat16>& b,
                                       float atol, float rtol)
{
    Bf16CompareResult r;
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i)
    {
        float fa = __bfloat162float(a[i]);
        float fb = __bfloat162float(b[i]);
        float diff = std::fabs(fa - fb);
        float allowed = atol + rtol * std::fabs(fb);
        if (diff > allowed)
        {
            ++r.mismatches;
            r.max_abs = std::max(r.max_abs, diff);
            r.max_rel = std::max(r.max_rel, diff / std::max(std::fabs(fb), 1e-6f));
        }
    }
    return r;
}

// ===========================================================================
// Section 1: GELU correctness (vs CPU reference)
// ===========================================================================

void test_gelu_correctness_vs_cpu()
{
    const int N = 1024;
    Tensor<float> A = Tensor<float>::randn({N});
    Tensor<float> C = A.gelu();

    auto h_in  = A.to_vector();
    auto h_out = C.to_vector();

    float max_diff = 0.0f;
    for (int i = 0; i < N; ++i)
        max_diff = std::max(max_diff, std::fabs(cpu_gelu(h_in[i]) - h_out[i]));

    std::cout << "  gelu max_diff vs CPU: " << max_diff << "\n";
    assert(max_diff < 1e-4f);
    std::cout << "[OK] gelu_correctness_vs_cpu\n";
}

// ===========================================================================
// Section 2: Softmax correctness (vs CPU reference, FP32)
// ===========================================================================

void test_softmax_correctness_vs_cpu()
{
    const int rows = 4, cols = 16;
    Tensor<float> A = Tensor<float>::randn({rows, cols});
    Tensor<float> C = A.softmax();

    auto h_in  = A.to_vector();
    auto h_out = C.to_vector();

    float max_diff = 0.0f;
    for (int r = 0; r < rows; ++r) {
        std::vector<float> row(h_in.begin() + r * cols, h_in.begin() + (r + 1) * cols);
        std::vector<float> ref = cpu_softmax_row(row);
        for (int c = 0; c < cols; ++c)
            max_diff = std::max(max_diff, std::fabs(ref[c] - h_out[r * cols + c]));
    }

    std::cout << "  softmax max_diff vs CPU: " << max_diff << "\n";
    assert(max_diff < 1e-5f);
    std::cout << "[OK] softmax_correctness_vs_cpu\n";
}

// ===========================================================================
// Section 3: MatMul correctness (Custom vs cuBLAS, BF16)
// ===========================================================================

void test_matmul_correctness(int M, int N, int K, unsigned seed_a = 1, unsigned seed_b = 2)
{
    Tensor<__nv_bfloat16> A = Tensor<__nv_bfloat16>::randn({M, K}, seed_a);
    Tensor<__nv_bfloat16> B = Tensor<__nv_bfloat16>::randn({K, N}, seed_b);

    auto C_mine = A.matmul(B, Backend::Custom);
    auto C_ref  = A.matmul(B, Backend::CuBLAS);

    // For large K (e.g. 8192), accumulation needs a slightly looser atol.
    const float atol = (K >= 4096) ? 2e-2f : 1e-2f;
    const float rtol = 1e-2f;

    auto result = compare_bf16_vectors(C_mine.to_vector(), C_ref.to_vector(), atol, rtol);
    double pct = 100.0 * static_cast<double>(result.mismatches) / C_mine.to_vector().size();

    std::cout << "MatMul correctness " << M << "x" << K << " @ " << K << "x" << N << "\n";
    if (result.mismatches > 0) {
        std::cout << "  [FAIL] Elements out of tolerance: " << result.mismatches
                  << " / " << C_mine.to_vector().size() << " (" << pct << "%)\n"
                  << "  Max abs error: " << result.max_abs
                  << " | Max rel error: " << result.max_rel << "\n";
        assert(result.mismatches == 0);
    } else {
        std::cout << "  [PASS] Custom vs cuBLAS within BF16 tolerance (atol="
                  << atol << ", rtol=" << rtol << ").\n";
    }
}

void test_auto_dispatch_matmul_picks_cublas()
{
    auto resolution = OpsDispatch::resolve(Op::MatMul, DType::BF16, Backend::Auto);
    std::cout << "Backend::Auto for matmul/bf16 resolved to: "
              << backend_name(resolution.backend) << "\n";
    assert(resolution.backend == Backend::CuBLAS);
    assert(!resolution.dtype_was_downgraded);
    std::cout << "  [PASS] Auto picked cuBLAS without Tensor having to request it explicitly.\n";
}

// ===========================================================================
// Section 4: Softmax correctness (Custom vs cuDNN, BF16) + dispatch
// ===========================================================================

void test_softmax_correctness_bf16(int rows, int cols, unsigned seed = 7)
{
    Tensor<__nv_bfloat16> X = Tensor<__nv_bfloat16>::randn({rows, cols}, seed);

    auto Y_mine  = X.softmax(Backend::Custom);
    auto Y_cudnn = X.softmax(Backend::CuDNN);

    const float atol = 1e-2f, rtol = 1e-2f;
    auto result = compare_bf16_vectors(Y_mine.to_vector(), Y_cudnn.to_vector(), atol, rtol);
    double pct = 100.0 * static_cast<double>(result.mismatches) / Y_mine.to_vector().size();

    std::cout << "Softmax correctness " << rows << "x" << cols << "\n";
    if (result.mismatches > 0) {
        std::cout << "  [FAIL] Elements out of tolerance: " << result.mismatches
                  << " / " << Y_mine.to_vector().size() << " (" << pct << "%)\n"
                  << "  Max abs error: " << result.max_abs
                  << " | Max rel error: " << result.max_rel << "\n";
        assert(result.mismatches == 0);
    } else {
        std::cout << "  [PASS] Custom vs cuDNN within BF16 tolerance (atol="
                  << atol << ", rtol=" << rtol << ").\n";
    }
}

void test_auto_dispatch_softmax_picks_cudnn()
{
    auto resolution = OpsDispatch::resolve(Op::Softmax, DType::BF16, Backend::Auto);
    std::cout << "Backend::Auto for softmax/bf16 resolved to: "
              << backend_name(resolution.backend) << "\n";
    assert(resolution.backend == Backend::CuDNN);
    std::cout << "  [PASS] Auto picked cuDNN without Tensor having to request it explicitly.\n";
}

// ===========================================================================
// Section 5: Benchmarks (parameterized by Backend)
// ===========================================================================

constexpr int kBenchIters = 20;

void benchmark_matmul(const char* label, Backend backend, int M, int N, int K)
{
    Tensor<__nv_bfloat16> A = Tensor<__nv_bfloat16>::randn({M, K});
    Tensor<__nv_bfloat16> B = Tensor<__nv_bfloat16>::randn({K, N});

    // Warm-up: for cuBLAS this creates the handle once via BackendContext;
    // for Custom it is a no-op but keeps the same code shape.
    A.matmul(B, backend);

    float total_ms = 0.0f;
    for (int i = 0; i < kBenchIters; ++i)
        total_ms += time_kernel_ms([&] { A.matmul(B, backend); });

    float ms = total_ms / kBenchIters;
    double gflops = 2.0 * M * N * K / (ms * 1e6);

    std::cout << label << "  " << M << "x" << K << " @ " << K << "x" << N
              << " -> " << ms << " ms (avg of " << kBenchIters << ") ("
              << gflops << " GFLOPS)\n";
}

void benchmark_softmax(const char* label, Backend backend, int rows, int cols)
{
    Tensor<__nv_bfloat16> X = Tensor<__nv_bfloat16>::randn({rows, cols});
    X.softmax(backend);

    float total_ms = 0.0f;
    for (int i = 0; i < kBenchIters; ++i)
        total_ms += time_kernel_ms([&] { X.softmax(backend); });

    float ms = total_ms / kBenchIters;
    std::cout << label << "  softmax " << rows << "x" << cols
              << " -> " << ms << " ms (avg of " << kBenchIters << ")\n";
}

// cuDNN does not expose GELU via its stable activation API (cudnnActivationMode_t
// has no GELU -- only the Graph API does, which is not implemented yet, see
// backend_caps.hpp). This benchmark keeps your custom kernel's reference
// number on hand, without comparing against anything.
void benchmark_gelu_custom_only(int n)
{
    Tensor<__nv_bfloat16> X = Tensor<__nv_bfloat16>::randn({n});
    X.gelu(Backend::Custom);

    float total_ms = 0.0f;
    for (int i = 0; i < kBenchIters; ++i)
        total_ms += time_kernel_ms([&] { X.gelu(Backend::Custom); });

    float ms = total_ms / kBenchIters;
    std::cout << "Mine  gelu n=" << n << " -> " << ms << " ms (avg of " << kBenchIters
              << ") -- no cuDNN comparison (unsupported, see backend_caps.hpp)\n";
}

// ===========================================================================
int main()
{
    try
    {
        std::cout << "=== KERNELS & BACKENDS TEST SUITE ===\n\n";

        std::cout << "-- FP32 correctness vs CPU --\n";
        test_gelu_correctness_vs_cpu();
        test_softmax_correctness_vs_cpu();

        std::cout << "\n-- MatMul BF16: Custom vs cuBLAS --\n";
        test_matmul_correctness(256, 256, 256);
        test_matmul_correctness(8192, 8192, 8192);
        test_auto_dispatch_matmul_picks_cublas();

        std::cout << "\n-- Softmax BF16: Custom vs cuDNN --\n";
        test_softmax_correctness_bf16(256, 256);
        test_softmax_correctness_bf16(4096, 4096);
        test_auto_dispatch_softmax_picks_cudnn();

        std::cout << "\n-- Benchmarks --\n";
        benchmark_matmul("Mine  ", Backend::Custom, 8192, 8192, 8192);
        benchmark_matmul("cuBLAS", Backend::CuBLAS, 8192, 8192, 8192);
        benchmark_softmax("Mine  ", Backend::Custom, 4096, 4096);
        benchmark_softmax("cuDNN", Backend::CuDNN, 4096, 4096);
        benchmark_gelu_custom_only(4096 * 4096);

        std::cout << "\n(LayerNorm is not in this suite -- Tensor::layer_norm and a\n"
                  << "custom kernel do not exist yet, see backend_caps.hpp.)\n";

        std::cout << "\nAll kernel tests passed.\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    return 0;
}