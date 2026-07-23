#include "core/tensor.hpp"
#include "core/event.hpp"
#include <cublas_v2.h>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>

using namespace yadrakova::core;

// ---------------------------------------------------------------------------
// CuBLAS helper (C = A * B,  bf16)
// ---------------------------------------------------------------------------
void cublas_gemm_bf16(const Tensor<__nv_bfloat16>& A,
                      const Tensor<__nv_bfloat16>& B,
                      Tensor<__nv_bfloat16>& C,
                      Stream& stream)
{
    static cublasHandle_t handle = nullptr;
    if (!handle)
    {
        cublasCreate(&handle);
    }
    cublasSetStream(handle, stream.raw());

    const float alpha = 1.0f, beta = 0.0f;
    // A and B are row-major. cublas expects column-major, so we do
    // C^t = B^t * A^t  =>  C = A * B.
    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 B.shape()[1], A.shape()[0], A.shape()[1],
                 &alpha,
                 B.data(), CUDA_R_16BF, B.shape()[1],
                 A.data(), CUDA_R_16BF, A.shape()[1],
                 &beta,
                 C.data(), CUDA_R_16BF, B.shape()[1],
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// ---------------------------------------------------------------------------
// Correctness: compares your kernel with cuBLAS for a given size.
// Now called both for a small size (256) and for the real size being
// benchmarked (8192), because the relative error in BF16 can grow with
// larger K (more accumulated terms = more rounding).
// ---------------------------------------------------------------------------
void test_matmul_wmma_correctness(int M, int N, int K, unsigned seed_a = 1, unsigned seed_b = 2)
{
    Tensor<__nv_bfloat16> A = Tensor<__nv_bfloat16>::randn({M, K}, seed_a);
    Tensor<__nv_bfloat16> B = Tensor<__nv_bfloat16>::randn({K, N}, seed_b);

    Stream stream;
    auto C_mine = A.matmul(B, stream);
    stream.synchronize();

    Tensor<__nv_bfloat16> C_ref({M, N});
    cublas_gemm_bf16(A, B, C_ref, stream);
    stream.synchronize();

    auto vec_mine = C_mine.to_vector();
    auto vec_ref = C_ref.to_vector();

    size_t mismatches = 0;
    float max_rel = 0.0f;
    float max_abs = 0.0f;

    // Standard tolerances for BF16
    // For very large K (e.g., 8192), accumulation requires a slightly more relaxed atol
    const float atol = (K >= 4096) ? 2e-2f : 1e-2f;
    const float rtol = 1e-2f;

    for (size_t i = 0; i < vec_mine.size(); ++i)
    {
        float a = __bfloat162float(vec_mine[i]);
        float b = __bfloat162float(vec_ref[i]);
        float diff = std::fabs(a - b);

        // torch.allclose-style criterion: diff <= atol + rtol * |b|
        float allowed_tol = atol + rtol * std::fabs(b);

        if (diff > allowed_tol)
        {
            ++mismatches;
            float rel = diff / std::max(std::fabs(b), 1e-6f);
            max_rel = std::max(max_rel, rel);
            max_abs = std::max(max_abs, diff);
        }
    }

    double mismatch_pct = 100.0 * static_cast<double>(mismatches) / vec_mine.size();

    std::cout << "Correctness " << M << "x" << K << " @ " << K << "x" << N << "\n";
    if (mismatches > 0)
    {
        std::cout << "  [FAIL] Elements out of tolerance: " << mismatches
                  << " / " << vec_mine.size() << " (" << mismatch_pct << "%)\n"
                  << "  Max absolute error: " << max_abs
                  << " | Max relative error: " << max_rel << "\n";
        assert(mismatches == 0);
    }
    else
    {
        std::cout << "  [PASS] Within tolerances for BF16 (atol="
                  << atol << ", rtol=" << rtol << ").\n";
    }

    std::cout << "[OK] matmul_wmma correctness (" << M << "x" << K << ")\n";
}
// ---------------------------------------------------------------------------
// Benchmark: same measurement mechanism (CUDA events via time_kernel_ms)
// for BOTH kernels, and average over N iterations to reduce noise.
// Previously: your kernel was measured with cudaEvent (GPU time only) and
// cuBLAS with host chrono + sync (includes launch overhead). This biased
// the comparison against cuBLAS, not in your favor -- so the real gap
// could be larger than what the previous table showed.
// ---------------------------------------------------------------------------
constexpr int kBenchIters = 20;

void benchmark_matmul_wmma(int M, int N, int K)
{
    Tensor<__nv_bfloat16> A = Tensor<__nv_bfloat16>::randn({M, K});
    Tensor<__nv_bfloat16> B = Tensor<__nv_bfloat16>::randn({K, N});

    Stream profile_stream;

    A.matmul(B, profile_stream);
    profile_stream.synchronize();

    float total_ms = 0.0f;
    for (int i = 0; i < kBenchIters; ++i)
    {
        total_ms += time_kernel_ms(profile_stream, [&]
        {
            A.matmul(B, profile_stream);
        });
    }
    float ms = total_ms / kBenchIters;

    double gflops = 2.0 * M * N * K / (ms * 1e6);
    std::cout << "Mine    " << M << "x" << K << " @ " << K << "x" << N
        << " -> " << ms << " ms (avg of " << kBenchIters << ") ("
        << gflops << " GFLOPS)\n";
}

void benchmark_cublas(int M, int N, int K)
{
    Tensor<__nv_bfloat16> A = Tensor<__nv_bfloat16>::randn({M, K});
    Tensor<__nv_bfloat16> B = Tensor<__nv_bfloat16>::randn({K, N});
    Tensor<__nv_bfloat16> C({M, N});
    Stream stream;

    // Warm-up (cublas handle is created the first time; cublasCreate
    // + algorithm heuristic selection have fixed cost).
    cublas_gemm_bf16(A, B, C, stream);
    stream.synchronize();

    float total_ms = 0.0f;
    for (int i = 0; i < kBenchIters; ++i)
    {
        total_ms += time_kernel_ms(stream, [&]
        {
            cublas_gemm_bf16(A, B, C, stream);
        });
    }
    float ms = total_ms / kBenchIters;

    double gflops = 2.0 * M * N * K / (ms * 1e6);
    std::cout << "cuBLAS  " << M << "x" << K << " @ " << K << "x" << N
        << " -> " << ms << " ms (avg of " << kBenchIters << ") ("
        << gflops << " GFLOPS)\n";
}

int main()
{
    try
    {
        const int M = 8192, N = 8192, K = 8192;

        // Correctness at a small size (fast, first sign of life)
        test_matmul_wmma_correctness(256, 256, 256);

        // Correctness at the SAME size being benchmarked. The relative
        // error in BF16 can grow with larger K, so this is what truly
        // backs the number you are going to publish.
        test_matmul_wmma_correctness(M, N, K);

        benchmark_matmul_wmma(M, N, K);
        benchmark_cublas(M, N, K);

        std::cout << "All tests passed.\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    return 0;
}