#include "core/tensor.hpp"
#include "core/event.hpp"
#include <cublas_v2.h>
#include <vector>
#include <cmath>
#include <cassert>
#include <chrono>
#include <iostream>

using namespace yadrakova::core;

// ---------------------------------------------------------------------------
// CuBLAS helper (C = A * B, todo en bf16)
// ---------------------------------------------------------------------------
void cublas_gemm_bf16(const Tensor<__nv_bfloat16>& A,
                      const Tensor<__nv_bfloat16>& B,
                      Tensor<__nv_bfloat16>& C,
                      Stream& stream)
{
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
    }
    cublasSetStream(handle, stream.raw());

    const float alpha = 1.0f, beta = 0.0f;
    // A y B en row-major. cublas espera column-major, así que hacemos
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
// Corrección: compara tu kernel con cuBLAS en tamaño pequeño
// ---------------------------------------------------------------------------
void test_matmul_wmma_correctness()
{
    const int M = 256, N = 256, K = 256;

    Tensor<__nv_bfloat16> A = Tensor<__nv_bfloat16>::randn({M, K});
    Tensor<__nv_bfloat16> B = Tensor<__nv_bfloat16>::randn({K, N});

    Stream stream;
    auto C_mine = A.matmul(B, stream);   // asume que matmul devuelve Tensor
    stream.synchronize();

    Tensor<__nv_bfloat16> C_ref({M, N});
    cublas_gemm_bf16(A, B, C_ref, stream);
    stream.synchronize();

    auto vec_mine = C_mine.to_vector();
    auto vec_ref  = C_ref.to_vector();

    size_t mismatches = 0;
    float max_rel = 0.0f;
    for (size_t i = 0; i < vec_mine.size(); ++i) {
        if (vec_mine[i] != vec_ref[i]) {
            ++mismatches;
            float a = __bfloat162float(vec_mine[i]);
            float b = __bfloat162float(vec_ref[i]);
            float diff = std::fabs(a - b);
            float rel = diff / std::max(std::fabs(b), 1e-6f);
            max_rel = std::max(max_rel, rel);
        }
    }

    std::cout << "Corrección " << M << "x" << K << " @ " << K << "x" << N << "\n";
    if (mismatches > 0) {
        std::cout << "  elementos diferentes: " << mismatches
                  << " (error relativo máximo: " << max_rel << ")\n";
        assert(max_rel < 1e-2f);
    } else {
        std::cout << "  resultados idénticos bit‑a‑bit con cuBLAS.\n";
    }
    std::cout << "[OK] matmul_wmma correctness\n";
}

// ---------------------------------------------------------------------------
// Benchmark: mide solo tiempo (sin transferencias a host)
// ---------------------------------------------------------------------------
void benchmark_matmul_wmma(int M, int N, int K)
{
    Tensor<__nv_bfloat16> A = Tensor<__nv_bfloat16>::randn({M, K});
    Tensor<__nv_bfloat16> B = Tensor<__nv_bfloat16>::randn({K, N});

    Stream profile_stream;
    float ms = time_kernel_ms(profile_stream, [&] {
        A.matmul(B, profile_stream);
    });

    double gflops = 2.0 * M * N * K / (ms * 1e6);
    std::cout << "Benchmark " << M << "x" << K << " @ " << K << "x" << N
              << " -> " << ms << " ms (" << gflops << " GFLOPS)\n";
}
void benchmark_cublas(int M, int N, int K) {
    Tensor<__nv_bfloat16> A = Tensor<__nv_bfloat16>::randn({M, K});
    Tensor<__nv_bfloat16> B = Tensor<__nv_bfloat16>::randn({K, N});
    Tensor<__nv_bfloat16> C({M, N});
    Stream stream;

    auto start = std::chrono::high_resolution_clock::now();
    cublas_gemm_bf16(A, B, C, stream);
    stream.synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(end - start).count();
    double gflops = 2.0 * M * N * K / (ms * 1e6);
    std::cout << "cuBLAS   " << M << "x" << K << " @ " << K << "x" << N
              << " -> " << ms << " ms (" << gflops << " GFLOPS)\n";
}

int main()
{
    try {
        test_matmul_wmma_correctness();
        benchmark_matmul_wmma(8192, 8192, 8192);
        benchmark_cublas(8192, 8192, 8192);
        std::cout << "Todos los tests pasaron.\n";
    } catch (const std::exception& e) {
        std::cerr << "EXCEPCION: " << e.what() << "\n";
        return 1;
    }
    return 0;
}