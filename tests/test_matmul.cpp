#include "core/tensor.hpp"
#include "core/event.hpp"
#include "core/backend_dispatch.hpp"
#include <cmath>
#include <cassert>
#include <iostream>

using namespace yadrakova::core;

// ---------------------------------------------------------------------------
// Correctness: compara tu kernel (Backend::Custom) contra cuBLAS
// (Backend::CuBLAS), ambos a traves de la MISMA llamada Tensor::matmul --
// ya no hay un cublas_gemm_bf16 a mano por separado, el backend es
// literalmente el unico parametro que cambia.
// ---------------------------------------------------------------------------
void test_matmul_correctness(int M, int N, int K, unsigned seed_a = 1, unsigned seed_b = 2)
{
    Tensor<__nv_bfloat16> A = Tensor<__nv_bfloat16>::randn({M, K}, seed_a);
    Tensor<__nv_bfloat16> B = Tensor<__nv_bfloat16>::randn({K, N}, seed_b);


    auto C_mine = A.matmul(B, Backend::Custom);

    auto C_ref  = A.matmul(B, Backend::CuBLAS);

    auto vec_mine = C_mine.to_vector();
    auto vec_ref = C_ref.to_vector();

    size_t mismatches = 0;
    float max_rel = 0.0f;
    float max_abs = 0.0f;

    // Tolerancias estandar para BF16. Para K grande (ej. 8192), la
    // acumulacion necesita un atol un poco mas relajado.
    const float atol = (K >= 4096) ? 2e-2f : 1e-2f;
    const float rtol = 1e-2f;

    for (size_t i = 0; i < vec_mine.size(); ++i)
    {
        float a = __bfloat162float(vec_mine[i]);
        float b = __bfloat162float(vec_ref[i]);
        float diff = std::fabs(a - b);

        // Mismo criterio que torch.allclose: diff <= atol + rtol * |b|
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
        std::cout << "  [FAIL] Elementos fuera de tolerancia: " << mismatches
                  << " / " << vec_mine.size() << " (" << mismatch_pct << "%)\n"
                  << "  Max error absoluto: " << max_abs
                  << " | Max error relativo: " << max_rel << "\n";
        assert(mismatches == 0);
    }
    else
    {
        std::cout << "  [PASS] Custom vs CuBLAS dentro de tolerancia BF16 (atol="
                  << atol << ", rtol=" << rtol << ").\n";
    }
}

// ---------------------------------------------------------------------------
// Prueba de que el dispatch funciona de verdad: con la fila de
// BackendCaps ya llena, Backend::Auto para MatMul/BF16 tiene que resolver
// solo a CuBLAS (esta antes que Custom en preference_order). Antes de este
// backend, esto hubiera resuelto a Custom.
// ---------------------------------------------------------------------------
void test_auto_dispatch_picks_cublas()
{
    auto resolution = OpsDispatch::resolve(Op::MatMul, DType::BF16, Backend::Auto);
    std::cout << "Backend::Auto para matmul/bf16 resolvio a: "
              << backend_name(resolution.backend) << "\n";
    assert(resolution.backend == Backend::CuBLAS);
    assert(!resolution.dtype_was_downgraded);
    std::cout << "  [PASS] Auto eligio CuBLAS sin que Tensor tuviera que pedirlo explicitamente.\n";
}

// ---------------------------------------------------------------------------
// Benchmark generico parametrizado por Backend -- antes tenias
// benchmark_matmul_wmma() y benchmark_cublas() casi duplicadas; ahora es
// la misma funcion, cambia el argumento.
// ---------------------------------------------------------------------------
constexpr int kBenchIters = 20;

void benchmark_matmul(const char* label, Backend backend, int M, int N, int K)
{
    Tensor<__nv_bfloat16> A = Tensor<__nv_bfloat16>::randn({M, K});
    Tensor<__nv_bfloat16> B = Tensor<__nv_bfloat16>::randn({K, N});


    // Warm-up: para CuBLAS crea el handle (cublasCreate) una sola vez via
    // BackendContext; para Custom no hace nada especial, pero mantiene el
    // mismo shape de codigo para las dos rutas.
    A.matmul(B, backend);

    float total_ms = 0.0f;
    for (int i = 0; i < kBenchIters; ++i)
    {
        total_ms += time_kernel_ms( [&]
        {
            A.matmul(B, backend);
        });
    }
    float ms = total_ms / kBenchIters;

    double gflops = 2.0 * M * N * K / (ms * 1e6);
    std::cout << label << "  " << M << "x" << K << " @ " << K << "x" << N
        << " -> " << ms << " ms (avg de " << kBenchIters << ") ("
        << gflops << " GFLOPS)\n";
}

int main()
{
    try
    {
        const int M = 8192, N = 8192, K = 8192;
        test_matmul_correctness(256, 256, 256);
        test_matmul_correctness(M, N, K);
        test_auto_dispatch_picks_cublas();
        benchmark_matmul("Mine  ", Backend::Custom, M, N, K);
        benchmark_matmul("cuBLAS", Backend::CuBLAS, M, N, K);
        std::cout << "Todos los tests pasaron.\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    return 0;
}