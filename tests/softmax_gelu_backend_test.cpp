#include "core/tensor.hpp"
#include "core/event.hpp"
#include "core/ops_dispatch.hpp"
#include <cmath>
#include <cassert>
#include <iostream>

using namespace yadrakova::core;

constexpr int kBenchIters = 20;

// ---------------------------------------------------------------------------
// Softmax: comparacion real Custom vs cuDNN, mismo patron que el test de
// matmul -- el backend es el unico parametro que cambia.
// ---------------------------------------------------------------------------
void test_softmax_correctness(int rows, int cols, unsigned seed = 7)
{
    Tensor<__nv_bfloat16> X = Tensor<__nv_bfloat16>::randn({rows, cols}, seed);

    auto Y_mine = X.softmax(Backend::Custom);

    auto Y_cudnn = X.softmax(Backend::CuDNN);

    auto vec_mine = Y_mine.to_vector();
    auto vec_ref  = Y_cudnn.to_vector();

    size_t mismatches = 0;
    float max_abs = 0.0f, max_rel = 0.0f;
    const float atol = 1e-2f, rtol = 1e-2f;

    for (size_t i = 0; i < vec_mine.size(); ++i)
    {
        float a = __bfloat162float(vec_mine[i]);
        float b = __bfloat162float(vec_ref[i]);
        float diff = std::fabs(a - b);
        float allowed = atol + rtol * std::fabs(b);
        if (diff > allowed)
        {
            ++mismatches;
            max_abs = std::max(max_abs, diff);
            max_rel = std::max(max_rel, diff / std::max(std::fabs(b), 1e-6f));
        }
    }

    double pct = 100.0 * static_cast<double>(mismatches) / vec_mine.size();
    std::cout << "Softmax correctness " << rows << "x" << cols << "\n";
    if (mismatches > 0)
    {
        std::cout << "  [FAIL] Elementos fuera de tolerancia: " << mismatches
            << " / " << vec_mine.size() << " (" << pct << "%)\n"
            << "  Max error absoluto: " << max_abs << " | Max error relativo: " << max_rel << "\n";
        assert(mismatches == 0);
    }
    else
    {
        std::cout << "  [PASS] Custom vs cuDNN dentro de tolerancia BF16 (atol="
            << atol << ", rtol=" << rtol << ").\n";
    }
}

void test_softmax_auto_dispatch_picks_cudnn()
{
    auto resolution = OpsDispatch::resolve(Op::Softmax, DType::BF16, Backend::Auto);
    std::cout << "Backend::Auto para softmax/bf16 resolvio a: "
        << backend_name(resolution.backend) << "\n";
    assert(resolution.backend == Backend::CuDNN);
    std::cout << "  [PASS] Auto eligio cuDNN sin que Tensor tuviera que pedirlo explicitamente.\n";
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
              << " -> " << ms << " ms (avg de " << kBenchIters << ")\n";
}

// ---------------------------------------------------------------------------
// Gelu: SOLO Custom. cuDNN no tiene GELU en su API de activaciones estable
// (cudnnActivationMode_t no incluye GELU -- solo existe via la Graph API,
// que no esta implementada todavia, ver nota en backend_caps.hpp). Este
// benchmark deja el numero de referencia de tu kernel a mano, sin comparar
// contra nada -- no hay nada real contra que comparar todavia.
// ---------------------------------------------------------------------------
void benchmark_gelu_custom_only(int n)
{
    Tensor<__nv_bfloat16> X = Tensor<__nv_bfloat16>::randn({n});

    X.gelu(Backend::Custom);

    float total_ms = 0.0f;
    for (int i = 0; i < kBenchIters; ++i)
        total_ms += time_kernel_ms([&] { X.gelu(Backend::Custom); });
    float ms = total_ms / kBenchIters;
    std::cout << "Mine  gelu n=" << n << " -> " << ms << " ms (avg de " << kBenchIters
        << ") -- sin comparacion cuDNN (no soportado, ver backend_caps.hpp)\n";
}

int main()
{
    try
    {
        const int rows = 4096, cols = 4096;

        test_softmax_correctness(256, 256);
        test_softmax_correctness(rows, cols);
        test_softmax_auto_dispatch_picks_cudnn();

        benchmark_softmax("Mine ", Backend::Custom, rows, cols);
        benchmark_softmax("cuDNN", Backend::CuDNN, rows, cols);

        benchmark_gelu_custom_only(rows * cols);

        std::cout << "Todos los tests pasaron.\n";
        std::cout << "(LayerNorm no esta en este test -- no hay Tensor::layer_norm "
            "ni kernel custom todavia, ver la respuesta para las opciones.)\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
