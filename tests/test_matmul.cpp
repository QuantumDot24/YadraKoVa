#include "core/tensor.h"
#include "core/stream.h"
#include "core/event.h"
#include "kernels/registry.h"
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <iostream>

using namespace yadrakova::core;
using namespace yadrakova::kernels;

// Referencia CPU, ingenua a proposito -- es la fuente de verdad contra
// la que comparamos el resultado del kernel GPU.
std::vector<float> cpu_matmul(const std::vector<float>& A, const std::vector<float>& B,
                               int M, int N, int K) {
    std::vector<float> C(M * N, 0.0f);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
    return C;
}

void test_matmul_correctness() {
    const int M = 64, N = 64, K = 64;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_A(M * K), h_B(K * N);
    for (auto& v : h_A) v = dist(rng);
    for (auto& v : h_B) v = dist(rng);

    Tensor<float> d_A({M, K});
    Tensor<float> d_B({K, N});
    Tensor<float> d_C({M, N});

    cudaMemcpy(d_A.data(), h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.data(), h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice);

    CUfunction fn = KernelRegistry::instance().get_function("matmul", Arch::SM_86, DType::FP32);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    float* A_ptr = d_A.data();
    float* B_ptr = d_B.data();
    float* C_ptr = d_C.data();
    void* args[] = { &A_ptr, &B_ptr, &C_ptr, (void*)&M, (void*)&N, (void*)&K };

    CUresult err = cuLaunchKernel(fn, grid.x, grid.y, 1, block.x, block.y, 1,
                                   0, nullptr, args, nullptr);
    assert(err == CUDA_SUCCESS);
    cudaDeviceSynchronize();

    std::vector<float> h_C(M * N);
    cudaMemcpy(h_C.data(), d_C.data(), h_C.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> ref = cpu_matmul(h_A, h_B, M, N, K);

    float max_diff = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_diff = std::max(max_diff, std::fabs(ref[i] - h_C[i]));

    std::cout << "  max_diff vs CPU: " << max_diff << "\n";
    assert(max_diff < 1e-3f); // tolerancia razonable para fp32 con K=64
    std::cout << "[OK] matmul_correctness\n";
}

void test_matmul_timing() {
    const int M = 512, N = 512, K = 512;
    Tensor<float> d_A({M, K});
    Tensor<float> d_B({K, N});
    Tensor<float> d_C({M, N});
    cudaMemset(d_A.data(), 0, d_A.numel() * sizeof(float));
    cudaMemset(d_B.data(), 0, d_B.numel() * sizeof(float));

    CUfunction fn = KernelRegistry::instance().get_function("matmul", Arch::SM_86, DType::FP32);
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    float* A_ptr = d_A.data();
    float* B_ptr = d_B.data();
    float* C_ptr = d_C.data();
    void* args[] = { &A_ptr, &B_ptr, &C_ptr, (void*)&M, (void*)&N, (void*)&K };

    Stream s;
    // warm-up: la primera ejecucion incluye overhead de lazy-init del
    // driver, no representativo -- lo descartamos.
    cuLaunchKernel(fn, grid.x, grid.y, 1, block.x, block.y, 1, 0, s.raw(), args, nullptr);
    s.synchronize();

    float ms = time_kernel_ms(s, [&]() {
        cuLaunchKernel(fn, grid.x, grid.y, 1, block.x, block.y, 1, 0, s.raw(), args, nullptr);
    });
    std::cout << "[INFO] matmul 512x512x512 (naive): " << ms << " ms "
              << "(no importa que sea lento hoy, es baseline de correctitud)\n";
}

int main() {
    test_matmul_correctness();
    test_matmul_timing();
    std::cout << "Todos los tests de matmul pasaron.\n";
    return 0;
}