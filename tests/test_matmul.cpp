#include "core/tensor.h"
#include "core/cuda_error.h"
#include "kernels/registry.h"
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <iostream>

using namespace yadrakova::core;
using namespace yadrakova::kernels;

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
    std::cout << "-> entrando a test_matmul_correctness\n" << std::flush;

    const int M = 64, N = 64, K = 64;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_A(M * K), h_B(K * N);
    for (auto& v : h_A) v = dist(rng);
    for (auto& v : h_B) v = dist(rng);

    std::cout << "-> creando tensores\n" << std::flush;
    Tensor<float> d_A({M, K});
    Tensor<float> d_B({K, N});
    Tensor<float> d_C({M, N});

    std::cout << "-> copiando a device\n" << std::flush;
    CUDA_CHECK(cudaMemcpy(d_A.data(), h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B.data(), h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "-> obteniendo kernel de registry\n" << std::flush;
    CUfunction fn = KernelRegistry::instance().get_function("matmul", Arch::SM_86, DType::FP32);
    std::cout << "-> kernel obtenido, fn=" << fn << "\n" << std::flush;

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    float* A_ptr = d_A.data();
    float* B_ptr = d_B.data();
    float* C_ptr = d_C.data();
    void* args[] = { &A_ptr, &B_ptr, &C_ptr, (void*)&M, (void*)&N, (void*)&K };

    std::cout << "-> lanzando kernel\n" << std::flush;
    CU_CHECK(cuLaunchKernel(fn, grid.x, grid.y, 1, block.x, block.y, 1,
                             0, nullptr, args, nullptr));
    std::cout << "-> kernel lanzado, sincronizando\n" << std::flush;
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_C(M * N);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C.data(), h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> ref = cpu_matmul(h_A, h_B, M, N, K);

    float max_diff = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_diff = std::max(max_diff, std::fabs(ref[i] - h_C[i]));

    std::cout << "  max_diff vs CPU: " << max_diff << "\n";
    assert(max_diff < 1e-3f);
    std::cout << "[OK] matmul_correctness\n";
}

int main() {
    try {
        test_matmul_correctness();
        std::cout << "Todos los tests de matmul pasaron.\n";
    } catch (const std::exception& e) {
        std::cerr << "EXCEPCION: " << e.what() << "\n";
        return 1;
    }
    return 0;
}