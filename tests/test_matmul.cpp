#include "core/tensor.hpp"
#include "core/event.hpp"
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>

using namespace yadrakova::core;

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

    Tensor<float> A = Tensor<float>::randn({M, K});
    Tensor<float> B = Tensor<float>::randn({K, N});

    Tensor<float> C({M, N});  // preasignado afuera, para medir solo el kernel

    Stream profile_stream;
    float ms = time_kernel_ms(profile_stream, [&] {
        C = A.matmul(B, profile_stream);
    });
    profile_stream.synchronize();

    std::cout << "matmul " << M << "x" << K << " @ " << K << "x" << N
              << " -> " << ms << " ms\n";

    auto h_A = A.to_vector();
    auto h_B = B.to_vector();
    auto h_C = C.to_vector();

    auto ref = cpu_matmul(h_A, h_B, M, N, K);

    float max_diff = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_diff = std::max(max_diff, std::fabs(ref[i] - h_C[i]));

    std::cout << "max_diff vs CPU: " << max_diff << "\n";
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