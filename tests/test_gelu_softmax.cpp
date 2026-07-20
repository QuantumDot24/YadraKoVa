#include "core/tensor.hpp"
#include "core/stream.hpp"
#include "kernels/registry.hpp"
#include "kernels/embedded/gelu_embedded.cuh"
#include "kernels/embedded/softmax_embedded.cuh"
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <iostream>

using namespace yadrakova::core;
using namespace yadrakova::kernels;

// Referencias CPU -- fuente de verdad.
float cpu_gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
}

std::vector<float> cpu_softmax_row(const std::vector<float>& row) {
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

void test_gelu_correctness() {
    const int N = 1024;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    std::vector<float> h_in(N);
    for (auto& v : h_in) v = dist(rng);

    Tensor<float> d_in({N});
    Tensor<float> d_out({N});
    cudaMemcpy(d_in.data(), h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    CUfunction fn = KernelRegistry::instance().get_function("gelu", Arch::SM_86, DType::FP32);

    float* in_ptr = d_in.data();
    float* out_ptr = d_out.data();
    void* args[] = { &in_ptr, &out_ptr, (void*)&N };

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr);
    cudaDeviceSynchronize();

    std::vector<float> h_out(N);
    cudaMemcpy(h_out.data(), d_out.data(), N * sizeof(float), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    for (int i = 0; i < N; ++i)
        max_diff = std::max(max_diff, std::fabs(cpu_gelu(h_in[i]) - h_out[i]));

    std::cout << "  gelu max_diff vs CPU: " << max_diff << "\n";
    assert(max_diff < 1e-4f);
    std::cout << "[OK] gelu_correctness\n";
}

void test_softmax_correctness() {
    const int rows = 4, cols = 16; // cols <= 32 para esta version simple
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    std::vector<float> h_in(rows * cols);
    for (auto& v : h_in) v = dist(rng);

    Tensor<float> d_in({rows, cols});
    Tensor<float> d_out({rows, cols});
    cudaMemcpy(d_in.data(), h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice);

    CUfunction fn = KernelRegistry::instance().get_function("softmax", Arch::SM_86, DType::FP32);

    float* in_ptr = d_in.data();
    float* out_ptr = d_out.data();
    void* args[] = { &in_ptr, &out_ptr, (void*)&rows, (void*)&cols };

    // Un block por fila, un warp (32 threads) por block.
    cuLaunchKernel(fn, rows, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr);
    cudaDeviceSynchronize();

    std::vector<float> h_out(rows * cols);
    cudaMemcpy(h_out.data(), d_out.data(), h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    for (int r = 0; r < rows; ++r) {
        std::vector<float> row(h_in.begin() + r * cols, h_in.begin() + (r + 1) * cols);
        std::vector<float> ref = cpu_softmax_row(row);
        for (int c = 0; c < cols; ++c)
            max_diff = std::max(max_diff, std::fabs(ref[c] - h_out[r * cols + c]));
    }

    std::cout << "  softmax max_diff vs CPU: " << max_diff << "\n";
    assert(max_diff < 1e-5f);
    std::cout << "[OK] softmax_correctness\n";
}

int main() {
    test_gelu_correctness();
    test_softmax_correctness();
    std::cout << "Todos los tests de gelu/softmax pasaron.\n";
    return 0;
}