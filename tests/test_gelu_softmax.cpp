#include "core/tensor.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>

using namespace yadrakova::core;

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

    Tensor<float> A = Tensor<float>::randn({N});
    Tensor<float> C = A.gelu();

    auto h_in = A.to_vector();
    auto h_out = C.to_vector();

    float max_diff = 0.0f;
    for (int i = 0; i < N; ++i)
        max_diff = std::max(max_diff, std::fabs(cpu_gelu(h_in[i]) - h_out[i]));

    std::cout << "  gelu max_diff vs CPU: " << max_diff << "\n";
    assert(max_diff < 1e-4f);
    std::cout << "[OK] gelu_correctness\n";
}

void test_softmax_correctness() {
    const int rows = 4, cols = 16;

    Tensor<float> A = Tensor<float>::randn({rows, cols});
    Tensor<float> C = A.softmax();

    auto h_in = A.to_vector();
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
    std::cout << "[OK] softmax_correctness\n";
}

int main() {
    try {
        test_gelu_correctness();
        test_softmax_correctness();
        std::cout << "Todos los tests de gelu/softmax pasaron.\n";
    } catch (const std::exception& e) {
        std::cerr << "EXCEPCION: " << e.what() << "\n";
        return 1;
    }
    return 0;
}