#include "core/stream.hpp"
#include "core/event.hpp"
#include "core/tensor.hpp"
#include <cassert>
#include <iostream>

using namespace yadrakova::core;

void test_stream_create_and_sync() {
    Stream s;
    assert(s.raw() != nullptr);
    s.synchronize();
    std::cout << "[OK] stream_create_and_sync\n";
}

void test_event_timing() {
    Stream s;
    Tensor<float> t({1024, 1024});

    // CORRECCIÓN: El orden de los parámetros es (func, stream)
    float ms = time_kernel_ms([&]() {
        cudaMemsetAsync(t.data(), 0, t.numel() * sizeof(float), s.raw());
    }, s);

    assert(ms >= 0.0f); // El tiempo real variará por hardware, solo validamos que se pudo medir
    std::cout << "[OK] event_timing (memset de 1024x1024 tomó " << ms << " ms)\n";
}

void test_stream_is_done_polling() {
    Stream s;
    Tensor<float> t({256, 256});
    cudaMemsetAsync(t.data(), 0, t.numel() * sizeof(float), s.raw());
    s.synchronize();
    assert(s.is_done() == true);
    std::cout << "[OK] stream_is_done_polling\n";
}

int main() {
    std::cout << "=== SUITE DE TESTS: STREAM & EVENT ===\n";
    test_stream_create_and_sync();
    test_event_timing();
    test_stream_is_done_polling();
    std::cout << "Todos los tests de stream/event pasaron correctamente.\n\n";
    return 0;
}