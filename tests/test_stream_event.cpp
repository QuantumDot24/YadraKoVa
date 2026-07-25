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

    float ms = time_kernel_ms(s, [&]() {
        cudaMemsetAsync(t.data(), 0, t.numel() * sizeof(float), s.raw());
    });

    assert(ms >= 0.0f); // el tiempo real variara por hardware, solo validamos que se pudo medir
    std::cout << "[OK] event_timing (memset de 1024x1024 tomo " << ms << " ms)\n";
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
    test_stream_create_and_sync();
    test_event_timing();
    test_stream_is_done_polling();
    std::cout << "Todos los tests de stream/event pasaron.\n";
    return 0;
}