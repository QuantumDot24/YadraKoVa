#include "core/stream.h"
#include "core/event.h"
#include "core/tensor.h"
#include <cassert>
#include <iostream>

using namespace yadrakova::core;

// Un "kernel" muy simple solo para tener algo que medir -- cudaMemset
// no es un kernel real, pero corre en GPU y sirve para probar timing
// sin necesitar todavia kernels/registry.h.
void test_stream_create_and_sync() {
    Stream s;
    assert(s.raw() != nullptr);
    s.synchronize(); // stream vacio, debe retornar de inmediato
    std::cout << "[OK] stream_create_and_sync\n";
}

void test_event_timing() {
    Stream s;
    Tensor<float> t({1024, 1024}); // ~4MB, suficiente para que el memset tome tiempo medible

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
    s.synchronize(); // forzamos a que termine antes de preguntar
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