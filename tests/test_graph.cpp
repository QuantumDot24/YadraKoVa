#include "core/graph.h"
#include "core/graph_manager.h"
#include "core/tensor.h"
#include <cassert>
#include <iostream>

using namespace yadrakova::core;

void test_basic_capture_and_launch() {
    Stream stream;
    MemoryPool pool(0);

    // Preasignamos ANTES de capturar -- regla de oro.
    Tensor<float> t({1024, 1024}, pool);

    Graph g("simple_memset");
    g.begin_capture(stream, pool, /*strict=*/true);
    cudaMemsetAsync(t.data(), 0, t.numel() * sizeof(float), stream.raw());
    g.end_capture(stream);

    assert(g.is_instantiated());
    g.launch(stream);
    stream.synchronize();

    std::cout << "[OK] basic_capture_and_launch\n";
}
void test_strict_mode_catches_allocation_during_capture() {
    Stream stream;
    MemoryPool pool(0);

    Graph g("bad_capture_allocates_mid_graph");
    g.begin_capture(stream, pool, /*strict=*/true);

    bool threw = false;
    try {
        // ERROR INTENCIONAL: CUDA mismo prohibe cudaMalloc durante
        // una captura activa -- esto lanza ANTES de llegar a
        // end_capture(), no despues.
        Tensor<float> t_created_mid_capture({512, 512}, pool);
        cudaMemsetAsync(t_created_mid_capture.data(), 0,
                         t_created_mid_capture.numel() * sizeof(float), stream.raw());
    } catch (const std::runtime_error& e) {
        threw = true;
        std::cout << "  (excepcion esperada al intentar allocar: " << e.what() << ")\n";
        g.abort_capture(stream);
    }
    assert(threw);

    // La captura quedo "colgada" (begin_capture ya se llamo, nunca
    // se llamo end_capture exitosamente) -- hay que cerrarla para
    // no dejar el stream en un estado invalido para el resto del programa.
    cudaGraph_t dangling_graph = nullptr;
    cudaStreamEndCapture(stream.raw(), &dangling_graph);
    if (dangling_graph) cudaGraphDestroy(dangling_graph);

    std::cout << "[OK] strict_mode_catches_allocation_during_capture\n";
}

void test_graph_manager_multiple_named_graphs() {
    Stream stream;
    MemoryPool pool(0);

    Tensor<float> g_tensor({256, 256}, pool);
    Tensor<float> d_tensor({256, 256}, pool);

    GraphManager manager(pool);

    Graph& g_forward = manager.get_or_create("G_forward");
    g_forward.begin_capture(stream, pool);
    cudaMemsetAsync(g_tensor.data(), 0, g_tensor.numel() * sizeof(float), stream.raw());
    g_forward.end_capture(stream);

    Graph& d_forward = manager.get_or_create("D_forward");
    d_forward.begin_capture(stream, pool);
    cudaMemsetAsync(d_tensor.data(), 0, d_tensor.numel() * sizeof(float), stream.raw());
    d_forward.end_capture(stream);

    assert(manager.count() == 2);
    assert(manager.has("G_forward"));
    assert(manager.has("D_forward"));

    // Simula alternancia GAN: 2 pasos de D por cada paso de G.
    manager.launch("D_forward", stream);
    manager.launch("D_forward", stream);
    manager.launch("G_forward", stream);
    stream.synchronize();

    std::cout << "[OK] graph_manager_multiple_named_graphs\n";
}

int main() {
    test_basic_capture_and_launch();
    test_strict_mode_catches_allocation_during_capture();
    test_graph_manager_multiple_named_graphs();
    std::cout << "Todos los tests de graph pasaron.\n";
    return 0;
}