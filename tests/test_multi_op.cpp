#include "core/tensor.hpp"
#include "core/graph_manager.hpp"
#include "telemetry/telemetry.hpp"
#include "telemetry/telemetry_scope.hpp"
#include "telemetry/telemetry_mqtt_publisher.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

using namespace yadrakova::core;

// Reconstruye el resumen de consola a partir de los records ya resueltos
// (duration_ms real, post resolve_pending()) en vez de time_kernel_ms.
void print_telemetry_from_records(const std::string& label_filter,
                                   const std::vector<TelemetryRecord>& records,
                                   size_t expected_count)
{
    std::vector<float> durations;
    durations.reserve(expected_count);

    // Si label_filter es "toy_pipeline" (graph replay), cada record ya es
    // un replay completo. Si es "" (sin capture), agrupamos matmul+gelu+softmax
    // de 3 en 3 para reconstruir el total por iteracion.
    if (label_filter == "toy_pipeline")
    {
        for (const auto& r : records)
        {
            if (r.label == "toy_pipeline") durations.push_back(r.duration_ms);
        }
    }
    else
    {
        float acc = 0.0f;
        int count_in_group = 0;
        for (const auto& r : records)
        {
            if (r.label == "matmul" || r.label == "gelu" || r.label == "softmax")
            {
                acc += r.duration_ms;
                ++count_in_group;
                if (count_in_group == 3)
                {
                    durations.push_back(acc);
                    acc = 0.0f;
                    count_in_group = 0;
                }
            }
        }
    }

    if (durations.empty())
    {
        std::cout << "[" << label_filter << "] sin datos resueltos\n";
        return;
    }

    float total = 0.0f;
    for (float d : durations) total += d;

    std::cout << "[" << (label_filter.empty() ? "sin_capture" : label_filter) << "] "
        << durations.size() << " replays, "
        << "avg " << (total / durations.size()) << " ms, "
        << "primer replay " << durations.front() << " ms, "
        << "ultimo replay " << durations.back() << " ms\n";
}

void check_pipeline_replay_matches_direct(
    const std::vector<__nv_bfloat16>& vec_graph,
    const std::vector<__nv_bfloat16>& vec_ref,
    const std::string& label)
{
    assert(vec_graph.size() == vec_ref.size());

    const float atol = 1e-2f;
    const float rtol = 1e-2f;

    size_t mismatches = 0;
    float max_abs = 0.0f, max_rel = 0.0f;

    for (size_t i = 0; i < vec_graph.size(); ++i)
    {
        float a = __bfloat162float(vec_graph[i]);
        float b = __bfloat162float(vec_ref[i]);
        float diff = std::fabs(a - b);
        float allowed_tol = atol + rtol * std::fabs(b);

        if (diff > allowed_tol)
        {
            ++mismatches;
            max_abs = std::max(max_abs, diff);
            max_rel = std::max(max_rel, diff / std::max(std::fabs(b), 1e-6f));
        }
    }

    std::cout << "  [" << label << "] correctness graph_replay vs direct: ";
    if (mismatches > 0)
    {
        std::cout << "[FAIL] " << mismatches << " / " << vec_graph.size()
            << " fuera de tolerancia | max_abs=" << max_abs
            << " max_rel=" << max_rel << "\n";
        assert(mismatches == 0);
    }
    else
    {
        std::cout << "[PASS]\n";
    }
}

int main()
{
    try
    {
        MemoryPool pool(0);
        GraphManager manager(pool);

        const int64_t M = 512, K = 512, N = 16;
        constexpr int kReplays = 50;

        Tensor<__nv_bfloat16> A = Tensor<__nv_bfloat16>::randn({M, K}, 1, pool);
        Tensor<__nv_bfloat16> B = Tensor<__nv_bfloat16>::randn({K, N}, 2, pool);

        Tensor<__nv_bfloat16> out_matmul({M, N}, pool);
        Tensor<__nv_bfloat16> out_gelu({M, N}, pool);
        Tensor<__nv_bfloat16> out_softmax({M, N}, pool);

        Stream& stream = manager.stream_for("toy_pipeline");
        cudaStream_t raw_stream = stream.raw();

        // Warmup / compilacion JIT previa
        out_matmul = A.matmul(B, stream);
        out_gelu = out_matmul.gelu(stream);
        out_softmax = out_gelu.softmax(stream);
        stream.synchronize();

        manager.telemetry().start_session();

        // --- Benchmark SIN graph capture (via TelemetryScope, sin sync en hot path) ---
        for (int i = 0; i < kReplays; ++i)
        {
            {
                TelemetryScope scope(&manager.telemetry(), "matmul", OpKind::Gemm, raw_stream);
                out_matmul = A.matmul(B, stream);
            }
            {
                TelemetryScope scope(&manager.telemetry(), "gelu", OpKind::Elementwise, raw_stream);
                out_gelu = out_matmul.gelu(stream);
            }
            {
                TelemetryScope scope(&manager.telemetry(), "softmax", OpKind::Reduction, raw_stream);
                out_softmax = out_gelu.softmax(stream);
            }
        }

        // --- Captura ---
        manager.capture("toy_pipeline", /*strict=*/true, [&]
        {
            out_matmul = A.matmul(B, stream);
            out_gelu = out_matmul.gelu(stream);
            out_softmax = out_gelu.softmax(stream);
        });
        assert(manager.get("toy_pipeline").is_instantiated());

        // Warmup del grafo: el primer launch tras cudaGraphInstantiate paga
        // el costo de "subir" la topologia al device (graph upload), algo
        // que no se repite en replays subsecuentes. Igual que arriba con
        // matmul/gelu/softmax, se descarta y no entra a la telemetria medida.
        manager.launch("toy_pipeline");
        stream.synchronize();

        // --- Benchmark CON graph replay (via TelemetryScope) ---
        for (int i = 0; i < kReplays; ++i)
        {
            TelemetryScope scope(&manager.telemetry(), "toy_pipeline", OpKind::GraphReplay,
                                  raw_stream, raw_stream, /*from_graph_replay=*/true);
            manager.launch("toy_pipeline");
        }

        // Un solo punto de sincronizacion: resuelve TODOS los eventos pendientes
        // (sin_capture + graph_replay) y los vuelca a records_.
        manager.telemetry().resolve_pending();

        // Reconstruye los logs de consola a partir de los records ya resueltos.
        print_telemetry_from_records("", manager.telemetry().records(), kReplays);
        print_telemetry_from_records("toy_pipeline", manager.telemetry().records(), kReplays);

        // --- Correctness: graph replay vs ejecucion directa ---
        auto vec_graph = out_softmax.to_vector();

        Tensor<__nv_bfloat16> ref_matmul = A.matmul(B, stream);
        Tensor<__nv_bfloat16> ref_gelu = ref_matmul.gelu(stream);
        Tensor<__nv_bfloat16> ref_softmax = ref_gelu.softmax(stream);
        stream.synchronize();
        auto vec_ref = ref_softmax.to_vector();

        check_pipeline_replay_matches_direct(vec_graph, vec_ref, "toy_pipeline");

        std::cout << "Toy pipeline test OK.\n";

        // --- Telemetria: publicar a MQTT para la app Android ---
        std::cout << "\n--- telemetry (publicando a MQTT) ---\n";
        TelemetryMqttPublisher publisher("tcp://localhost:1883", "yadrakova_benchmark");
        bool ok = publisher.publish_lines(manager.telemetry().export_all_as_json_lines(),
                                          "yadrakova/telemetry");
        std::cout << (ok ? "Telemetria publicada OK.\n" : "Fallo publicando telemetria (broker caido?).\n");
    }
    catch (const std::exception& e)
    {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    return 0;
}