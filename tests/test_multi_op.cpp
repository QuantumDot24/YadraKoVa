#include "core/tensor.hpp"
#include "core/graph_manager.hpp"
#include "core/event.hpp"
#include "telemetry/telemetry.hpp"
#include "telemetry/telemetry_mqtt_publisher.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

#include "telemetry/telemetry_scope.hpp"

using namespace yadrakova::core;

struct ReplayTelemetry
{
    int replay_idx;
    float ms;
};

void print_telemetry(const std::string& label, const std::vector<ReplayTelemetry>& log)
{
    float total = 0.0f;
    for (const auto& e : log) total += e.ms;
    std::cout << "[" << label << "] " << log.size() << " replays, "
        << "avg " << (total / log.size()) << " ms, "
        << "primer replay " << log.front().ms << " ms, "
        << "ultimo replay " << log.back().ms << " ms\n";
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

        // Warmup / compilación JIT previa
        out_matmul = A.matmul(B, stream);
        out_gelu = out_matmul.gelu(stream);
        out_softmax = out_gelu.softmax(stream);
        stream.synchronize();

        // Iniciar sesión de telemetría explícitamente
        manager.telemetry().start_session();

        // --- Benchmark SIN graph capture ---
        std::vector<ReplayTelemetry> no_capture_log;
        no_capture_log.reserve(kReplays);

        for (int i = 0; i < kReplays; ++i)
        {
            // 1. Mide GPU real para Matmul
            float ms_matmul = time_kernel_ms(stream, [&] {
                out_matmul = A.matmul(B, stream);
            });
            manager.telemetry().record("matmul", OpKind::Gemm, ms_matmul, stream.raw(), false);

            // 2. Mide GPU real para GELU
            float ms_gelu = time_kernel_ms(stream, [&] {
                out_gelu = out_matmul.gelu(stream);
            });
            manager.telemetry().record("gelu", OpKind::Elementwise, ms_gelu, stream.raw(), false);

            // 3. Mide GPU real para Softmax
            float ms_softmax = time_kernel_ms(stream, [&] {
                out_softmax = out_gelu.softmax(stream);
            });
            manager.telemetry().record("softmax", OpKind::Reduction, ms_softmax, stream.raw(), false);

            no_capture_log.push_back({i, ms_matmul + ms_gelu + ms_softmax});
        }
        print_telemetry("sin_capture", no_capture_log);

        // --- Captura ---
        manager.capture("toy_pipeline", /*strict=*/true, [&]
        {
            out_matmul = A.matmul(B, stream);
            out_gelu = out_matmul.gelu(stream);
            out_softmax = out_gelu.softmax(stream);
        });
        assert(manager.get("toy_pipeline").is_instantiated());

        // --- Benchmark CON graph replay ---
        std::vector<ReplayTelemetry> replay_log;
        replay_log.reserve(kReplays);
        for (int i = 0; i < kReplays; ++i)
        {
            float ms = time_kernel_ms(stream, [&]
            {
                manager.launch("toy_pipeline");
            });
            manager.telemetry().record("toy_pipeline", OpKind::GraphReplay, ms,
                                       stream.raw(), /*from_graph_replay=*/true);
            replay_log.push_back({i, ms});
        }
        print_telemetry("graph_replay", replay_log);

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