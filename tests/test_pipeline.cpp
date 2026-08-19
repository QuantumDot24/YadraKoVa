#include "core/tensor.hpp"
#include "core/graph_manager.hpp"
#include "telemetry/kova_metry.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

using namespace yadrakova::core;

// ---------------------------------------------------------------------------
// Helper: compare two BF16 vectors (graph replay output vs direct execution)
// using the same torch.allclose-style tolerance.
// ---------------------------------------------------------------------------
static void check_pipeline_replay_matches_direct(
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
        float allowed = atol + rtol * std::fabs(b);

        if (diff > allowed)
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
                  << " out of tolerance | max_abs=" << max_abs
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
        std::cout << "=== MULTI-OP PIPELINE TEST SUITE ===\n\n";

        GraphManager manager;
        const int64_t M = 512, K = 512, N = 16;
        constexpr int kReplays = 50;

        Tensor<__nv_bfloat16> A = Tensor<__nv_bfloat16>::randn({M, K}, 1, manager.pool());
        Tensor<__nv_bfloat16> B = Tensor<__nv_bfloat16>::randn({K, N}, 2, manager.pool());

        Tensor<__nv_bfloat16> out_matmul ({M, N}, manager.pool());
        Tensor<__nv_bfloat16> out_gelu   ({M, N}, manager.pool());
        Tensor<__nv_bfloat16> out_softmax({M, N}, manager.pool());

        Stream& stream = manager.stream_for("toy_pipeline");

        // --- Baseline direct execution (no graph capture) ---
        out_matmul  = A.matmul (B, Backend::Custom, stream);
        out_gelu    = out_matmul.gelu   (Backend::Custom, stream);
        out_softmax = out_gelu  .softmax(Backend::Custom, stream);
        stream.synchronize();

        manager.telemetry().start_session();

        manager.benchmark("no_capture", 3, kReplays, [&](int)
        {
            manager.time("matmul",  OpKind::Gemm,       stream, [&] { out_matmul  = A.matmul(B, Backend::Custom, stream); });
            manager.time("gelu",    OpKind::Elementwise,stream, [&] { out_gelu    = out_matmul.gelu   (Backend::Custom, stream); });
            manager.time("softmax", OpKind::Reduction,  stream, [&] { out_softmax = out_gelu  .softmax(Backend::Custom, stream); });
        });

        // --- Graph capture ---
        manager.capture("toy_pipeline", /*strict=*/true, [&]
        {
            out_matmul  = A.matmul (B, Backend::Custom, stream);
            out_gelu    = out_matmul.gelu   (Backend::Custom, stream);
            out_softmax = out_gelu  .softmax(Backend::Custom, stream);
        });

        manager.launch("toy_pipeline");
        stream.synchronize();

        manager.benchmark("toy_pipeline", 1, kReplays, [&](int)
        {
            manager.time("toy_pipeline", OpKind::GraphReplay, stream,
                         [&] { manager.launch("toy_pipeline"); }, /*from_graph_replay=*/true);
        });

        manager.telemetry().resolve_pending();
        manager.telemetry().print_summary("no_capture");
        manager.telemetry().print_summary("toy_pipeline");

        // --- Correctness: graph replay vs direct execution ---
        auto vec_graph = out_softmax.to_vector();

        Tensor<__nv_bfloat16> ref_matmul  = A.matmul (B, Backend::Custom, stream);
        Tensor<__nv_bfloat16> ref_gelu    = ref_matmul.gelu   (Backend::Custom, stream);
        Tensor<__nv_bfloat16> ref_softmax = ref_gelu  .softmax(Backend::Custom, stream);
        stream.synchronize();

        auto vec_ref = ref_softmax.to_vector();
        check_pipeline_replay_matches_direct(vec_graph, vec_ref, "toy_pipeline");

        std::cout << "\nToy pipeline test OK.\n";

        // --- Telemetry: publish to MQTT for the Android app ---
        std::cout << "\n--- telemetry (publishing to MQTT) ---\n";
        TelemetryMqttPublisher publisher("tcp://localhost:1883", "yadrakova_benchmark");
        bool ok = publisher.publish_lines(manager.telemetry().export_all_as_json_lines(),
                                          "yadrakova/telemetry");
        std::cout << (ok ? "Telemetry published OK.\n"
                         : "Failed publishing telemetry (broker down?).\n");

        std::cout << "\nAll pipeline tests passed.\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    return 0;
}