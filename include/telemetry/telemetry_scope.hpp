#pragma once
#include "telemetry/telemetry.hpp"
#include <cuda_runtime.h>

namespace yadrakova::core {

    // RAII: al construirse encola un evento CUDA de inicio en el stream dado;
    // al destruirse encola el de fin. NO sincroniza en ningun punto -- el
    // hot path solo paga 2 cudaEventRecord async. Los tiempos reales se
    // resuelven despues, todos juntos, con Telemetry::resolve_pending().
    class TelemetryScope {
    public:
        TelemetryScope(Telemetry* telemetry, std::string label, OpKind kind,
                       cudaStream_t stream, const void* stream_hint = nullptr,
                       bool from_graph_replay = false,
                       uint64_t bytes_moved = 0, double gflops = 0.0)
            : telemetry_(telemetry), label_(std::move(label)), kind_(kind),
              stream_hint_(stream_hint), from_graph_replay_(from_graph_replay),
              bytes_moved_(bytes_moved), gflops_(gflops)
        {
            if (telemetry_) {
                handle_ = telemetry_->begin_async(stream);
            }
        }

        ~TelemetryScope() {
            if (telemetry_) {
                telemetry_->end_async(handle_, label_, kind_, stream_hint_,
                                      from_graph_replay_, bytes_moved_, gflops_);
            }
        }

        TelemetryScope(const TelemetryScope&) = delete;
        TelemetryScope& operator=(const TelemetryScope&) = delete;

    private:
        Telemetry* telemetry_;
        std::string label_;
        OpKind kind_;
        const void* stream_hint_;
        bool from_graph_replay_;
        uint64_t bytes_moved_;
        double gflops_;
        size_t handle_ = 0;
    };

} // namespace yadrakova::core