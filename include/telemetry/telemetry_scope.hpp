#pragma once
#include "telemetry/telemetry.hpp"
#include <chrono>
#include "core/cuda_error.hpp"
using namespace yadrakova::core;

// RAII Scope Guard para medir operaciones de manera transparente
    class TelemetryScope {
    public:
        TelemetryScope(Telemetry* telemetry, std::string label, OpKind kind,
                       const void* stream_hint = nullptr, bool from_graph_replay = false,
                       uint64_t bytes_moved = 0, double gflops = 0.0)
            : telemetry_(telemetry), label_(std::move(label)), kind_(kind),
              stream_hint_(stream_hint), from_graph_replay_(from_graph_replay),
              bytes_moved_(bytes_moved), gflops_(gflops)
        {
            if (telemetry_) {
                start_time_ = std::chrono::steady_clock::now();
            }
        }

        ~TelemetryScope() {
            if (telemetry_) {
                auto end_time = std::chrono::steady_clock::now();
                float duration_ms = std::chrono::duration<float, std::milli>(
                    end_time - start_time_).count();

                telemetry_->record(label_, kind_, duration_ms, stream_hint_,
                                  from_graph_replay_, bytes_moved_, gflops_);
            }
        }

    private:
        Telemetry* telemetry_;
        std::string label_;
        OpKind kind_;
        const void* stream_hint_;
        bool from_graph_replay_;
        uint64_t bytes_moved_;
        double gflops_;
        std::chrono::steady_clock::time_point start_time_;
    };