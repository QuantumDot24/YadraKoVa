#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include "op_kind.hpp"

namespace yadrakova::core
{
    class Stream;
    const char* to_string(OpKind kind);

    struct TelemetryRecord
    {
        uint64_t sequence_id;
        std::string label;
        OpKind op_kind;
        double host_timestamp_ms;
        float duration_ms;
        int stream_id;
        bool from_graph_replay;
        uint64_t bytes_moved = 0;
        double gflops = 0.0;
        std::string group_label;
    };

    struct InFlightEvent
    {
        cudaEvent_t start;
        cudaStream_t stream;
    };

    struct PendingEvent
    {
        cudaEvent_t start;
        cudaEvent_t end;
        std::string label;
        OpKind op_kind;
        const void* stream_hint;
        bool from_graph_replay;
        uint64_t bytes_moved;
        double gflops;
        std::string group_label;
    };

    class Telemetry
    {
    public:
        explicit Telemetry(size_t reserve_capacity = 4096);

        void start_session();

        void record(std::string label, OpKind kind, float duration_ms,
                    const void* stream_id_hint, bool from_graph_replay,
                    uint64_t bytes_moved = 0, double gflops = 0.0);
        void record_with_group(std::string label, OpKind kind, float duration_ms, const void* stream_id_hint,
                               bool from_graph_replay, std::string group_label, uint64_t bytes_moved, double gflops);

        // --- Ruta async (usada por TelemetryScope) ---
        // Fase 1: encola evento de inicio en el stream. No sincroniza.
        // Devuelve un handle para pasar a end_async().
        size_t begin_async(cudaStream_t stream);


        void end_async(size_t handle, std::string label, OpKind kind,
                       const void* stream_id_hint, bool from_graph_replay,
                       uint64_t bytes_moved = 0, double gflops = 0.0);


        void resolve_pending();

        [[nodiscard]] const std::vector<TelemetryRecord>& records() const { return records_; }
        void clear();

        [[nodiscard]] static std::string to_json_line(const TelemetryRecord& r);
        [[nodiscard]] std::vector<std::string> export_all_as_json_lines() const;
        void begin_group(std::string label, size_t ops_per_iteration = 1);
        void end_group();

        template <typename F>
        auto time(std::string label, OpKind kind, Stream& stream, F&& fn,
                  bool from_graph_replay = false,
                  uint64_t bytes_moved = 0, double gflops = 0.0)
            -> decltype(fn());

        template <typename F>
        void benchmark(std::string group_label, size_t ops_per_iteration,
                       int replays, F&& iteration_fn)
        {
            begin_group(std::move(group_label), ops_per_iteration);
            for (int i = 0; i < replays; ++i) iteration_fn(i);
            end_group();
        }

        void print_summary(const std::string& group_label) const;

    private:
        std::string current_group_label_;
        std::unordered_map<std::string, size_t> group_ops_per_iter_;
        int resolve_stream_id(const void* stream_ptr);

        std::chrono::steady_clock::time_point session_start_;
        uint64_t next_sequence_ = 0;
        std::vector<TelemetryRecord> records_;
        std::unordered_map<const void*, int> stream_id_map_;
        int next_stream_id_ = 0;

        std::vector<InFlightEvent> in_flight_;
        std::vector<PendingEvent> pending_;
    };
} // namespace yadrakova::core
