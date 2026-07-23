#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include "op_kind.hpp"

namespace yadrakova::core {

    const char* to_string(OpKind kind);

    struct TelemetryRecord {
        uint64_t sequence_id;
        std::string label;
        OpKind op_kind;
        double host_timestamp_ms;
        float duration_ms;
        int stream_id;
        bool from_graph_replay;
        uint64_t bytes_moved = 0;
        double gflops = 0.0;
    };

    // Evento CUDA aun sin cerrar: se creo start pero no end.
    struct InFlightEvent {
        cudaEvent_t start;
        cudaStream_t stream;
    };

    // Par de eventos ya cerrado (start+end grabados) pero sin sincronizar/leer.
    struct PendingEvent {
        cudaEvent_t start;
        cudaEvent_t end;
        std::string label;
        OpKind op_kind;
        const void* stream_hint;
        bool from_graph_replay;
        uint64_t bytes_moved;
        double gflops;
    };

    class Telemetry {
    public:
        explicit Telemetry(size_t reserve_capacity = 4096);

        void start_session();

        // Ruta sincrona existente: duration_ms ya viene medido (time_kernel_ms, etc.)
        void record(std::string label, OpKind kind, float duration_ms,
                    const void* stream_id_hint, bool from_graph_replay,
                    uint64_t bytes_moved = 0, double gflops = 0.0);

        // --- Ruta async (usada por TelemetryScope) ---
        // Fase 1: encola evento de inicio en el stream. No sincroniza.
        // Devuelve un handle para pasar a end_async().
        size_t begin_async(cudaStream_t stream);

        // Fase 1: encola evento de fin en el mismo stream. No sincroniza.
        // Los datos quedan en pending_ hasta que se llame resolve_pending().
        void end_async(size_t handle, std::string label, OpKind kind,
                       const void* stream_id_hint, bool from_graph_replay,
                       uint64_t bytes_moved = 0, double gflops = 0.0);

        // Fase 2: sincroniza TODOS los eventos pendientes de una vez,
        // calcula duraciones reales via cudaEventElapsedTime, y los
        // vuelca a records_ (via record()). Destruye los eventos CUDA.
        // Llamar antes de export_all_as_json_lines() / al final de un batch.
        void resolve_pending();

        const std::vector<TelemetryRecord>& records() const { return records_; }
        void clear();

        std::string to_json_line(const TelemetryRecord& r) const;
        std::vector<std::string> export_all_as_json_lines() const;

    private:
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