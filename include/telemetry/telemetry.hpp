// telemetry.hpp
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cstdint>
#include "op_kind.hpp"

namespace yadrakova::core {

    const char* to_string(OpKind kind);

    struct TelemetryRecord {
        uint64_t sequence_id;
        std::string label;
        OpKind op_kind;
        double host_timestamp_ms;   // ms desde el inicio de la sesion, no epoch
        float duration_ms;
        int stream_id;              // id pequeno y estable (0,1,2...), no puntero crudo
        bool from_graph_replay;
        uint64_t bytes_moved = 0;   // 0 = no aplica/no medido
        double gflops = 0.0;        // 0 = no aplica/no medido
    };

    // Buffer de captura en host, pensado para bajo overhead: nada de
    // allocations por evento (reserve() al inicio), nada de formateo
    // de string en la ruta caliente -- eso se hace al exportar, no al grabar.
    class Telemetry {
    public:
        explicit Telemetry(size_t reserve_capacity = 4096);

        // Marca t=0 de la sesion -- llamar una vez al arrancar.
        void start_session();

        // Registra una operacion ya medida (duration_ms viene de
        // time_kernel_ms o de tu propio par start/end Event).
        // stream_id_hint: pasa stream.raw() reinterpretado, la funcion
        // se encarga de mapearlo a un id chico y estable.
        void record(std::string label, OpKind kind, float duration_ms,
                    const void* stream_id_hint, bool from_graph_replay,
                    uint64_t bytes_moved = 0, double gflops = 0.0);

        const std::vector<TelemetryRecord>& records() const { return records_; }
        void clear();

        // Exportar para transporte -- JSON lines, pensado para que
        // cada linea sea un mensaje MQTT independiente hacia Android.
        std::string to_json_line(const TelemetryRecord& r) const;
        std::vector<std::string> export_all_as_json_lines() const;

    private:
        // Asigna (o reutiliza) un id pequeno y estable para un puntero de stream.
        int resolve_stream_id(const void* stream_ptr);

        std::chrono::steady_clock::time_point session_start_;
        uint64_t next_sequence_ = 0;
        std::vector<TelemetryRecord> records_;
        std::unordered_map<const void*, int> stream_id_map_;
        int next_stream_id_ = 0;
    };

} // namespace yadrakova::core