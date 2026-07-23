// telemetry.cpp
#include "telemetry/telemetry.hpp"
#include <sstream>
#include <iomanip>

namespace yadrakova::core {

const char* to_string(OpKind kind) {
    switch (kind) {
        case OpKind::Gemm:        return "gemm";
        case OpKind::Elementwise: return "elementwise";
        case OpKind::Reduction:   return "reduction";
        case OpKind::GraphReplay: return "graph_replay";
        case OpKind::Memcpy:      return "memcpy";
        case OpKind::Other:       return "other";
    }
    return "other";
}

Telemetry::Telemetry(size_t reserve_capacity) {
    records_.reserve(reserve_capacity);
    // session_start_ se inicializa "vacio" hasta que se llame start_session();
    // si alguien graba antes de arrancar sesion, los timestamps saldran
    // relativos a este constructor, no es un error fatal pero conviene
    // llamar start_session() explicitamente.
    session_start_ = std::chrono::steady_clock::now();
}

void Telemetry::start_session() {
    session_start_ = std::chrono::steady_clock::now();
    next_sequence_ = 0;
    stream_id_map_.clear();
    next_stream_id_ = 0;
}

int Telemetry::resolve_stream_id(const void* stream_ptr) {
    auto it = stream_id_map_.find(stream_ptr);
    if (it != stream_id_map_.end()) {
        return it->second;
    }
    int id = next_stream_id_++;
    stream_id_map_.emplace(stream_ptr, id);
    return id;
}

void Telemetry::record(std::string label, OpKind kind, float duration_ms,
                        const void* stream_id_hint, bool from_graph_replay,
                        uint64_t bytes_moved, double gflops) {
    auto now = std::chrono::steady_clock::now();
    double host_ts_ms = std::chrono::duration<double, std::milli>(
        now - session_start_).count();

    TelemetryRecord r;
    r.sequence_id       = next_sequence_++;
    r.label              = std::move(label);
    r.op_kind            = kind;
    r.host_timestamp_ms  = host_ts_ms;
    r.duration_ms        = duration_ms;
    r.stream_id          = resolve_stream_id(stream_id_hint);
    r.from_graph_replay  = from_graph_replay;
    r.bytes_moved        = bytes_moved;
    r.gflops             = gflops;

    records_.push_back(std::move(r));
}

void Telemetry::clear() {
    records_.clear();
    // Nota: a proposito NO reseteamos next_sequence_ ni stream_id_map_ aqui.
    // clear() es "borra lo capturado hasta ahora" (p.ej. tras exportar un
    // lote a MQTT); start_session() es "empieza una sesion nueva de cero".
    // Si los mezclas y esperas sequence_id reiniciado en 0 tras un clear(),
    // ese es el motivo -- llama start_session() si quieres eso.
}

namespace {

// Escapa lo minimo necesario para que un string quede valido dentro de
// un JSON: comillas, backslash y saltos de linea. No es un escapador
// JSON completo (no cubre unicode raro), pero para labels de kernels
// ("matmul_wmma", etc.) es mas que suficiente.
std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

} // namespace

std::string Telemetry::to_json_line(const TelemetryRecord& r) const {
    std::ostringstream os;
    os << std::setprecision(6) << std::fixed;
    os << "{"
       << "\"sequence_id\":" << r.sequence_id << ","
       << "\"label\":\"" << json_escape(r.label) << "\","
       << "\"op_kind\":\"" << to_string(r.op_kind) << "\","
       << "\"host_timestamp_ms\":" << r.host_timestamp_ms << ","
       << "\"duration_ms\":" << r.duration_ms << ","
       << "\"stream_id\":" << r.stream_id << ","
       << "\"from_graph_replay\":" << (r.from_graph_replay ? "true" : "false") << ","
       << "\"bytes_moved\":" << r.bytes_moved << ","
       << "\"gflops\":" << r.gflops
       << "}";
    return os.str();
}

std::vector<std::string> Telemetry::export_all_as_json_lines() const {
    std::vector<std::string> lines;
    lines.reserve(records_.size());
    for (const auto& r : records_) {
        lines.push_back(to_json_line(r));
    }
    return lines;
}

} // namespace yadrakova::core