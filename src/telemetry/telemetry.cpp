// telemetry.cpp
#include "telemetry/telemetry.hpp"
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace yadrakova::core {

    const char* to_string(OpKind kind) {
        switch (kind) {
        case OpKind::Gemm:        return "gemm";
        case OpKind::Gemv:        return "gemv";
        case OpKind::Elementwise: return "elementwise";
        case OpKind::Reduction:   return "reduction";
        case OpKind::Attention:   return "attention";
        case OpKind::Conv2D:      return "conv2d";
        case OpKind::GraphReplay: return "graph_replay";
        case OpKind::Memcpy:      return "memcpy";
        case OpKind::Other:       return "other";
        }
        return "other";
    }

Telemetry::Telemetry(size_t reserve_capacity) {
    records_.reserve(reserve_capacity);
    pending_.reserve(reserve_capacity);
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

size_t Telemetry::begin_async(cudaStream_t stream) {
    cudaEvent_t start;
    cudaError_t err = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaEventCreate (start) fallo: ") +
                                  cudaGetErrorString(err));
    }
    err = cudaEventRecord(start, stream);
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        throw std::runtime_error(std::string("cudaEventRecord (start) fallo: ") +
                                  cudaGetErrorString(err));
    }

    in_flight_.push_back(InFlightEvent{start, stream});
    return in_flight_.size() - 1;
}

void Telemetry::end_async(size_t handle, std::string label, OpKind kind,
                           const void* stream_id_hint, bool from_graph_replay,
                           uint64_t bytes_moved, double gflops) {
    // handle invalido (fuera de rango) -> no reventar en release, solo ignorar.
    if (handle >= in_flight_.size()) {
        return;
    }
    InFlightEvent in_flight = in_flight_[handle];

    cudaEvent_t end;
    cudaError_t err = cudaEventCreate(&end);
    if (err != cudaSuccess) {
        cudaEventDestroy(in_flight.start);
        throw std::runtime_error(std::string("cudaEventCreate (end) fallo: ") +
                                  cudaGetErrorString(err));
    }
    err = cudaEventRecord(end, in_flight.stream);
    if (err != cudaSuccess) {
        cudaEventDestroy(in_flight.start);
        cudaEventDestroy(end);
        throw std::runtime_error(std::string("cudaEventRecord (end) fallo: ") +
                                  cudaGetErrorString(err));
    }

    pending_.push_back(PendingEvent{
        in_flight.start, end, std::move(label), kind,
        stream_id_hint, from_graph_replay, bytes_moved, gflops
    });

    // No compactamos in_flight_ (evitar invalidar handles activos concurrentes).
    // Se limpia entero en resolve_pending().
}

void Telemetry::resolve_pending() {
    for (auto& p : pending_) {
        // cudaEventSynchronize espera a que 'end' se complete, lo cual
        // implica que todo lo encolado antes en ese stream (incluyendo
        // 'start') ya termino tambien. Un solo punto de sincronizacion
        // por evento, no un barrier global de todos los streams.
        cudaError_t err = cudaEventSynchronize(p.end);
        if (err != cudaSuccess) {
            cudaEventDestroy(p.start);
            cudaEventDestroy(p.end);
            throw std::runtime_error(std::string("cudaEventSynchronize fallo: ") +
                                      cudaGetErrorString(err));
        }

        float ms = 0.0f;
        err = cudaEventElapsedTime(&ms, p.start, p.end);
        if (err != cudaSuccess) {
            cudaEventDestroy(p.start);
            cudaEventDestroy(p.end);
            throw std::runtime_error(std::string("cudaEventElapsedTime fallo: ") +
                                      cudaGetErrorString(err));
        }

        record(std::move(p.label), p.op_kind, ms, p.stream_hint,
               p.from_graph_replay, p.bytes_moved, p.gflops);

        cudaEventDestroy(p.start);
        cudaEventDestroy(p.end);
    }
    pending_.clear();
    in_flight_.clear();
}

void Telemetry::clear() {
    records_.clear();
}

namespace {

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