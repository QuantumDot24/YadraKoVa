#include "core/graph.hpp"
#include "core/cuda_error.hpp"

namespace yadrakova::core {

Graph::Graph(std::string name) : name_(std::move(name)) {}

Graph::~Graph() {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (graph_) cudaGraphDestroy(graph_);
}

void Graph::begin_capture(Stream& stream, MemoryPool& pool, bool strict) {
    if (capturing_) {
        throw std::runtime_error("Graph '" + name_ + "': ya hay una captura en progreso.");
    }
    strict_ = strict;
    pool_ = &pool;
    bytes_reserved_before_ = pool.bytes_reserved();

    CUDA_CHECK_CTX(cudaStreamBeginCapture(stream.raw(), cudaStreamCaptureModeGlobal),
                   "Graph '" + name_ + "'");
    capturing_ = true;
}

void Graph::end_capture(Stream& stream) {
    if (!capturing_) {
        throw std::runtime_error("Graph '" + name_ + "': no hay captura en progreso para cerrar.");
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    cudaError_t err = cudaStreamEndCapture(stream.raw(), &graph_);
    capturing_ = false;
    CUDA_CHECK_CTX(err, "Graph '" + name_ + "'");

    if (strict_) {
        size_t bytes_after = pool_->bytes_reserved();
        if (bytes_after != bytes_reserved_before_) {
            throw std::runtime_error(
                "Graph '" + name_ + "': el MemoryPool crecio de " +
                std::to_string(bytes_reserved_before_) + " a " +
                std::to_string(bytes_after) +
                " bytes DURANTE la captura. Preasigna toda la memoria antes de begin_capture().");
        }
    }
    instantiate();
}

void Graph::launch(Stream& stream) {
    if (!graph_exec_) {
        throw std::runtime_error("Graph '" + name_ + "': no instanciado, llama end_capture() primero.");
    }
    CUDA_CHECK_CTX(cudaGraphLaunch(graph_exec_, stream.raw()), "Graph '" + name_ + "'");
}

void Graph::update_from(Graph& new_topology) {
    cudaGraphExecUpdateResultInfo result_info{};
    CUDA_CHECK_CTX(
        cudaGraphExecUpdate(graph_exec_, new_topology.graph_, &result_info),
        "Graph '" + name_ + "': cudaGraphExecUpdate fallo (topologia probablemente cambio)");
}

void Graph::abort_capture(Stream& stream) noexcept {
    if (!capturing_) return;
    cudaGraph_t dangling = nullptr;
    cudaStreamEndCapture(stream.raw(), &dangling);
    if (dangling) cudaGraphDestroy(dangling);
    capturing_ = false;
}

void Graph::instantiate() {
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    CUDA_CHECK_CTX(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0),
                   "Graph '" + name_ + "'");
}

} // namespace yadrakova::core