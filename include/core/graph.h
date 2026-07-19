#pragma once
#include "core/stream.h"
#include "core/memory.h"
#include "core/cuda_error.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace yadrakova::core {

class Graph {
public:
    explicit Graph(std::string name) : name_(std::move(name)) {}

    ~Graph() {
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
        if (graph_) cudaGraphDestroy(graph_);
    }

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    void begin_capture(Stream& stream, MemoryPool& pool = default_pool(), bool strict = true) {
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

    void end_capture(Stream& stream) {
        if (!capturing_) {
            throw std::runtime_error("Graph '" + name_ + "': no hay captura en progreso para cerrar.");
        }

        if (graph_) {
            cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }

        // No CUDA_CHECK_CTX directo aqui: si cudaStreamEndCapture falla,
        // igual necesitamos bajar capturing_ a false antes de propagar,
        // para que el objeto no quede "atorado" en estado capturing.
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
                    " bytes DURANTE la captura. Esto significa que se creo un "
                    "Tensor/buffer nuevo sin preallocate() previo -- "
                    "preasigna toda la memoria necesaria ANTES de begin_capture().");
            }
        }

        instantiate();
    }

    void launch(Stream& stream) {
        if (!graph_exec_) {
            throw std::runtime_error("Graph '" + name_ + "': no instanciado, llama end_capture() primero.");
        }
        CUDA_CHECK_CTX(cudaGraphLaunch(graph_exec_, stream.raw()), "Graph '" + name_ + "'");
    }

    // Para cuando la TOPOLOGIA es identica pero cambiaron punteros de
    // datos (nuevo batch, pesos actualizados) -- evita re-capturar/
    // re-instanciar desde cero, que es mas costoso.
    void update_from(Graph& new_topology) {
        cudaGraphExecUpdateResultInfo result_info{};
        cudaError_t err = cudaGraphExecUpdate(graph_exec_, new_topology.graph_, &result_info);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                "Graph '" + name_ + "': cudaGraphExecUpdate fallo (topologia probablemente cambio, "
                "se requiere re-captura completa): " + cudaGetErrorString(err));
        }
    }
    // Aborta una captura en progreso que fallo por una operacion prohibida
    // (ej: cudaMalloc durante captura). A diferencia de end_capture(), no
    // espera terminar con un grafo valido -- limpia el estado de captura
    // del stream y resetea capturing_, para que el objeto quede reusable.
    // Uso tipico: dentro de un catch() despues de que algo dentro de
    // begin_capture()/.../end_capture() lanzo.
    void abort_capture(Stream& stream) noexcept {
        if (!capturing_) return; // nada que abortar
        cudaGraph_t dangling = nullptr;
        cudaStreamEndCapture(stream.raw(), &dangling); // error esperado, se ignora
        if (dangling) cudaGraphDestroy(dangling);
        capturing_ = false;
    }
    const std::string& name() const { return name_; }
    bool is_instantiated() const { return graph_exec_ != nullptr; }

private:
    void instantiate() {
        if (graph_exec_) {
            cudaGraphExecDestroy(graph_exec_);
            graph_exec_ = nullptr;
        }
        CUDA_CHECK_CTX(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0),
                       "Graph '" + name_ + "'");
    }

    std::string name_;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    bool capturing_ = false;
    bool strict_ = true;
    MemoryPool* pool_ = nullptr;
    size_t bytes_reserved_before_ = 0;
};

} // namespace yadrakova::core