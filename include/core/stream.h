#pragma once
#include "core/cuda_error.h"
#include <cuda_runtime.h>

namespace yadrakova::core {

// Wrapper de cudaStream_t -- tu equivalente a un VkCommandBuffer/queue
// de Vulkan, pero mucho mas ligero: aqui un stream es solo una "cola
// de ejecucion ordenada" en la que lanzas kernels/memcpys de forma
// asincrona respecto al host.
//
// Regla de oro: operaciones DENTRO del mismo stream se ejecutan en
// orden (como un command buffer). Operaciones en streams DISTINTOS
// pueden solaparse (cómputo + transferencia en paralelo), que es
// exactamente lo que quieres para telemetry corriendo aparte del
// forward/backward.
class Stream {
public:
    // non_blocking=true crea el stream con cudaStreamNonBlocking,
    // para que no se serialice implicitamente con el stream 0 (default)
    // de otras partes del codigo -- casi siempre quieres esto en true.
    explicit Stream(bool non_blocking = true) {
        unsigned int flags = non_blocking ? cudaStreamNonBlocking : cudaStreamDefault;
        CUDA_CHECK(cudaStreamCreateWithFlags(&handle_, flags));
    }

    ~Stream() {
        if (handle_) cudaStreamDestroy(handle_);
    }

    // No copiable -- un stream es un recurso unico del driver.
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    // Movible -- para poder guardar Streams en contenedores/retornarlos.
    Stream(Stream&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    Stream& operator=(Stream&& other) noexcept {
        if (this != &other) {
            if (handle_) cudaStreamDestroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    // Bloquea el host hasta que todo lo encolado en este stream termine.
    // Equivalente a esperar un fence en Vulkan.
    void synchronize() const {
        CUDA_CHECK(cudaStreamSynchronize(handle_));
    }

    // No bloquea -- solo pregunta si ya termino. Util para polling
    // desde telemetry sin detener el hilo principal.
    bool is_done() const {
        cudaError_t err = cudaStreamQuery(handle_);
        if (err == cudaSuccess) return true;
        if (err == cudaErrorNotReady) return false;
        CUDA_CHECK(err); // cualquier otro codigo es un error real -> lanza con contexto
        return false; // inalcanzable, silencia warning
    }

    cudaStream_t raw() const { return handle_; }

private:
    cudaStream_t handle_ = nullptr;
};

// El "stream por defecto" del sistema -- para tests o casos donde no
// te importa solapar ejecucion. Los kernels que no reciban un Stream
// explicito deberian caer aqui.
Stream& default_stream();

} // namespace yadrakova::core