#pragma once
#include "core/stream.h"
#include "core/cuda_error.h"
#include <cuda_runtime.h>

namespace yadrakova::core {

// Wrapper de cudaEvent_t. Dos usos principales:
//   1. Medir tiempo REAL de ejecucion en GPU entre dos puntos
//      (no tiempo de wall-clock del host, que incluye overhead
//      de lanzamiento de kernel).
//   2. Sincronizar entre streams sin bloquear todo el device
//      (ej: que el stream de training espere a que el stream
//      de telemetry termine de leer un buffer, sin un
//      cudaDeviceSynchronize() global).
class Event {
public:
    // timing=true habilita medicion de tiempo (cudaEventDefault).
    // timing=false usa cudaEventDisableTiming, un poco mas rapido
    // si solo lo usas para sincronizar, no para medir.
    explicit Event(bool timing = true) {
        unsigned int flags = timing ? cudaEventDefault : cudaEventDisableTiming;
        CUDA_CHECK(cudaEventCreateWithFlags(&handle_, flags));
    }

    ~Event() {
        if (handle_) cudaEventDestroy(handle_);
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    Event(Event&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    Event& operator=(Event&& other) noexcept {
        if (this != &other) {
            if (handle_) cudaEventDestroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    // Inserta este evento en el stream dado -- se "dispara" cuando
    // todo lo encolado ANTES de esta llamada en ese stream termina.
    void record(const Stream& stream) {
        CUDA_CHECK(cudaEventRecord(handle_, stream.raw()));
    }

    // Bloquea el host hasta que el evento se dispare.
    void synchronize() const {
        CUDA_CHECK(cudaEventSynchronize(handle_));
    }

    cudaEvent_t raw() const { return handle_; }

private:
    cudaEvent_t handle_ = nullptr;
};

// Mide tiempo real de GPU en milisegundos entre dos eventos ya grabados
// (start DEBE haberse recordado antes que end en el mismo stream).
// Este es el numero que le importa a telemetry/monitor.cpp -- tiempo
// real de ejecucion en el device, no wall-clock del host que incluye
// overhead de CPU/driver al lanzar el kernel.
float elapsed_ms(const Event& start, const Event& end);

// Helper de conveniencia: mide el tiempo de ejecutar `fn` (una lambda
// que lanza uno o mas kernels) dentro del stream dado. Bloquea al
// final para poder leer el tiempo -- solo usar en profiling/tests,
// NUNCA en el hot path de training real (el synchronize rompe el
// solape asincrono que quieres en produccion).
template <typename Fn>
float time_kernel_ms(Stream& stream, Fn&& fn) {
    Event start(true), end(true);
    start.record(stream);
    fn();
    end.record(stream);
    end.synchronize();
    return elapsed_ms(start, end);
}

} // namespace yadrakova::core