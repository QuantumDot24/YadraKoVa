#pragma once
#include <cuda_runtime.h>

namespace yadrakova::core
{
    class Stream
    {
    public:
        explicit Stream(bool non_blocking = true);
        ~Stream();

        Stream(const Stream&) = delete;
        Stream& operator=(const Stream&) = delete;

        Stream(Stream&& other) noexcept;
        Stream& operator=(Stream&& other) noexcept;

        void synchronize() const;
        [[nodiscard]] bool is_done() const;

        [[nodiscard]] cudaStream_t raw() const { return handle_; }

    private:
        cudaStream_t handle_ = nullptr;
    };

    // Stream activo del hilo actual. Lazy: se crea solo la primera vez que
    // se pide, uno por hilo. Es lo que usan matmul/softmax/gelu/time_kernel_ms
    // como default -- en el uso normal nadie necesita saber que esto existe.
    Stream& default_stream();

    // Azucar: equivalente a default_stream().synchronize().
    void synchronize();

    // Escape hatch para multi-stream real (overlap, multi-GPU). Mientras el
    // guard esta vivo, default_stream() devuelve `s` en vez del stream lazy
    // del hilo. Restaura el anterior al salir de scope -- soporta anidar.
    class StreamGuard
    {
    public:
        explicit StreamGuard(Stream& s);
        ~StreamGuard();

        StreamGuard(const StreamGuard&) = delete;
        StreamGuard& operator=(const StreamGuard&) = delete;

    private:
        Stream* previous_;
    };
} // namespace yadrakova::core