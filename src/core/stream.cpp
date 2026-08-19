#include "core/stream.hpp"
#include "core/cuda_error.hpp"
#include "core/backend_dispatch.hpp"

namespace yadrakova::core
{
    Stream::Stream(const bool non_blocking)
    {
        const unsigned int flags = non_blocking ? cudaStreamNonBlocking : cudaStreamDefault;
        CUDA_CHECK(cudaStreamCreateWithFlags(&handle_, flags));
    }

    Stream::~Stream()
    {
        // Los handles de cuBLAS/cuDNN/cuTENSOR/cuRAND estan indexados por el
        // cudaStream_t crudo (ver BackendContext). Si no los liberamos aqui,
        // un Stream nuevo que reuse esa misma direccion heredaria handles de
        // otro stream ya destruido.
        if (handle_) BackendContext::instance().reset(*this);
        if (handle_) cudaStreamDestroy(handle_);
    }

    Stream::Stream(Stream&& other) noexcept : handle_(other.handle_)
    {
        other.handle_ = nullptr;
    }

    Stream& Stream::operator=(Stream&& other) noexcept
    {
        if (this != &other)
        {
            if (handle_) BackendContext::instance().reset(*this);
            if (handle_) cudaStreamDestroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    void Stream::synchronize() const
    {
        CUDA_CHECK(cudaStreamSynchronize(handle_));
    }

    bool Stream::is_done() const
    {
        cudaError_t err = cudaStreamQuery(handle_);
        if (err == cudaSuccess) return true;
        if (err == cudaErrorNotReady) return false;
        CUDA_CHECK(err);
        return false;
    }

    // -----------------------------------------------------------------------
    // Stream por defecto -- thread_local, lazy, redirigible con StreamGuard.
    //
    // g_current_stream es el "override" activo (nullptr si nadie lo puso via
    // StreamGuard). lazy_stream es el stream que cada hilo crea solo, una
    // unica vez, la primera vez que default_stream() se llama sin override.
    //
    // Ambos son thread_local: cada hilo tiene su propio lazy_stream y su
    // propio puntero de override, asi que dos hilos nunca compiten por el
    // mismo cudaStream_t sin que el usuario haya hecho nada especial.
    // -----------------------------------------------------------------------
    namespace
    {
        thread_local Stream* g_current_stream = nullptr;
    }

    Stream& default_stream()
    {
        thread_local Stream lazy_stream(/*non_blocking=*/true);
        return g_current_stream ? *g_current_stream : lazy_stream;
    }

    void synchronize()
    {
        default_stream().synchronize();
    }

    StreamGuard::StreamGuard(Stream& s) : previous_(g_current_stream)
    {
        g_current_stream = &s;
    }

    StreamGuard::~StreamGuard()
    {
        g_current_stream = previous_;
    }
} // namespace yadrakova::core