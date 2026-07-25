#include "core/event.hpp"
#include "core/cuda_error.hpp"

namespace yadrakova::core
{
    Event::Event(bool timing)
    {
        unsigned int flags = timing ? cudaEventDefault : cudaEventDisableTiming;
        CUDA_CHECK(cudaEventCreateWithFlags(&handle_, flags));
    }

    Event::~Event()
    {
        if (handle_) cudaEventDestroy(handle_);
    }

    Event::Event(Event&& other) noexcept : handle_(other.handle_)
    {
        other.handle_ = nullptr;
    }

    Event& Event::operator=(Event&& other) noexcept
    {
        if (this != &other)
        {
            if (handle_) cudaEventDestroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    void Event::record(const Stream& stream)
    {
        CUDA_CHECK(cudaEventRecord(handle_, stream.raw()));
    }

    void Event::synchronize() const
    {
        CUDA_CHECK(cudaEventSynchronize(handle_));
    }

    float elapsed_ms(const Event& start, const Event& end)
    {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.raw(), end.raw()));
        return ms;
    }
} // namespace yadrakova::core
