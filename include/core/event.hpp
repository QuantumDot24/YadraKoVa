#pragma once
#include "core/stream.hpp"
#include <cuda_runtime.h>

namespace yadrakova::core
{
    class Event
    {
    public:
        explicit Event(bool timing = true);
        ~Event();

        Event(const Event&) = delete;
        Event& operator=(const Event&) = delete;

        Event(Event&& other) noexcept;
        Event& operator=(Event&& other) noexcept;

        void record(const Stream& stream);
        void synchronize() const;

        cudaEvent_t raw() const { return handle_; }

    private:
        cudaEvent_t handle_ = nullptr;
    };

    float elapsed_ms(const Event& start, const Event& end);

    template <typename Fn>
    float time_kernel_ms(Stream& stream, Fn&& fn)
    {
        Event start(true), end(true);
        start.record(stream);
        fn();
        end.record(stream);
        end.synchronize();
        return elapsed_ms(start, end);
    }
} // namespace yadrakova::core
