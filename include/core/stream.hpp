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

    Stream& default_stream();
} // namespace yadrakova::core
