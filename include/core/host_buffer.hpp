#pragma once
#include "core/cuda_error.hpp"

namespace yadrakova::core
{
    template <typename T>
    class HostBuffer
    {
    public:
        enum class Flags : unsigned int
        {
            Default = cudaHostAllocDefault,
            Portable = cudaHostAllocPortable,
            Mapped = cudaHostAllocMapped,
            WriteCombined = cudaHostAllocWriteCombined,
        };

        HostBuffer() = default;

        explicit HostBuffer(size_t count, Flags flags = Flags::Default) : count_(count)
        {
            if (count_ == 0) return;
            void* ptr = nullptr;
            CUDA_CHECK(cudaHostAlloc(&ptr, count_ * sizeof(T), static_cast<unsigned int>(flags)));
            data_ = static_cast<T*>(ptr);
        }

        ~HostBuffer() { release(); }

        HostBuffer(const HostBuffer&) = delete;
        HostBuffer& operator=(const HostBuffer&) = delete;

        HostBuffer(HostBuffer&& o) noexcept : data_(o.data_), count_(o.count_)
        {
            o.data_ = nullptr;
            o.count_ = 0;
        }

        HostBuffer& operator=(HostBuffer&& o) noexcept
        {
            if (this != &o)
            {
                release();
                data_ = o.data_;
                count_ = o.count_;
                o.data_ = nullptr;
                o.count_ = 0;
            }
            return *this;
        }

        T* data() noexcept { return data_; }
        const T* data() const noexcept { return data_; }
        [[nodiscard]] size_t size() const noexcept { return count_; }
        [[nodiscard]] size_t bytes() const noexcept { return count_ * sizeof(T); }
        [[nodiscard]] bool empty() const noexcept { return count_ == 0; }
        T& operator[](size_t i) { return data_[i]; }
        const T& operator[](size_t i) const { return data_[i]; }

    private:
        void release() noexcept
        {
            if (data_)
            {
                cudaFreeHost(data_);
                data_ = nullptr;
                count_ = 0;
            }
        }

        T* data_ = nullptr;
        size_t count_ = 0;
    };
} // namespace yadrakova::core
