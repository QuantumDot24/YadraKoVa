#pragma once
#include <cuda_runtime.h>
#include <memory>

namespace yadrakova::core
{
    struct DeviceBuffer
    {
        void* ptr = nullptr;
        size_t size_bytes = 0;
        int device_id = 0;

        DeviceBuffer() = default;

        DeviceBuffer(void* p, size_t s, int dev) : ptr(p), size_bytes(s), device_id(dev)
        {
        }
    };

    class MemoryPool
    {
    public:
        explicit MemoryPool(int device_id = 0);
        ~MemoryPool();
        MemoryPool(const MemoryPool&) = delete;
        MemoryPool& operator=(const MemoryPool&) = delete;

        void preallocate(size_t size_bytes, size_t count) const;
        std::shared_ptr<DeviceBuffer> allocate(size_t size_bytes) const;

        [[nodiscard]] size_t bytes_in_use() const;
        [[nodiscard]] size_t bytes_reserved() const;

        [[nodiscard]] long outstanding_buffers() const;

        struct Impl;

    private:
        std::shared_ptr<Impl> impl_;
    };

    MemoryPool& default_pool();
} // namespace yadrakova::core
