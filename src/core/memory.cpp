#include "core/memory.hpp"
#include "core/cuda_error.hpp"
#include "core/stream.hpp"
#include <map>
#include <vector>
#include <mutex>
#include <iostream>
#include <ranges>

namespace yadrakova::core
{
    struct PooledBlock
    {
        void* ptr = nullptr;
        cudaEvent_t event = nullptr;
    };

    static size_t round_to_size_class(size_t bytes)
    {
        size_t sz = 256;
        while (sz < bytes) sz <<= 1;
        return sz;
    }

    struct MemoryPool::Impl
    {
        int device_id{};
        mutable std::mutex mtx;
        std::map<size_t, std::vector<PooledBlock>> free_blocks;
        size_t total_reserved = 0;
        size_t total_in_use = 0;
        std::atomic<long> outstanding_buffers{0};

        void* raw_alloc(size_t bytes)
        {
            void* p = nullptr;
            CUDA_CHECK_CTX(cudaMalloc(&p, bytes),
                           "MemoryPool::raw_alloc(" + std::to_string(bytes) + " bytes)");
            total_reserved += bytes;
            return p;
        }

        ~Impl()
        {
            if (outstanding_buffers.load() != 0)
            {
                std::cerr << "[MemoryPool::Impl] WARNING: destroying Impl with "
                    << outstanding_buffers.load()
                    << " buffers still marked as outstanding.\n";
            }
            std::lock_guard<std::mutex> lock(mtx);
            for (auto& blocks : free_blocks | std::views::values)
            {
                for (const auto& block : blocks)
                {
                    if (block.event) cudaEventDestroy(block.event);
                    if (block.ptr) cudaFree(block.ptr);
                }
            }
        }
    };

    MemoryPool::MemoryPool(const int device_id) : impl_(std::make_shared<Impl>())
    {
        impl_->device_id = device_id;
        CUDA_CHECK(cudaSetDevice(device_id));
    }

    MemoryPool::~MemoryPool() = default;

    void MemoryPool::preallocate(size_t size_bytes, size_t count) const
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        const size_t size_class = round_to_size_class(size_bytes);
        auto& bucket = impl_->free_blocks[size_class];
        for (size_t i = 0; i < count; ++i)
        {
            bucket.push_back({impl_->raw_alloc(size_class), nullptr});
        }
    }

    std::shared_ptr<DeviceBuffer> MemoryPool::allocate(const size_t size_bytes, Stream& stream) const
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        size_t size_class = round_to_size_class(size_bytes);
        auto& bucket = impl_->free_blocks[size_class];

        void* ptr = nullptr;
        if (!bucket.empty())
        {
            PooledBlock block = bucket.back();
            bucket.pop_back();
            ptr = block.ptr;

            if (block.event)
            {
                CUDA_CHECK(cudaStreamWaitEvent(stream.raw(), block.event, 0));
                CUDA_CHECK(cudaEventDestroy(block.event));
            }
        }
        else
        {
            ptr = impl_->raw_alloc(size_class);
        }
        impl_->total_in_use += size_class;
        impl_->outstanding_buffers.fetch_add(1);

        std::shared_ptr<Impl> impl_keepalive = impl_;
        cudaStream_t active_stream = stream.raw();
        int dev_id = impl_->device_id;

        auto deleter = [impl_keepalive, size_class, ptr_captured = ptr, active_stream, dev_id](const DeviceBuffer* buf) noexcept
        {
            try
            {
                cudaSetDevice(dev_id);
                cudaEvent_t event = nullptr;
                if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess)
                {
                    cudaEventRecord(event, active_stream);
                }

                {
                    std::lock_guard<std::mutex> lock(impl_keepalive->mtx);
                    impl_keepalive->free_blocks[size_class].push_back({ptr_captured, event});
                    impl_keepalive->total_in_use -= size_class;
                }
                impl_keepalive->outstanding_buffers.fetch_sub(1);
            }
            catch (...)
            {
                // Evita que excepciones se propaguen fuera del destructor/deleter
            }
            delete buf;
        };

        return {
            new DeviceBuffer(ptr, size_bytes, impl_->device_id), deleter
        };
    }

    size_t MemoryPool::bytes_in_use() const
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        return impl_->total_in_use;
    }

    size_t MemoryPool::bytes_reserved() const
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        return impl_->total_reserved;
    }

    long MemoryPool::outstanding_buffers() const
    {
        return impl_->outstanding_buffers.load();
    }

    MemoryPool& default_pool()
    {
        static MemoryPool instance(/*device_id=*/0);
        return instance;
    }
} // namespace yadrakova::core