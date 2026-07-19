#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <atomic>

namespace yadrakova::core {

    struct DeviceBuffer {
        void* ptr = nullptr;
        size_t size_bytes = 0;
        int device_id = 0;

        DeviceBuffer() = default;
        DeviceBuffer(void* p, size_t s, int dev) : ptr(p), size_bytes(s), device_id(dev) {}
    };

    // MemoryPool es una "handle" liviana. La memoria real y el free-list
    // viven en Impl, que se mantiene vivo por shared_ptr compartido entre
    // el propio MemoryPool y CADA DeviceBuffer que haya salido de allocate().
    //
    // Garantía: mientras exista un solo Tensor vivo en el programa (que
    // mantiene un shared_ptr<DeviceBuffer> con un deleter que a su vez
    // mantiene un shared_ptr<Impl>), el Impl no se destruye, sin importar
    // si el objeto MemoryPool que lo creó ya salió de scope. El cudaFree
    // real solo ocurre cuando el ÚLTIMO shared_ptr<Impl> muere.
    class MemoryPool {
    public:
        explicit MemoryPool(int device_id = 0);
        ~MemoryPool(); // no libera memoria si aún hay buffers vivos -- ver Impl::~Impl

        MemoryPool(const MemoryPool&) = delete;
        MemoryPool& operator=(const MemoryPool&) = delete;

        void preallocate(size_t size_bytes, size_t count);
        std::shared_ptr<DeviceBuffer> allocate(size_t size_bytes);

        size_t bytes_in_use() const;
        size_t bytes_reserved() const;

        // Diagnóstico: cuántos DeviceBuffers siguen vivos referenciando
        // este pool. Útil para detectar leaks de tensors al cerrar la app.
        long outstanding_buffers() const;

        struct Impl; // definido en memory.cpp

    private:
        std::shared_ptr<Impl> impl_;
    };

    MemoryPool& default_pool();

} // namespace yadrakova::core