#include "core/memory.h"
#include "core/cuda_error.h"
#include <map>
#include <vector>
#include <mutex>
#include <stdexcept>
#include <iostream>

namespace yadrakova::core {

static size_t round_to_size_class(size_t bytes) {
    size_t sz = 256;
    while (sz < bytes) sz <<= 1;
    return sz;
}

struct MemoryPool::Impl {
    int device_id;
    mutable std::mutex mtx;
    std::map<size_t, std::vector<void*>> free_blocks;
    size_t total_reserved = 0;
    size_t total_in_use = 0;
    std::atomic<long> outstanding_buffers{0}; // buffers con shared_ptr vivo fuera del free-list

    void* raw_alloc(size_t bytes) {
        void* p = nullptr;
        CUDA_CHECK_CTX(cudaMalloc(&p, bytes),
                       "MemoryPool::raw_alloc(" + std::to_string(bytes) + " bytes)");
        total_reserved += bytes;
        return p;
    }

    // Solo se ejecuta cuando el ÚLTIMO shared_ptr<Impl> muere -- es decir,
    // cuando ya no hay ningún MemoryPool NI ningún DeviceBuffer vivo
    // referenciando este Impl. Aquí sí es seguro llamar cudaFree.
    ~Impl() {
        if (outstanding_buffers.load() != 0) {
            // Esto NO debería poder pasar dado el diseño (ver allocate()),
            // pero lo dejamos como señal fuerte de bug si algún día aparece.
            std::cerr << "[MemoryPool::Impl] ADVERTENCIA: destruyendo Impl con "
                      << outstanding_buffers.load()
                      << " buffers todavia marcados como outstanding.\n";
        }
        std::lock_guard<std::mutex> lock(mtx);
        for (auto& [size_class, blocks] : free_blocks) {
            for (void* p : blocks) cudaFree(p); // destructor: nunca lanza, error se ignora a proposito
        }
    }
};

MemoryPool::MemoryPool(int device_id) : impl_(std::make_shared<Impl>()) {
    impl_->device_id = device_id;
    CUDA_CHECK(cudaSetDevice(device_id));
}

// Este destructor casi nunca hace el trabajo real de liberar memoria --
// solo decrementa el shared_ptr<Impl>. Si hay Tensors vivos en otras
// partes del programa (cada uno con su propio shared_ptr<Impl> capturado
// en el deleter de su DeviceBuffer), Impl sigue vivo y la memoria real
// se libera hasta que el ultimo de ellos muera. Esto es intencional:
// resuelve el orden de destrucción entre `static MemoryPool` y tensors
// que puedan sobrevivirlo.
MemoryPool::~MemoryPool() = default;

void MemoryPool::preallocate(size_t size_bytes, size_t count) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    size_t size_class = round_to_size_class(size_bytes);
    auto& bucket = impl_->free_blocks[size_class];
    for (size_t i = 0; i < count; ++i) {
        bucket.push_back(impl_->raw_alloc(size_class));
    }
}

std::shared_ptr<DeviceBuffer> MemoryPool::allocate(size_t size_bytes) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    size_t size_class = round_to_size_class(size_bytes);
    auto& bucket = impl_->free_blocks[size_class];

    void* ptr;
    if (!bucket.empty()) {
        ptr = bucket.back();
        bucket.pop_back();
    } else {
        ptr = impl_->raw_alloc(size_class);
    }
    impl_->total_in_use += size_class;
    impl_->outstanding_buffers.fetch_add(1);

    // CLAVE DEL FIX: el deleter captura impl_ POR VALOR como shared_ptr,
    // no un puntero crudo. Esto extiende la vida de Impl (y por lo tanto
    // del free-list y de la memoria cruda) mientras exista este buffer,
    // sin importar si el MemoryPool original ya se destruyó.
    std::shared_ptr<Impl> impl_keepalive = impl_;
    auto deleter = [impl_keepalive, size_class, ptr_captured = ptr](DeviceBuffer* buf) {
        {
            std::lock_guard<std::mutex> lock(impl_keepalive->mtx);
            impl_keepalive->free_blocks[size_class].push_back(ptr_captured);
            impl_keepalive->total_in_use -= size_class;
        }
        impl_keepalive->outstanding_buffers.fetch_sub(1);
        delete buf;
        // impl_keepalive sale de scope aqui: si este era el ultimo
        // shared_ptr<Impl> vivo (pool ya destruido, sin otros tensors),
        // ahora si se dispara Impl::~Impl() y se hace cudaFree real.
    };

    return std::shared_ptr<DeviceBuffer>(
        new DeviceBuffer(ptr, size_bytes, impl_->device_id), deleter);
}

size_t MemoryPool::bytes_in_use() const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    return impl_->total_in_use;
}

size_t MemoryPool::bytes_reserved() const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    return impl_->total_reserved;
}

long MemoryPool::outstanding_buffers() const {
    return impl_->outstanding_buffers.load();
}

MemoryPool& default_pool() {
    static MemoryPool instance(/*device_id=*/0);
    return instance;
}

} // namespace yadrakova::core