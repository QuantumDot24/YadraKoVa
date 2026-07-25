#include "core/host_buffer.hpp"
#include "core/tensor.hpp"
#include "core/stream.hpp"
#include <cassert>
#include <iostream>
#include <vector>

using namespace yadrakova::core;

void test_host_buffer_alloc_and_access() {
    HostBuffer<float> buf(1024);
    assert(buf.size() == 1024);
    assert(buf.bytes() == 1024 * sizeof(float));
    assert(!buf.empty());
    assert(buf.data() != nullptr);

    for (size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<float>(i) * 0.5f;
    for (size_t i = 0; i < buf.size(); ++i) assert(buf[i] == static_cast<float>(i) * 0.5f);

    std::cout << "[OK] host_buffer_alloc_and_access\n";
}

void test_host_buffer_default_empty() {
    HostBuffer<float> buf; // sin count -- no debe llamar cudaHostAlloc
    assert(buf.empty());
    assert(buf.size() == 0);
    assert(buf.data() == nullptr);

    std::cout << "[OK] host_buffer_default_empty\n";
}

void test_host_buffer_move_semantics() {
    HostBuffer<float> a(256);
    a[0] = 42.0f;
    float* original_ptr = a.data();

    HostBuffer<float> b(std::move(a));
    assert(b.data() == original_ptr);
    assert(b.size() == 256);
    assert(b[0] == 42.0f);
    // El objeto movido-desde debe quedar vacio, no apuntando a memoria liberada.
    assert(a.data() == nullptr);
    assert(a.size() == 0);

    HostBuffer<float> c(64);
    c = std::move(b);
    assert(c.data() == original_ptr);
    assert(c[0] == 42.0f);
    assert(b.data() == nullptr);

    std::cout << "[OK] host_buffer_move_semantics\n";
}

void test_tensor_sync_roundtrip() {
    MemoryPool pool(0);
    const size_t n = 4096;

    HostBuffer<float> src(n);
    for (size_t i = 0; i < n; ++i) src[i] = static_cast<float>(i) * 1.5f - 100.0f;

    Tensor<float> t({static_cast<int64_t>(n)}, pool);
    t.to_device(src);

    HostBuffer<float> dst(n);
    t.to_host(dst);

    for (size_t i = 0; i < n; ++i) {
        assert(dst[i] == src[i]);
    }

    std::cout << "[OK] tensor_sync_roundtrip\n";
}

void test_tensor_sync_roundtrip_raw_ptr() {
    // La variante con T* crudo (no HostBuffer) tambien debe funcionar --
    // es el path que usan tests que todavia no migraron a pinned memory.
    MemoryPool pool(0);
    const size_t n = 512;

    std::vector<float> src(n);
    for (size_t i = 0; i < n; ++i) src[i] = static_cast<float>(i);

    Tensor<float> t({static_cast<int64_t>(n)}, pool);
    t.to_device(src.data(), src.size());

    std::vector<float> dst(n);
    t.to_host(dst.data(), dst.size());

    assert(dst == src);

    std::cout << "[OK] tensor_sync_roundtrip_raw_ptr\n";
}

void test_tensor_size_mismatch_throws() {
    MemoryPool pool(0);
    Tensor<float> t({1024}, pool);
    HostBuffer<float> wrong_size(512);

    bool threw = false;
    try {
        t.to_device(wrong_size);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    std::cout << "[OK] tensor_size_mismatch_throws\n";
}

void test_tensor_async_transfer() {
    MemoryPool pool(0);
    Stream stream;
    const size_t n = 8192;

    HostBuffer<float> src(n);
    for (size_t i = 0; i < n; ++i) src[i] = static_cast<float>(i) * 0.25f;

    Tensor<float> t({static_cast<int64_t>(n)}, pool);
    t.to_device_async(src, stream);

    HostBuffer<float> dst(n);
    t.to_host_async(dst, stream);
    stream.synchronize();

    for (size_t i = 0; i < n; ++i) {
        assert(dst[i] == src[i]);
    }

    std::cout << "[OK] tensor_async_transfer\n";
}

void test_non_contiguous_transfer_throws() {
    MemoryPool pool(0);
    Tensor<float> t({64, 64}, pool);
    Tensor<float> view = t.transpose(0, 1);
    assert(!view.is_contiguous());

    HostBuffer<float> dst(64 * 64);
    bool threw = false;
    try {
        view.to_host(dst);
    } catch (const std::runtime_error& e) {
        threw = true;
        std::cout << "  (excepcion esperada: " << e.what() << ")\n";
    }
    assert(threw);

    std::cout << "[OK] non_contiguous_transfer_throws\n";
}

int main() {
    test_host_buffer_alloc_and_access();
    test_host_buffer_default_empty();
    test_host_buffer_move_semantics();
    test_tensor_sync_roundtrip();
    test_tensor_sync_roundtrip_raw_ptr();
    test_tensor_size_mismatch_throws();
    test_tensor_async_transfer();
    test_non_contiguous_transfer_throws();
    std::cout << "Todos los tests de host_buffer/transfer pasaron.\n";
    return 0;
}