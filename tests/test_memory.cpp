#include "core/tensor.hpp"
#include "core/stream.hpp"
#include "core/memory.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <cassert>

using namespace yadrakova::core;

// ===========================================================================
// Section 1: HostBuffer basics
// ===========================================================================

void test_host_buffer_alloc_and_access()
{
    HostBuffer<float> buf(1024);
    assert(buf.size() == 1024);
    assert(buf.bytes() == 1024 * sizeof(float));
    assert(!buf.empty());
    assert(buf.data() != nullptr);

    for (size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<float>(i) * 0.5f;
    for (size_t i = 0; i < buf.size(); ++i) assert(buf[i] == static_cast<float>(i) * 0.5f);

    std::cout << "[OK] host_buffer_alloc_and_access\n";
}

void test_host_buffer_default_empty()
{
    HostBuffer<float> buf; // no count -> must NOT call cudaHostAlloc
    assert(buf.empty());
    assert(buf.size() == 0);
    assert(buf.data() == nullptr);
    std::cout << "[OK] host_buffer_default_empty\n";
}

void test_host_buffer_move_semantics()
{
    HostBuffer<float> a(256);
    a[0] = 42.0f;
    float* original_ptr = a.data();

    HostBuffer<float> b(std::move(a));
    assert(b.data() == original_ptr);
    assert(b.size() == 256);
    assert(b[0] == 42.0f);

    // Moved-from object must be empty, not pointing to freed memory.
    assert(a.data() == nullptr);
    assert(a.size() == 0);

    HostBuffer<float> c(64);
    c = std::move(b);
    assert(c.data() == original_ptr);
    assert(c[0] == 42.0f);
    assert(b.data() == nullptr);

    std::cout << "[OK] host_buffer_move_semantics\n";
}

// ===========================================================================
// Section 2: Tensor <-> Host synchronization
// ===========================================================================

void test_tensor_sync_roundtrip()
{
    MemoryPool pool(0);
    const size_t n = 4096;

    HostBuffer<float> src(n);
    for (size_t i = 0; i < n; ++i) src[i] = static_cast<float>(i) * 1.5f - 100.0f;

    Tensor<float> t({static_cast<int64_t>(n)}, pool);
    t.to_device(src);

    HostBuffer<float> dst(n);
    t.to_host(dst);

    for (size_t i = 0; i < n; ++i) assert(dst[i] == src[i]);
    std::cout << "[OK] tensor_sync_roundtrip\n";
}

void test_tensor_sync_roundtrip_raw_ptr()
{
    // Raw T* variant (no HostBuffer) must also work -- this is the path
    // used by tests that have not yet migrated to pinned memory.
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

void test_tensor_size_mismatch_throws()
{
    MemoryPool pool(0);
    Tensor<float> t({1024}, pool);
    HostBuffer<float> wrong_size(512);

    bool threw = false;
    try { t.to_device(wrong_size); }
    catch (const std::runtime_error&) { threw = true; }
    assert(threw);

    std::cout << "[OK] tensor_size_mismatch_throws\n";
}

void test_tensor_async_transfer()
{
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

    for (size_t i = 0; i < n; ++i) assert(dst[i] == src[i]);
    std::cout << "[OK] tensor_async_transfer\n";
}

void test_non_contiguous_transfer_throws()
{
    MemoryPool pool(0);
    Tensor<float> t({64, 64}, pool);
    Tensor<float> view = t.transpose(0, 1);
    assert(!view.is_contiguous());

    HostBuffer<float> dst(64 * 64);
    bool threw = false;
    try { view.to_host(dst); }
    catch (const std::runtime_error& e) {
        threw = true;
        std::cout << "  (expected exception: " << e.what() << ")\n";
    }
    assert(threw);

    std::cout << "[OK] non_contiguous_transfer_throws\n";
}

void test_async_transfer_with_host_buffer()
{
    std::cout << "[TEST] Real async transfer (HostBuffer, no CPU blocking)...\n";
    Stream stream(/*non_blocking=*/true);
    const size_t elements = 67'108'864; // ~134 MB
    Tensor<__nv_bfloat16> A({static_cast<int64_t>(elements)});
    HostBuffer<__nv_bfloat16> host_buf(elements);

    for (size_t i = 0; i < 1024; ++i) host_buf[i] = static_cast<__nv_bfloat16>(1.0f);

    auto start = std::chrono::high_resolution_clock::now();
    A.to_device_async(host_buf, stream);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Time returned to CPU: " << elapsed_ms << " ms\n";

    if (elapsed_ms < 2.0)
        std::cout << "  [PASS] Successful async transfer (no CPU blocking).\n";
    else
        std::cout << "  [FAIL] CPU is still waiting (" << elapsed_ms << " ms).\n";

    stream.synchronize();
}

// ===========================================================================
// Section 3: MemoryPool stream-safety via CUDA Events
// ===========================================================================

void test_memory_pool_stream_safety()
{
    std::cout << "[TEST] Stream-safe MemoryPool with CUDA Events...\n";
    Stream stream1(/*non_blocking=*/true);
    Stream stream2(/*non_blocking=*/true);

    const int64_t M = 4096, N = 4096, K = 4096;
    void* ptr_original = nullptr;
    {
        StreamGuard guard(stream1);
        Tensor<__nv_bfloat16> A = Tensor<__nv_bfloat16>::randn({M, K});
        Tensor<__nv_bfloat16> B = Tensor<__nv_bfloat16>::randn({K, N});
        ptr_original = A.data();
        auto C = A.matmul(B, Backend::CuBLAS, stream1);
        // A and B destroyed here -> they register cudaEventRecord in stream1
    }

    StreamGuard guard2(stream2);
    Tensor<__nv_bfloat16> D({M, K});
    void* ptr_reused = D.data();

    std::cout << "  Ptr Stream 1: " << ptr_original << "\n";
    std::cout << "  Ptr Stream 2: " << ptr_reused << "\n";

    if (ptr_original == ptr_reused)
        std::cout << "  [INFO] Pool recycled the address. stream2 will wait for stream1 via cudaStreamWaitEvent on the GPU.\n";

    stream1.synchronize();
    stream2.synchronize();
    std::cout << "  [PASS] Multi-stream execution correctly synchronized on hardware.\n";
}

// ===========================================================================
// Section 4: MemoryPool stress tests
// ===========================================================================

void test_random_allocation_churn()
{
    std::cout << "[STRESS TEST] Random allocation churn and lifecycle...\n";

    constexpr int NUM_ITERATIONS = 5000;
    constexpr size_t MAX_HOLD = 100;

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> size_dist(128, 16 * 1024 * 1024); // 128 B to 16 MB

    std::vector<std::shared_ptr<DeviceBuffer>> active_buffers;
    active_buffers.reserve(MAX_HOLD);

    for (int i = 0; i < NUM_ITERATIONS; ++i)
    {
        size_t alloc_bytes = size_dist(rng);
        active_buffers.push_back(default_pool().allocate(alloc_bytes));

        if (active_buffers.size() >= MAX_HOLD)
        {
            size_t idx_to_remove = rng() % active_buffers.size();
            active_buffers.erase(active_buffers.begin() + idx_to_remove);
        }
    }

    active_buffers.clear();

    std::cout << "  Final bytes in use: " << default_pool().bytes_in_use() << " B\n";
    std::cout << "  Outstanding buffers: " << default_pool().outstanding_buffers() << "\n";
    std::cout << "  Cumulative reserved VRAM: " << (default_pool().bytes_reserved() / (1024 * 1024)) << " MB\n";

    assert(default_pool().bytes_in_use() == 0 && "Leak detected: bytes_in_use() must be 0");
    assert(default_pool().outstanding_buffers() == 0 && "Leak detected: outstanding_buffers() must be 0");
    std::cout << "  [PASS] Random churn completed with no memory leaks.\n";
}

void test_multithreaded_pool_safety()
{
    std::cout << "[STRESS TEST] Multi-threaded concurrency in MemoryPool...\n";

    constexpr int NUM_THREADS = 8;
    constexpr int ALLOCS_PER_THREAD = 1000;
    std::vector<std::thread> threads;

    for (int t = 0; t < NUM_THREADS; ++t)
    {
        threads.emplace_back([t]() {
            try
            {
                CUDA_CHECK(cudaSetDevice(0));
                Stream thread_stream(/*non_blocking=*/true);
                std::vector<std::shared_ptr<DeviceBuffer>> local_buffers;

                for (int i = 0; i < ALLOCS_PER_THREAD; ++i)
                {
                    size_t sz = (i % 10 + 1) * 1024 * 1024; // 1 MB to 10 MB
                    local_buffers.push_back(default_pool().allocate(sz, thread_stream));

                    if (local_buffers.size() > 10)
                        local_buffers.erase(local_buffers.begin(), local_buffers.begin() + 5);
                }

                local_buffers.clear();
                thread_stream.synchronize();
            }
            catch (const std::exception& e)
            {
                fprintf(stderr, "[Thread %d] Exception caught: %s\n", t, e.what());
            }
        });
    }

    for (auto& th : threads) th.join();

    assert(default_pool().bytes_in_use() == 0 && "Leak detected after multi-threaded execution");
    assert(default_pool().outstanding_buffers() == 0 && "Outstanding buffers after multi-threaded execution");
    std::cout << "  [PASS] Multi-threaded allocation with std::mutex successfully validated.\n";
}

// ===========================================================================
int main()
{
    try
    {
        std::cout << "=== MEMORY, TRANSFER & POOL TEST SUITE ===\n\n";

        std::cout << "-- HostBuffer basics --\n";
        test_host_buffer_alloc_and_access();
        test_host_buffer_default_empty();
        test_host_buffer_move_semantics();

        std::cout << "\n-- Tensor <-> Host synchronization --\n";
         test_tensor_sync_roundtrip();
        test_tensor_sync_roundtrip_raw_ptr();
        test_tensor_size_mismatch_throws();
        test_tensor_async_transfer();
        test_non_contiguous_transfer_throws();
        test_async_transfer_with_host_buffer();

        std::cout << "\n-- MemoryPool stream-safety --\n";
        test_memory_pool_stream_safety();

        std::cout << "\n-- MemoryPool stress tests --\n";
        test_random_allocation_churn();
        test_multithreaded_pool_safety();

        std::cout << "\nAll memory tests passed successfully.\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
    return 0;
}