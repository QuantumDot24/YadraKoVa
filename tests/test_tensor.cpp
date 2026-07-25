#include "core/tensor.hpp"
#include <cassert>
#include <iostream>

using namespace yadrakova::core;

void test_contiguous_alloc()
{
    Tensor<float> t({2, 3, 4});
    assert(t.numel() == 24);
    assert(t.is_contiguous());
    std::cout << "[OK] contiguous_alloc\n";
}

static void test_transpose_no_copy()
{
    Tensor<float> t({2, 3, 4});
    const float* original_ptr = t.data();
    Tensor<float> tt = t.transpose(0, 1);
    assert(tt.data() == original_ptr);
    assert(tt.shape()[0] == 3 && tt.shape()[1] == 2);
    assert(!tt.is_contiguous());
    std::cout << "[OK] transpose_no_copy\n";
}

void test_slice_offset()
{
    Tensor<float> t({4, 8});
    Tensor<float> s = t.slice(0, 1, 2);
    assert(s.shape()[0] == 2 && s.shape()[1] == 8);
    assert(s.data() == t.data() + 8);
    std::cout << "[OK] slice_offset\n";
}

void test_reshape_throws_on_noncontiguous()
{
    Tensor<float> t({2, 3});
    Tensor<float> tt = t.transpose(0, 1);
    bool threw = false;
    try
    {
        tt.reshape({6});
    }
    catch (const std::runtime_error&)
    {
        threw = true;
    }
    assert(threw);
    std::cout << "[OK] reshape_throws_on_noncontiguous\n";
}

void test_pool_outlives_tensor()
{
    std::shared_ptr<Tensor<float>> surviving_tensor;
    {
        MemoryPool local_pool(0);
        surviving_tensor = std::make_shared<Tensor<float>>(Shape{4, 4}, local_pool);
        assert(local_pool.outstanding_buffers() == 1);
    }


    assert(surviving_tensor->numel() == 16);
    assert(surviving_tensor->data() != nullptr);

    cudaMemset(surviving_tensor->data(), 0, surviving_tensor->numel() * sizeof(float));
    cudaError_t err = cudaGetLastError();
    assert(err == cudaSuccess);

    std::cout << "[OK] pool_outlives_tensor (memoria real sigue viva tras destruir el pool)\n";

    surviving_tensor.reset();
}

int main()
{
    test_contiguous_alloc();
    test_transpose_no_copy();
    test_slice_offset();
    test_reshape_throws_on_noncontiguous();
    test_pool_outlives_tensor();
    std::cout << "Todos los tests de tensor.cpp pasaron.\n";
    return 0;
}
