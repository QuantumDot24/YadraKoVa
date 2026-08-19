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

void test_tensor_contiguous_kernel()
{
    Stream stream;

    // 1. Create a 2x3 tensor on CPU/GPU with values:
    // [ 1.0, 2.0, 3.0 ]
    // [ 4.0, 5.0, 6.0 ]
    std::vector<float> host_data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    Tensor<float> A = Tensor<float>::from_vector(host_data, {2, 3});
    assert(A.is_contiguous() == true);

    // 2. Transpose the matrix (results in shape 3x2 with inverted strides)
    Tensor<float> A_T = A.transpose(0, 1);
    assert(A_T.shape()[0] == 3);
    assert(A_T.shape()[1] == 2);
    assert(A_T.is_contiguous() == false);

    // 3. Force contiguous materialization via the CUDA kernel
    Tensor<float> A_T_contig = A_T.contiguous(Backend::Auto, stream);
    stream.synchronize();

    assert(A_T_contig.is_contiguous() == true);

    // 4. Download to CPU vector and verify the linear order in VRAM
    // The mathematical transpose of A is:
    // [ 1.0, 4.0 ]
    // [ 2.0, 5.0 ]
    // [ 3.0, 6.0 ]
    // Therefore, in linear contiguous memory it must be stored as: [1, 4, 2, 5, 3, 6]
    std::vector<float> result = A_T_contig.to_vector();
    std::vector<float> expected = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};

    assert(result.size() == expected.size());
    for (size_t i = 0; i < result.size(); ++i)
    {
        assert(result[i] == expected[i]);
    }

    std::cout << "[OK] tensor_contiguous_kernel (successful GPU reordering)\n";
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

    std::cout << "[OK] pool_outlives_tensor (actual memory remains alive after pool destruction)\n";

    surviving_tensor.reset();
}

int main()
{
    test_contiguous_alloc();
    test_transpose_no_copy();
    test_slice_offset();
    test_reshape_throws_on_noncontiguous();
    test_pool_outlives_tensor();
    test_tensor_contiguous_kernel();
    std::cout << "All tensor.cpp tests passed.\n";
    return 0;
}