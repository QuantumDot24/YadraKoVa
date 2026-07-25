#pragma once
#include "core/memory.hpp"
#include "core/host_buffer.hpp"
#include "core/cuda_error.hpp"
#include "core/stream.hpp"
#include "core/executor.hpp"
#include <vector>
#include <numeric>
#include <cassert>
#include <stdexcept>
#include <cuda_runtime.h>
#include <random>

#include "dtype_utils.hpp"

namespace yadrakova::core
{
    using Shape = std::vector<int64_t>;
    using Strides = std::vector<int64_t>;

    Strides contiguous_strides(const Shape& shape);

    template <typename T = __nv_bfloat16>
    class Tensor
    {
    public:
        static constexpr DType dtype = dtype_traits<T>::value;

        explicit Tensor(Shape shape, MemoryPool& pool = default_pool())
            : shape_(std::move(shape))
              , strides_(contiguous_strides(shape_))
              , offset_(0)
        {
            buffer_ = pool.allocate(static_cast<size_t>(numel()) * sizeof(T));
            data_ptr_ = static_cast<T*>(buffer_->ptr);
        }

        Tensor(std::shared_ptr<DeviceBuffer> buffer, Shape shape, Strides strides, int64_t offset)
            : buffer_(std::move(buffer))
              , shape_(std::move(shape))
              , strides_(std::move(strides))
              , offset_(offset)
        {
            data_ptr_ = static_cast<T*>(buffer_->ptr) + offset_;
        }

        static Tensor<T> from_nested(std::initializer_list<std::initializer_list<T>> rows,
                                     MemoryPool& pool = default_pool())
        {
            auto num_rows = static_cast<int64_t>(rows.size());
            if (num_rows == 0) throw std::runtime_error("from_nested: empty list");

            auto num_cols = static_cast<int64_t>(rows.begin()->size());
            std::vector<T> flat;
            flat.reserve(num_rows * num_cols);
            for (const auto& row : rows)
            {
                if (static_cast<int64_t>(row.size()) != num_cols)
                    throw std::runtime_error("from_nested: all rows must have the same number of columns");
                for (const T& v : row) flat.push_back(v);
            }

            Tensor<T> t({num_rows, num_cols}, pool);
            t.to_device(flat.data(), flat.size());
            return t;
        }

        static Tensor<T> from_vector(const std::vector<T>& data, Shape shape,
                                     MemoryPool& pool = default_pool())
        {
            Tensor<T> t(std::move(shape), pool);
            t.to_device(data.data(), data.size());
            return t;
        }

        static Tensor<T> randn(Shape shape, unsigned seed = 42, MemoryPool& pool = default_pool())
        {
            Tensor<T> t(shape, pool);
            int64_t n = t.numel();

            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            std::vector<T> host(n);
            for (auto& v : host) v = static_cast<T>(dist(rng));

            t.to_device(host.data(), host.size());
            return t;
        }

        std::vector<T> to_vector() const
        {
            std::vector<T> host(static_cast<size_t>(numel()));
            to_host(host.data(), host.size());
            return host;
        }

        T* data() { return data_ptr_; }
        const T* data() const { return data_ptr_; }
        [[nodiscard]] const Shape& shape() const { return shape_; }
        [[nodiscard]] const Strides& strides() const { return strides_; }
        [[nodiscard]] int64_t offset() const { return offset_; }
        [[nodiscard]] size_t ndim() const { return shape_.size(); }

        [[nodiscard]] int64_t numel() const
        {
            return std::accumulate(shape_.begin(), shape_.end(), int64_t{1}, std::multiplies<>());
        }

        [[nodiscard]] bool is_contiguous() const { return strides_ == contiguous_strides(shape_); }

        Tensor transpose(int dim0, int dim1) const
        {
            assert(dim0 >= 0 && dim0 < (int)ndim() && dim1 >= 0 && dim1 < (int)ndim());
            Shape new_shape = shape_;
            Strides new_strides = strides_;
            std::swap(new_shape[dim0], new_shape[dim1]);
            std::swap(new_strides[dim0], new_strides[dim1]);
            return Tensor(buffer_, std::move(new_shape), std::move(new_strides), offset_);
        }

        Tensor permute(const std::vector<int>& dims) const
        {
            assert(dims.size() == ndim());
            Shape new_shape(ndim());
            Strides new_strides(ndim());
            for (size_t i = 0; i < ndim(); ++i)
            {
                new_shape[i] = shape_[dims[i]];
                new_strides[i] = strides_[dims[i]];
            }
            return Tensor(buffer_, std::move(new_shape), std::move(new_strides), offset_);
        }

        Tensor slice(int dim, int64_t start, int64_t length) const
        {
            assert(dim >= 0 && dim < (int)ndim());
            assert(start + length <= shape_[dim]);
            Shape new_shape = shape_;
            new_shape[dim] = length;
            int64_t new_offset = offset_ + start * strides_[dim];
            return Tensor(buffer_, std::move(new_shape), strides_, new_offset);
        }

        Tensor reshape(Shape new_shape) const
        {
            if (!is_contiguous())
            {
                throw std::runtime_error(
                    "reshape() requires contiguous tensor; current stride pattern does not allow it without copying.");
            }
            assert(std::accumulate(new_shape.begin(), new_shape.end(), int64_t{1}, std::multiplies<>()) == numel());
            return Tensor(buffer_, std::move(new_shape), contiguous_strides(new_shape), offset_);
        }

        Tensor gelu(Stream& stream = default_stream()) const
        {
            if (!is_contiguous())
                throw std::runtime_error(
                    "gelu: non-contiguous tensor (view from transpose/permute/slice). "
                    "Materialize a contiguous copy before applying gelu.");

            Tensor<T> out(shape_);

            const T* in_ptr = this->data_ptr_;
            T* out_ptr = out.data_ptr_;
            int64_t n = numel();

            std::vector<void*> args = {&in_ptr, &out_ptr, &n};

            Executor::execute<T>("gelu", DimMap{{"n", n}}, args, stream);
            return out;
        }

        Tensor softmax(Stream& stream = default_stream()) const
        {
            if (ndim() != 2)
                throw std::runtime_error(
                    "softmax: expected 2D tensor (ndim()=" + std::to_string(ndim()) + ")");
            if (!is_contiguous())
                throw std::runtime_error(
                    "softmax: non-contiguous tensor (view from transpose/permute/slice). "
                    "Materialize a contiguous copy before applying softmax.");

            const int64_t rows = shape_[0];
            const int64_t cols = shape_[1];

            Tensor<T> out(shape_);

            const T* in_ptr = this->data_ptr_;
            T* out_ptr = out.data_ptr_;

            std::vector<void*> args = {&in_ptr, &out_ptr, (void*)&rows, (void*)&cols};

            Executor::execute<T>("softmax", DimMap{{"rows", rows}, {"cols", cols}}, args, stream);
            return out;
        }

        Tensor matmul(const Tensor<T>& B, Stream& stream) const
        {
            if (ndim() != 2 || B.ndim() != 2)
            {
                throw std::runtime_error(
                    "matmul: only 2D tensors supported (A.ndim()=" +
                    std::to_string(ndim()) + ", B.ndim()=" + std::to_string(B.ndim()) + ")");
            }
            if (shape_[1] != B.shape_[0])
            {
                throw std::runtime_error(
                    "matmul: incompatible shapes (" + std::to_string(shape_[0]) + "x" +
                    std::to_string(shape_[1]) + ") @ (" + std::to_string(B.shape_[0]) + "x" +
                    std::to_string(B.shape_[1]) + ")");
            }
            if (!is_contiguous() || !B.is_contiguous())
            {
                throw std::runtime_error(
                    "matmul: both operands must be contiguous (materialize a copy "
                    "if they come from transpose/permute/slice)");
            }

            const int64_t M = shape_[0];
            const int64_t K = shape_[1];
            const int64_t N = B.shape_[1];

            Tensor<T> C({M, N});

            const T* a_ptr = this->data_ptr_;
            const T* b_ptr = B.data_ptr_;
            T* c_ptr = C.data_ptr_;

            std::vector<void*> args = {
                &a_ptr, &b_ptr, &c_ptr, (void*)&M, (void*)&N, (void*)&K
            };

            const char* kernel_name = "matmul_wmma";

            Executor::execute<T>(kernel_name, DimMap{{"M", M}, {"N", N}, {"K", K}}, args, stream);
            return C;
        }

        void to_device(const T* host_ptr, size_t count)
        {
            check_contiguous_for_transfer("to_device");
            if (count != static_cast<size_t>(numel()))
                throw std::runtime_error("to_device: size mismatch (expected " +
                    std::to_string(numel()) + ", received " + std::to_string(count) + ")");
            CUDA_CHECK(cudaMemcpy(data_ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
        }

        void to_host(T* host_ptr, size_t count) const
        {
            check_contiguous_for_transfer("to_host");
            if (count != static_cast<size_t>(numel()))
                throw std::runtime_error("to_host: size mismatch (expected " +
                    std::to_string(numel()) + ", received " + std::to_string(count) + ")");
            CUDA_CHECK(cudaMemcpy(host_ptr, data_ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
        }

        void to_device(const HostBuffer<T>& host_buf) { to_device(host_buf.data(), host_buf.size()); }
        void to_host(HostBuffer<T>& host_buf) const { to_host(host_buf.data(), host_buf.size()); }

        void to_device_async(const HostBuffer<T>& host_buf, Stream& stream)
        {
            check_contiguous_for_transfer("to_device_async");
            if (host_buf.size() != static_cast<size_t>(numel()))
                throw std::runtime_error("to_device_async: size mismatch");
            CUDA_CHECK(cudaMemcpyAsync(data_ptr_, host_buf.data(), host_buf.bytes(),
                cudaMemcpyHostToDevice, stream.raw()));
        }

        void to_host_async(HostBuffer<T>& host_buf, Stream& stream) const
        {
            check_contiguous_for_transfer("to_host_async");
            if (host_buf.size() != static_cast<size_t>(numel()))
                throw std::runtime_error("to_host_async: size mismatch");
            CUDA_CHECK(cudaMemcpyAsync(host_buf.data(), data_ptr_, host_buf.bytes(),
                cudaMemcpyDeviceToHost, stream.raw()));
        }

    private:
        void check_contiguous_for_transfer(const char* who) const
        {
            if (!is_contiguous())
            {
                throw std::runtime_error(
                    std::string(who) + ": non-contiguous tensor (view from transpose/permute/slice). "
                    "Materialize a contiguous copy before transfer.");
            }
        }

        std::shared_ptr<DeviceBuffer> buffer_;
        T* data_ptr_ = nullptr;
        Shape shape_;
        Strides strides_;
        int64_t offset_ = 0;
    };
} // namespace yadrakova::core
