#pragma once
#include "core/memory.h"
#include "core/host_buffer.h"
#include "core/cuda_error.h"
#include "core/stream.h"
#include <vector>
#include <numeric>
#include <cassert>
#include <stdexcept>
#include <cuda_runtime.h>

#include "dtype_utils.h"

namespace yadrakova::core
{
    using Shape = std::vector<int64_t>;
    using Strides = std::vector<int64_t>; // en elementos, no bytes

    Strides contiguous_strides(const Shape& shape);

    template <typename T>
    class Tensor
    {
    public:
        static constexpr DType dtype = dtype_traits<T>::value;

        // Aloca storage nuevo desde el pool.
        explicit Tensor(Shape shape, MemoryPool& pool = default_pool())
            : shape_(std::move(shape))
              , strides_(contiguous_strides(shape_))
              , offset_(0)
        {
            buffer_ = pool.allocate(static_cast<size_t>(numel()) * sizeof(T));
            data_ptr_ = static_cast<T*>(buffer_->ptr);
        }

        // Envuelve un buffer existente (usado internamente por view/transpose/slice).
        // Sin allocation, sin copia.
        Tensor(std::shared_ptr<DeviceBuffer> buffer, Shape shape, Strides strides, int64_t offset)
            : buffer_(std::move(buffer))
              , shape_(std::move(shape))
              , strides_(std::move(strides))
              , offset_(offset)
        {
            data_ptr_ = static_cast<T*>(buffer_->ptr) + offset_;
        }

        T* data() { return data_ptr_; }
        const T* data() const { return data_ptr_; }
        const Shape& shape() const { return shape_; }
        const Strides& strides() const { return strides_; }
        int64_t offset() const { return offset_; }
        size_t ndim() const { return shape_.size(); }

        int64_t numel() const
        {
            return std::accumulate(shape_.begin(), shape_.end(), int64_t{1}, std::multiplies<>());
        }

        bool is_contiguous() const { return strides_ == contiguous_strides(shape_); }

        // --- operaciones de vista, cero copia: solo tocan shape/strides/offset ---

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

        // Reinterpreta storage contiguo bajo un shape nuevo. Lanza excepción
        // si no es contiguo -- reshape() NUNCA copia en silencio.
        Tensor reshape(Shape new_shape) const
        {
            if (!is_contiguous())
            {
                throw std::runtime_error(
                    "reshape() requiere tensor contiguo; el stride pattern actual no lo permite sin copia.");
            }
            assert(std::accumulate(new_shape.begin(), new_shape.end(), int64_t{1}, std::multiplies<>()) == numel());
            return Tensor(buffer_, std::move(new_shape), contiguous_strides(new_shape), offset_);
        }

        // --- transferencias host <-> device ---
        //
        // Requieren contiguidad: un memcpy plano no respeta strides no-contiguos.
        // Si necesitas mover una view (p.ej. resultado de transpose/slice), primero
        // materializa con una copia contigua (eso vendrá con un kernel/cudaMemcpy2D
        // más adelante; por ahora se rechaza explícitamente en vez de copiar mal).

        // Síncronas, memoria host paginable o pinned (T* crudo).
        void to_device(const T* host_ptr, size_t count)
        {
            check_contiguous_for_transfer("to_device");
            if (count != static_cast<size_t>(numel()))
                throw std::runtime_error("to_device: size mismatch (esperado " +
                    std::to_string(numel()) + ", recibido " + std::to_string(count) + ")");
            CUDA_CHECK(cudaMemcpy(data_ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
        }

        void to_host(T* host_ptr, size_t count) const
        {
            check_contiguous_for_transfer("to_host");
            if (count != static_cast<size_t>(numel()))
                throw std::runtime_error("to_host: size mismatch (esperado " +
                    std::to_string(numel()) + ", recibido " + std::to_string(count) + ")");
            CUDA_CHECK(cudaMemcpy(host_ptr, data_ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
        }

        // Overloads de conveniencia con HostBuffer<T> (toma size() directamente).
        void to_device(const HostBuffer<T>& host_buf) { to_device(host_buf.data(), host_buf.size()); }
        void to_host(HostBuffer<T>& host_buf) const { to_host(host_buf.data(), host_buf.size()); }

        // Asíncronas: solo con HostBuffer (pinned). A propósito no hay overload
        // async con T* crudo -- con memoria paginable cudaMemcpyAsync degrada a
        // síncrono en silencio, y eso es justo el bug que esto busca evitar.
        void to_device_async(const HostBuffer<T>& host_buf, Stream& stream)
        {
            check_contiguous_for_transfer("to_device_async");
            if (host_buf.size() != static_cast<size_t>(numel()))
                throw std::runtime_error("to_device_async: size mismatch");
            CUDA_CHECK(cudaMemcpyAsync(data_ptr_, host_buf.data(), host_buf.bytes(),
                cudaMemcpyHostToDevice, stream.raw())); // antes: stream.handle()
        }

        void to_host_async(HostBuffer<T>& host_buf, Stream& stream) const
        {
            check_contiguous_for_transfer("to_host_async");
            if (host_buf.size() != static_cast<size_t>(numel()))
                throw std::runtime_error("to_host_async: size mismatch");
            CUDA_CHECK(cudaMemcpyAsync(host_buf.data(), data_ptr_, host_buf.bytes(),
                cudaMemcpyDeviceToHost, stream.raw())); // antes: stream.handle()
        }

    private:
        void check_contiguous_for_transfer(const char* who) const
        {
            if (!is_contiguous())
            {
                throw std::runtime_error(
                    std::string(who) + ": tensor no contiguo (view de transpose/permute/slice). "
                    "Materializa una copia contigua antes de transferir.");
            }
        }

        std::shared_ptr<DeviceBuffer> buffer_;
        T* data_ptr_ = nullptr;
        Shape shape_;
        Strides strides_;
        int64_t offset_ = 0;
    };
} // namespace yadrakova::core
