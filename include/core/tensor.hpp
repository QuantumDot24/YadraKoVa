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
    using Strides = std::vector<int64_t>; // en elementos, no bytes

    Strides contiguous_strides(const Shape& shape);

    template <typename T = __nv_bfloat16>
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

        // Como torch.tensor([[1,2],[3,4]]) -- para tus sistemas del libro de
        // algebra lineal. Infiera shape {rows, cols} de la lista anidada;
        // lanza si las filas no tienen el mismo ancho (matriz mal formada).
        static Tensor<T> from_nested(std::initializer_list<std::initializer_list<T>> rows,
                                     MemoryPool& pool = default_pool())
        {
            int64_t n_rows = static_cast<int64_t>(rows.size());
            if (n_rows == 0) throw std::runtime_error("from_nested: lista vacia");

            int64_t n_cols = static_cast<int64_t>(rows.begin()->size());
            std::vector<T> flat;
            flat.reserve(n_rows * n_cols);
            for (const auto& row : rows)
            {
                if (static_cast<int64_t>(row.size()) != n_cols)
                    throw std::runtime_error("from_nested: todas las filas deben tener el mismo numero de columnas");
                for (const T& v : row) flat.push_back(v);
            }

            Tensor<T> t({n_rows, n_cols}, pool);
            t.to_device(flat.data(), flat.size());
            return t;
        }

        // Vector plano -> Tensor con el shape que quieras. El caso general
        // detras de from_nested, util cuando ya traes datos en std::vector.
        static Tensor<T> from_vector(const std::vector<T>& data, Shape shape,
                                     MemoryPool& pool = default_pool())
        {
            Tensor<T> t(std::move(shape), pool);
            t.to_device(data.data(), data.size());
            return t;
        }

        // Como torch.randn(shape) -- llena en CPU con normal(0,1) y sube a
        // device. seed fijo por default para tests reproducibles (cambia esto
        // si quieres aleatoriedad real entre corridas).
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

        // Baja el tensor completo a un std::vector<T> en host -- el reverso
        // de from_vector/from_nested. Requiere contiguo, igual que to_host.
        std::vector<T> to_vector() const
        {
            std::vector<T> host(static_cast<size_t>(numel()));
            to_host(host.data(), host.size());
            return host;
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
        // --- activaciones / normalizacion: dispatch autogenerado via Executor ---

        // Elementwise, cualquier shape -- opera sobre numel() elementos planos.
        Tensor gelu(Stream& stream = default_stream()) const
        {
            if (!is_contiguous())
                throw std::runtime_error(
                    "gelu: tensor no contiguo (view de transpose/permute/slice). "
                    "Materializa una copia contigua antes de aplicar gelu.");

            Tensor<T> out(shape_);

            const T* in_ptr = this->data_ptr_;
            T* out_ptr = out.data_ptr_;
            int64_t n = numel();

            std::vector<void*> args = { &in_ptr, &out_ptr, &n };

            Executor::execute<T>("gelu", DimMap{{"n", n}}, args, stream);
            return out;
        }

        // Softmax por fila -- requiere 2D. La version actual del kernel asume
        // cols <= 32 (un warp por fila); shapes mas anchos necesitaran un
        // kernel multi-warp aparte mas adelante.
        Tensor softmax(Stream& stream = default_stream()) const
        {
            if (ndim() != 2)
                throw std::runtime_error(
                    "softmax: se espera un tensor 2D (ndim()=" + std::to_string(ndim()) + ")");
            if (!is_contiguous())
                throw std::runtime_error(
                    "softmax: tensor no contiguo (view de transpose/permute/slice). "
                    "Materializa una copia contigua antes de aplicar softmax.");

            const int64_t rows = shape_[0];
            const int64_t cols = shape_[1];

            Tensor<T> out(shape_);

            const T* in_ptr = this->data_ptr_;
            T* out_ptr = out.data_ptr_;

            std::vector<void*> args = { &in_ptr, &out_ptr, (void*)&rows, (void*)&cols };

            Executor::execute<T>("softmax", DimMap{{"rows", rows}, {"cols", cols}}, args, stream);
            return out;
        }
        // --- algebra: dispatch autogenerado via Executor ---
        //
        // Solo 2D por ahora (el kernel naive no soporta batching).
        // Asincrona respecto al host -- C queda listo logicamente pero
        // el computo puede seguir en vuelo en `stream` hasta el
        // siguiente synchronize().
        Tensor matmul(const Tensor<T>& B, Stream& stream) const
        {
            if (ndim() != 2 || B.ndim() != 2)
            {
                throw std::runtime_error(
                    "matmul: solo se soportan tensores 2D (A.ndim()=" +
                    std::to_string(ndim()) + ", B.ndim()=" + std::to_string(B.ndim()) + ")");
            }
            if (shape_[1] != B.shape_[0])
            {
                throw std::runtime_error(
                    "matmul: shapes incompatibles (" + std::to_string(shape_[0]) + "x" +
                    std::to_string(shape_[1]) + ") @ (" + std::to_string(B.shape_[0]) + "x" +
                    std::to_string(B.shape_[1]) + ")");
            }
            if (!is_contiguous() || !B.is_contiguous())
            {
                throw std::runtime_error(
                    "matmul: ambos operandos deben ser contiguos (materializa una copia "
                    "si vienen de transpose/permute/slice)");
            }

            const int64_t M = shape_[0];
            const int64_t K = shape_[1];
            const int64_t N = B.shape_[1];

            Tensor<T> C({M, N});

            // Punteros LOCALES: cuLaunchKernel espera un arreglo de
            // void*, cada uno apuntando a DONDE VIVE el valor real del
            // argumento -- no al valor en si. Para un argumento puntero
            // (A, B, C) eso significa "puntero a la variable que
            // contiene el puntero de device", no el puntero de device
            // directamente. Estas variables deben seguir vivas hasta
            // que cuLaunchKernel retorne (dentro de Executor::execute,
            // llamado de forma sincrona mas abajo) -- no hace falta que
            // sobrevivan a la ejecucion async del kernel en si.
            const T* a_ptr = this->data_ptr_;
            const T* b_ptr = B.data_ptr_;
            T* c_ptr = C.data_ptr_;

            // Orden EXACTO de matmul_kernel(A, B, C, M, N, K) -- debe
            // coincidir con el orden declarado en matmul.yaml (args:).
            std::vector<void*> args = {
                &a_ptr, &b_ptr, &c_ptr, (void*)&M, (void*)&N, (void*)&K
            };

            const char* kernel_name = "matmul_wmma";

            Executor::execute<T>(kernel_name, DimMap{{"M", M}, {"N", N}, {"K", K}}, args, stream);
            return C;
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
