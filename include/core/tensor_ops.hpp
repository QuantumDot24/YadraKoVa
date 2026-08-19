#pragma once
#include "core/tensor.hpp"
#include "core/executor.hpp"
#include "core/backend_dispatch.hpp"
#include <string>

namespace yadrakova::core
{
    // ------------------------------------------------------------------
    // Funciones libres -- toda la logica real. Mismo molde en las 6:
    // validar -> armar args -> dispatch_op -> devolver salida.
    // ------------------------------------------------------------------
    template <typename T>
    Tensor<T> gelu(const Tensor<T>& in, Backend backend = Backend::Auto, Stream& stream = default_stream())
    {
        if (!in.is_contiguous())
            throw std::runtime_error(
                "gelu: non-contiguous tensor (view from transpose/permute/slice). "
                "Materialize a contiguous copy before applying gelu.");

        Tensor<T> out(in.shape());
        const T* in_ptr = in.data();
        T* out_ptr = out.data();
        int64_t n = in.numel();

        std::vector<void*> args = {&in_ptr, &out_ptr, &n};
        dispatch_op<T>(Op::Gelu, "gelu", DimMap{{"n", n}}, args, backend, stream);
        return out;
    }

    template <typename T>
    Tensor<T> softmax(const Tensor<T>& in, Backend backend = Backend::Auto, Stream& stream = default_stream())
    {
        if (in.ndim() != 2)
            throw std::runtime_error("softmax: expected 2D tensor (ndim()=" + std::to_string(in.ndim()) + ")");
        if (!in.is_contiguous())
            throw std::runtime_error(
                "softmax: non-contiguous tensor (view from transpose/permute/slice). "
                "Materialize a contiguous copy before applying softmax.");

        const int64_t rows = in.shape()[0];
        const int64_t cols = in.shape()[1];

        Tensor<T> out(in.shape());
        const T* in_ptr = in.data();
        T* out_ptr = out.data();

        std::vector<void*> args = {&in_ptr, &out_ptr, (void*)&rows, (void*)&cols};
        dispatch_op<T>(Op::Softmax, "softmax", DimMap{{"rows", rows}, {"cols", cols}}, args, backend, stream);
        return out;
    }

    template <typename T>
    Tensor<T> matmul(const Tensor<T>& A, const Tensor<T>& B,
                      Backend backend = Backend::Auto, Stream& stream = default_stream())
    {
        if (A.ndim() != 2 || B.ndim() != 2)
        {
            throw std::runtime_error(
                "matmul: only 2D tensors supported (A.ndim()=" +
                std::to_string(A.ndim()) + ", B.ndim()=" + std::to_string(B.ndim()) + ")");
        }
        if (A.shape()[1] != B.shape()[0])
        {
            throw std::runtime_error(
                "matmul: incompatible shapes (" + std::to_string(A.shape()[0]) + "x" +
                std::to_string(A.shape()[1]) + ") @ (" + std::to_string(B.shape()[0]) + "x" +
                std::to_string(B.shape()[1]) + ")");
        }
        if (!A.is_contiguous() || !B.is_contiguous())
        {
            throw std::runtime_error(
                "matmul: both operands must be contiguous (materialize a copy "
                "if they come from transpose/permute/slice)");
        }

        const int64_t M = A.shape()[0];
        const int64_t K = A.shape()[1];
        const int64_t N = B.shape()[1];

        Tensor<T> C({M, N});
        const T* a_ptr = A.data();
        const T* b_ptr = B.data();
        T* c_ptr = C.data();

        std::vector<void*> args = {&a_ptr, &b_ptr, &c_ptr, (void*)&M, (void*)&N, (void*)&K};
        dispatch_op<T>(Op::MatMul, "matmul_wmma", DimMap{{"M", M}, {"N", N}, {"K", K}}, args, backend, stream);
        return C;
    }

    template <typename T>
    Tensor<T> add(const Tensor<T>& A, const Tensor<T>& B,
                  Backend backend = Backend::Auto, Stream& stream = default_stream())
    {
        if (A.numel() != B.numel())
            throw std::runtime_error("add: numel mismatch between operands");
        if (!A.is_contiguous() || !B.is_contiguous())
            throw std::runtime_error("add: both operands must be contiguous (materialize a copy if needed)");

        Tensor<T> out(A.shape());
        const void* a_ptr = A.data();
        const void* b_ptr = B.data();
        void* c_ptr = out.data();
        int64_t n = A.numel();

        std::vector<void*> args = {&a_ptr, &b_ptr, &c_ptr, &n};
        dispatch_op<T>(Op::Add, "add", DimMap{{"n", n}}, args, backend, stream);
        return out;
    }

    template <typename T>
    Tensor<T> mul(const Tensor<T>& A, const Tensor<T>& B,
                  Backend backend = Backend::Auto, Stream& stream = default_stream())
    {
        if (A.numel() != B.numel())
            throw std::runtime_error("mul: numel mismatch between operands");
        if (!A.is_contiguous() || !B.is_contiguous())
            throw std::runtime_error("mul: both operands must be contiguous (materialize a copy if needed)");

        Tensor<T> out(A.shape());
        const void* a_ptr = A.data();
        const void* b_ptr = B.data();
        void* c_ptr = out.data();
        int64_t n = A.numel();

        std::vector<void*> args = {&a_ptr, &b_ptr, &c_ptr, &n};
        dispatch_op<T>(Op::Mul, "mul", DimMap{{"n", n}}, args, backend, stream);
        return out;
    }

    template <typename T>
    Tensor<T> contiguous(const Tensor<T>& in, Backend backend = Backend::Auto, Stream& stream = default_stream())
    {
        if (in.is_contiguous())
            return in; // shallow copy: alias del mismo buffer, igual que el resto de las vistas

        Tensor<T> out(in.shape());

        constexpr int MAX_DIMS = 8;
        struct TensorViewInfo
        {
            int64_t shape[MAX_DIMS];
            int64_t strides[MAX_DIMS];
            int ndim;
        };

        TensorViewInfo view_info{};
        view_info.ndim = static_cast<int>(in.ndim());
        for (size_t i = 0; i < in.ndim(); ++i)
        {
            view_info.shape[i] = in.shape()[i];
            view_info.strides[i] = in.strides()[i];
        }

        int64_t total_elements = in.numel();
        const T* in_ptr = in.data();
        T* out_ptr = out.data();

        std::vector<void*> args = {
            (void*)&in_ptr, (void*)&out_ptr, (void*)&view_info, (void*)&total_elements
        };

        dispatch_op<T>(Op::Contiguous, "contiguous", DimMap{{"numel", total_elements}}, args, backend, stream);
        return out;
    }

    // ------------------------------------------------------------------
    // Wrappers fuera de linea de los metodos declarados en tensor.hpp.
    // A.matmul(B) / A.add(B, Backend::CuTensor) / A.contiguous() siguen
    // andando igual, no rompe tu test standalone.
    // ------------------------------------------------------------------
    template <typename T>
    Tensor<T> Tensor<T>::gelu(Backend backend, Stream& stream) const
    {
        return yadrakova::core::gelu(*this, backend, stream);
    }

    template <typename T>
    Tensor<T> Tensor<T>::softmax(Backend backend, Stream& stream) const
    {
        return yadrakova::core::softmax(*this, backend, stream);
    }

    template <typename T>
    Tensor<T> Tensor<T>::matmul(const Tensor<T>& B, Backend backend, Stream& stream) const
    {
        return yadrakova::core::matmul(*this, B, backend, stream);
    }

    template <typename T>
    Tensor<T> Tensor<T>::add(const Tensor<T>& B, Backend backend, Stream& stream) const
    {
        return yadrakova::core::add(*this, B, backend, stream);
    }

    template <typename T>
    Tensor<T> Tensor<T>::mul(const Tensor<T>& B, Backend backend, Stream& stream) const
    {
        return yadrakova::core::mul(*this, B, backend, stream);
    }

    template <typename T>
    Tensor<T> Tensor<T>::contiguous(Backend backend, Stream& stream) const
    {
        return yadrakova::core::contiguous(*this, backend, stream);
    }
} // namespace yadrakova::core