#include "core/tensor.h"

namespace yadrakova::core {

    Strides contiguous_strides(const Shape& shape) {
        Strides strides(shape.size());
        int64_t acc = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            strides[i] = acc;
            acc *= shape[i];
        }
        return strides;
    }

    // Instanciación explícita: solo estos 4 dtypes existen para Tensor<T>.
    // Usar Tensor<double> en otro .cpp falla en link time -- a propósito,
    // para que un dtype no soportado no se cuele silenciosamente.
    template class Tensor<__nv_bfloat16>;
    template class Tensor<float>;
    template class Tensor<__half>;
    template class Tensor<int8_t>;

} // namespace yadrakova::core