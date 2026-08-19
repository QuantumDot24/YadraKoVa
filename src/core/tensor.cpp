#include "core/tensor.hpp"
#include "core/tensor_ops.hpp" 
namespace yadrakova::core
{
    Strides contiguous_strides(const Shape& shape)
    {
        Strides strides(shape.size());
        int64_t acc = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
        {
            strides[i] = acc;
            acc *= shape[i];
        }
        return strides;
    }

    template class Tensor<__nv_bfloat16>;
    template class Tensor<float>;
    template class Tensor<__half>;
    template class Tensor<int8_t>;
} // namespace yadrakova::core
