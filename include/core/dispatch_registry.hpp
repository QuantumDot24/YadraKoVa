#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace yadrakova::core {

struct Dim3 {
    unsigned int x = 1, y = 1, z = 1;
};

struct DispatchDims {
    Dim3 grid;
    Dim3 block;
    unsigned int shared_mem_bytes = 0;
};

// Firma fija por ahora: M, N, K cubren matmul y la mayoria de kernels
// tipo BLAS. Si mas adelante aparece un kernel con otra forma de
// parametrizar su tamano (p.ej. reduce con un solo N, o conv con
// varios parametros), se puede generalizar a una firma basada en un
// mapa nombre->valor sin romper esta firma para los que ya existen.
using DispatchFunc = std::function<DispatchDims(int64_t M, int64_t N, int64_t K)>;

class DispatchRegistry {
public:
    static DispatchRegistry& instance() {
        static DispatchRegistry inst;
        return inst;
    }

    void register_dispatch(const std::string& kernel_name, DispatchFunc fn) {
        table_[kernel_name] = std::move(fn);
    }

    [[nodiscard]] DispatchDims get_dims(const std::string& kernel_name,
                           int64_t M, int64_t N, int64_t K) const {
        auto it = table_.find(kernel_name);
        if (it == table_.end()) {
            throw std::runtime_error(
                "DispatchRegistry: no hay regla de dispatch registrada para kernel '"
                + kernel_name + "' (¿falta el .yaml o no se linkeo con --whole-archive?)");
        }
        return it->second(M, N, K);
    }

    [[nodiscard]] bool has(const std::string& kernel_name) const {
        return table_.contains(kernel_name);
    }

private:
    DispatchRegistry() = default;
    std::unordered_map<std::string, DispatchFunc> table_;
};

} // namespace yadrakova::core