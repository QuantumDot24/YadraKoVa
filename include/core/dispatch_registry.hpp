#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace yadrakova::core
{
    struct Dim3
    {
        unsigned int x = 1, y = 1, z = 1;
    };

    struct DispatchDims
    {
        Dim3 grid;
        Dim3 block;
        unsigned int shared_mem_bytes = 0;
    };

    using DimMap = std::unordered_map<std::string, int64_t>;
    using DispatchFunc = std::function<DispatchDims(const DimMap&)>;

    class DispatchRegistry
    {
    public:
        static DispatchRegistry& instance()
        {
            static DispatchRegistry inst;
            return inst;
        }

        void register_dispatch(const std::string& kernel_name, DispatchFunc fn)
        {
            table_[kernel_name] = std::move(fn);
        }

        [[nodiscard]] DispatchDims get_dims(const std::string& kernel_name, const DimMap& dims) const
        {
            auto it = table_.find(kernel_name);
            if (it == table_.end())
                throw std::runtime_error("DispatchRegistry: no hay regla para '" + kernel_name + "'");
            return it->second(dims);
        }

        [[nodiscard]] bool has(const std::string& kernel_name) const
        {
            return table_.contains(kernel_name);
        }

    private:
        DispatchRegistry() = default;
        std::unordered_map<std::string, DispatchFunc> table_;
    };
} // namespace yadrakova::core
