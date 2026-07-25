#pragma once

#include <string>
#include "kernels/registry.hpp"

namespace yadrakova::core
{
    class Device
    {
    public:
        static Device& instance();

        [[nodiscard]] kernels::Arch arch() const { return arch_; }
        [[nodiscard]] const std::string& name() const { return name_; }
        [[nodiscard]] int major() const { return major_; }
        [[nodiscard]] int minor() const { return minor_; }
        [[nodiscard]] int sm_count() const { return sm_count_; }
        [[nodiscard]] size_t total_mem_bytes() const { return total_mem_bytes_; }
        [[nodiscard]] size_t shared_mem_per_block_bytes() const { return shared_mem_per_block_bytes_; }
        [[nodiscard]] int max_threads_per_block() const { return max_threads_per_block_; }

    private:
        Device();

        static kernels::Arch map_arch(int major, int minor);

        std::string name_;
        int major_ = 0;
        int minor_ = 0;
        int sm_count_ = 0;
        size_t total_mem_bytes_ = 0;
        size_t shared_mem_per_block_bytes_ = 0;
        int max_threads_per_block_ = 0;
        kernels::Arch arch_;
    };
} // namespace yadrakova::core
