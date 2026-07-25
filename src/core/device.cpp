#include "core/device.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace yadrakova::core {

    Device& Device::instance() {
        static Device dev;
        return dev;
    }

    Device::Device() {
        int device_id = 0;
        cudaDeviceProp props{};
        cudaError_t err = cudaGetDeviceProperties(&props, device_id);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                "Device: cudaGetDeviceProperties failed: " +
                std::string(cudaGetErrorString(err)));
        }

        name_ = props.name;
        major_ = props.major;
        minor_ = props.minor;
        sm_count_ = props.multiProcessorCount;
        total_mem_bytes_ = props.totalGlobalMem;
        shared_mem_per_block_bytes_ = props.sharedMemPerBlock;
        max_threads_per_block_ = props.maxThreadsPerBlock;

        arch_ = map_arch(major_, minor_);
    }

    kernels::Arch Device::map_arch(int major, int minor) {
        int cc = major * 10 + minor;
        switch (cc) {
        case 86:  return kernels::Arch::SM_86;
        case 100: return kernels::Arch::SM_100;
        default:
            throw std::runtime_error(
                "Device: this GPU is sm_" + std::to_string(cc) +
                ", no kernels::Arch mapped for it. If you want to "
                "support it: (1) add the case to Device::map_arch, "
                "(2) add the value to the kernels::Arch enum, "
                "(3) add the architecture to config.yaml and recompile the kernels.");
        }
    }

} // namespace yadrakova::core