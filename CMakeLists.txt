cmake_minimum_required(VERSION 4.3)
project(YadraKoVa LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(PahoMqttCpp CONFIG REQUIRED)
find_program(PYTHON3_EXECUTABLE NAMES python3 python REQUIRED)

# ---------------------------------------------------------------------------
# Language standards
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
# ---------------------------------------------------------------------------
# Auto-discovery of sources and includes (everything that is NOT auto-generated)
# ---------------------------------------------------------------------------
file(GLOB_RECURSE YADRAKOVA_CORE_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/src/core/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/core/*.cu"
        "${CMAKE_SOURCE_DIR}/src/telemetry/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/train/*.cpp"
)

file(GLOB_RECURSE YADRAKOVA_HEADERS CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/include/*.hpp"
        "${CMAKE_SOURCE_DIR}/include/*.cuh"
)

file(GLOB_RECURSE YADRAKOVA_TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/tests/*.cpp"
)

file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/kernels/*.cu")

# NEW: glob of dispatch .yaml files, with CONFIGURE_DEPENDS -- if
# you add/remove a .yaml, CMake must reconfigure, because further
# below we decide AT CONFIGURATION TIME (with EXISTS) whether that
# kernel has a *_dispatch.generated.cpp as an expected output.
# Without this glob, adding a new .yaml would not trigger a reconfiguration
# and the build would remain stale until a manual cmake.
file(GLOB_RECURSE KERNEL_YAML_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/kernels/*.yaml")

# ---------------------------------------------------------------------------
# Kernel codegen -- explicit OUTPUT, NOT glob, for the registration
# .cpp files. CMake knows the output names at configuration time
# (they are derived 1:1 from the input .cu files), so Ninja
# treats them as real dependencies regardless of timing/cache of the
# Python script. This eliminates the race that caused "no kernel
# registered" in subsequent builds.
#
# The .cuh files (data only) CAN live in include/ and be discovered by
# normal glob, because nobody compiles them directly -- they are just
# included. The race problem only applies to real compilation
# units (.cpp), not to headers.
#
# NEW: in addition to "{kernel}_registration.cpp" (always), we add
# "{kernel}_dispatch.generated.cpp" ONLY if a
# src/kernels/{kernel}.yaml exists -- incremental migration, kernel by
# kernel. compile_kernel.py does exactly this same check
# (find_dispatch_yaml) on the Python side; here we replicate it on
# the CMake side because the custom_command needs to declare its
# OUTPUT in advance, at configuration time.
# ---------------------------------------------------------------------------
set(KERNEL_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated/kernels")
set(KERNEL_GENERATED_SOURCES "")

foreach(kernel_src ${KERNEL_SOURCES})
    get_filename_component(kernel_name ${kernel_src} NAME_WE)
    get_filename_component(kernel_dir ${kernel_src} DIRECTORY)

    list(APPEND KERNEL_GENERATED_SOURCES "${KERNEL_GENERATED_DIR}/${kernel_name}_registration.cpp")

    if(EXISTS "${kernel_dir}/${kernel_name}.yaml")
        list(APPEND KERNEL_GENERATED_SOURCES "${KERNEL_GENERATED_DIR}/${kernel_name}_dispatch.generated.cpp")
    else()
        message(STATUS "YadraKoVa -- '${kernel_name}' without dispatch .yaml, not yet migrated")
    endif()
endforeach()

file(GLOB SHARED_KERNEL_HEADERS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/include/kernels/*.cuh")

add_custom_command(
        OUTPUT ${KERNEL_GENERATED_SOURCES}
        COMMAND ${PYTHON3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/compile_kernel.py
        --config ${CMAKE_SOURCE_DIR}/config.yaml
        --output-dir ${KERNEL_GENERATED_DIR}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS ${KERNEL_SOURCES} ${KERNEL_YAML_FILES} ${SHARED_KERNEL_HEADERS}
        ${CMAKE_SOURCE_DIR}/scripts/compile_kernel.py ${CMAKE_SOURCE_DIR}/config.yaml
        COMMENT "Compiling kernels and generating registration + dispatch (.cu/.yaml -> .cubin -> .cuh + .cpp)"
        VERBATIM
)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(yadrakova_core STATIC
        ${YADRAKOVA_CORE_SOURCES}
        ${KERNEL_GENERATED_SOURCES}
)

target_include_directories(yadrakova_core PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(yadrakova_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(yadrakova_core PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        PahoMqttCpp::paho-mqttpp3
)

# ---------------------------------------------------------------------------
# Tests -- one executable per file in tests/. /WHOLEARCHIVE is
# necessary because the registration .cpp files (both KernelRegistry and,
# now, DispatchRegistry) don't export symbols that anyone calls
# directly -- without this, the linker discards them from
# yadrakova_core.lib as "unused", and the static registration never
# runs (effect: "no kernel registered for X" or "no dispatch rule
# registered for X" at runtime, with a clean build).
# ---------------------------------------------------------------------------
enable_testing()

foreach(test_src ${YADRAKOVA_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE yadrakova_core)
    target_link_libraries(${test_name} PRIVATE CUDA::cublas)
    if(MSVC)
        target_link_options(${test_name} PRIVATE "/WHOLEARCHIVE:yadrakova_core")
    else()
        target_link_options(${test_name} PRIVATE "-Wl,--whole-archive" yadrakova_core "-Wl,--no-whole-archive")
    endif()

    set_target_properties(${test_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
message(STATUS "YadraKoVa -- core sources detected:")
foreach(src ${YADRAKOVA_CORE_SOURCES})
    message(STATUS "  ${src}")
endforeach()
message(STATUS "YadraKoVa -- .cu kernels detected:")
foreach(src ${KERNEL_SOURCES})
    message(STATUS "  ${src}")
endforeach()
message(STATUS "YadraKoVa -- generated .cpp (registration + dispatch, explicit OUTPUT):")
foreach(src ${KERNEL_GENERATED_SOURCES})
    message(STATUS "  ${src}")
endforeach()
message(STATUS "YadraKoVa -- tests detected:")
foreach(src ${YADRAKOVA_TEST_SOURCES})
    message(STATUS "  ${src}")
endforeach()