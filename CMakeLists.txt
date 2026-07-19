cmake_minimum_required(VERSION 4.3)
project(YadraKoVa LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_program(PYTHON3_EXECUTABLE NAMES python3 python REQUIRED)

# ---------------------------------------------------------------------------
# Estándares de lenguaje
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Auto-descubrimiento de fuentes e includes
# ---------------------------------------------------------------------------
file(GLOB_RECURSE YADRAKOVA_CORE_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/src/core/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/core/*.cu"
        "${CMAKE_SOURCE_DIR}/src/telemetry/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/train/*.cpp"
)

file(GLOB_RECURSE YADRAKOVA_HEADERS CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/include/*.h"
        "${CMAKE_SOURCE_DIR}/include/*.cuh"
)

file(GLOB_RECURSE YADRAKOVA_TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/tests/*.cpp"
)

file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/kernels/*.cu")

# ---------------------------------------------------------------------------
# Codegen de kernels: compila .cu -> .cubin -> .cuh embebido ANTES de
# que cualquier target intente incluir esos headers. Debe declararse
# antes de la libreria que depende de el.
# ---------------------------------------------------------------------------
add_custom_target(kernel_codegen
        COMMAND ${PYTHON3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/compile_kernel.py
        --config ${CMAKE_SOURCE_DIR}/config.yaml
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS ${KERNEL_SOURCES} ${CMAKE_SOURCE_DIR}/scripts/compile_kernel.py ${CMAKE_SOURCE_DIR}/config.yaml
        COMMENT "Compilando y embebiendo kernels CUDA (.cu -> .cubin -> .cuh)"
)

# ---------------------------------------------------------------------------
# Libreria nucleo
# ---------------------------------------------------------------------------
add_library(yadrakova_core STATIC ${YADRAKOVA_CORE_SOURCES})

target_include_directories(yadrakova_core PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(yadrakova_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

# cudart: runtime API (cudaMalloc, cudaMemcpy, streams, events, graphs)
# cuda_driver: driver API (cuModuleLoadData, cuLaunchKernel) -- necesario
# para el registry.h que carga cubins embebidos.
target_link_libraries(yadrakova_core PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
)

# La libreria necesita que los .cuh embebidos existan ANTES de compilarse.
add_dependencies(yadrakova_core kernel_codegen)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

foreach(test_src ${YADRAKOVA_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE yadrakova_core)
    set_target_properties(${test_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# ---------------------------------------------------------------------------
# Diagnostico
# ---------------------------------------------------------------------------
message(STATUS "YadraKoVa -- fuentes del core detectadas:")
foreach(src ${YADRAKOVA_CORE_SOURCES})
    message(STATUS "  ${src}")
endforeach()
message(STATUS "YadraKoVa -- kernels .cu detectados:")
foreach(src ${KERNEL_SOURCES})
    message(STATUS "  ${src}")
endforeach()
message(STATUS "YadraKoVa -- tests detectados:")
foreach(src ${YADRAKOVA_TEST_SOURCES})
    message(STATUS "  ${src}")
endforeach()