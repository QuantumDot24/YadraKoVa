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
        # ¡ESTO ES LO IMPORTANTE!: CMake detectará automáticamente tus .cpp autogenerados
        "${CMAKE_SOURCE_DIR}/include/kernels/embedded/*.cpp"
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
# Codegen de kernels
# ---------------------------------------------------------------------------
add_custom_target(kernel_codegen
        COMMAND ${PYTHON3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/compile_kernel.py
        --config ${CMAKE_SOURCE_DIR}/config.yaml
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS ${KERNEL_SOURCES} ${CMAKE_SOURCE_DIR}/scripts/compile_kernel.py ${CMAKE_SOURCE_DIR}/config.yaml
        COMMENT "Compilando y embebiendo kernels CUDA"
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

target_link_libraries(yadrakova_core PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
)

add_dependencies(yadrakova_core kernel_codegen)

# !!! AQUÍ ELIMINAMOS POR COMPLETO EL BUCLE FOREACH DE LOS TARGET_COMPILE_OPTIONS !!!
# Ya no inyectamos el .cpp en ningún lado.

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

foreach(test_src ${YADRAKOVA_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE yadrakova_core)

    # OBLIGATORIO PARA WINDOWS: Evita que el linker borre los .cpp de registro
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