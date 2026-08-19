cmake_minimum_required(VERSION 4.3)
project(YadraKoVa LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(PahoMqttCpp CONFIG REQUIRED)
find_program(PYTHON3_EXECUTABLE NAMES python3 python REQUIRED)

# ---------------------------------------------------------------------------
# cuDNN -- a diferencia de cuBLAS/cuRAND/cuTENSOR, NO viene incluido en
# CUDAToolkit ni tiene find_package propio de CMake. Se busca a mano.
# Si tu instalacion no esta en una ruta estandar, pasa -DCUDNN_ROOT=<ruta>
# a cmake (ej. "C:/Program Files/NVIDIA/CUDNN/v9.x" en Windows).
#
# Los paquetes de NVIDIA desde cuDNN 9.x+ NO ponen cudnn.h/cudnn.lib
# directo en include/ y lib/ -- meten una carpeta intermedia con la
# version de CUDA (ej. include/13.4/cudnn.h, lib/13.4/x64/cudnn.lib), y
# ese numero cambia segun que build bajaste. Por eso buscamos con GLOB_RECURSE
# en vez de un PATH_SUFFIXES fijo -- no queremos tener que actualizar el
# CMakeLists cada vez que cambia esa carpeta intermedia.
#
# IMPORTANTE: filtramos a proposito cualquier ruta que contenga "arm64" --
# NVIDIA tambien distribuye un build ARM64 de cuDNN (para Windows on ARM)
# bajo la MISMA carpeta de version si bajaste el paquete equivocado, y esas
# libs no linkean contra un build x64 (MSVC Hostx64 con CUDA x64). Si el
# glob no encuentra nada x64 mas abajo, el error te lo va a decir.
# ---------------------------------------------------------------------------
if(CUDNN_ROOT OR DEFINED ENV{CUDNN_ROOT})
    set(_cudnn_search_root "${CUDNN_ROOT}")
    if(NOT _cudnn_search_root)
        set(_cudnn_search_root "$ENV{CUDNN_ROOT}")
    endif()

    file(GLOB_RECURSE _cudnn_header_candidates "${_cudnn_search_root}/cudnn.h")
    file(GLOB_RECURSE _cudnn_lib_candidates "${_cudnn_search_root}/cudnn.lib")
    file(GLOB_RECURSE _cudnn_so_candidates "${_cudnn_search_root}/libcudnn.so")
    file(GLOB_RECURSE _cudnn_dll_candidates "${_cudnn_search_root}/cudnn64_9.dll")

    list(FILTER _cudnn_lib_candidates EXCLUDE REGEX "[Aa][Rr][Mm]64")
    list(FILTER _cudnn_header_candidates EXCLUDE REGEX "[Aa][Rr][Mm]64")
    list(FILTER _cudnn_dll_candidates EXCLUDE REGEX "[Aa][Rr][Mm]64")

    if(_cudnn_dll_candidates)
        list(GET _cudnn_dll_candidates 0 _cudnn_main_dll)
        get_filename_component(CUDNN_BIN_DIR "${_cudnn_main_dll}" DIRECTORY)
        # cuDNN 9.x viene partido en varias DLLs (ops/graph/engines/heuristic/...)
        # -- copiamos TODAS las que esten en ese bin/, no solo cudnn64_9.dll,
        # porque esa es solo la "fachada" y carga las demas en runtime.
        file(GLOB CUDNN_RUNTIME_DLLS "${CUDNN_BIN_DIR}/*.dll")
    endif()

    if(_cudnn_header_candidates)
        list(GET _cudnn_header_candidates 0 _cudnn_header_full_path)
        get_filename_component(CUDNN_INCLUDE_DIR "${_cudnn_header_full_path}" DIRECTORY)
    endif()

    if(_cudnn_lib_candidates)
        list(GET _cudnn_lib_candidates 0 CUDNN_LIBRARY)
    elseif(_cudnn_so_candidates)
        list(GET _cudnn_so_candidates 0 CUDNN_LIBRARY)
    endif()
endif()

# Fallback: por si algun dia lo instalan en una ruta que CMake ya conoce
# de memoria (find_path/find_library normales, sin glob).
find_path(CUDNN_INCLUDE_DIR cudnn.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS}
        PATH_SUFFIXES include
)
find_library(CUDNN_LIBRARY NAMES cudnn
        HINTS ${CUDAToolkit_LIBRARY_DIR}
        PATH_SUFFIXES lib lib/x64 lib64
)

if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(FATAL_ERROR
            "YadraKoVa -- cuDNN no encontrado (cudnn.h / cudnn.lib|libcudnn) "
            "buscando bajo CUDNN_ROOT='${CUDNN_ROOT}'. Revisa que: "
            "(1) sea el paquete Windows x86_64 de cuDNN, no ARM64; "
            "(2) CUDNN_ROOT apunte a la carpeta que tiene bin/include/lib adentro.")
endif()
add_library(CUDNN::cudnn UNKNOWN IMPORTED)
set_target_properties(CUDNN::cudnn PROPERTIES
        IMPORTED_LOCATION "${CUDNN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}"
)
message(STATUS "YadraKoVa -- cuDNN include: ${CUDNN_INCLUDE_DIR}")
message(STATUS "YadraKoVa -- cuDNN lib:     ${CUDNN_LIBRARY}")

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
        CUDA::cublas
        CUDA::curand
        CUDNN::cudnn
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
    if(MSVC)
        target_link_options(${test_name} PRIVATE "/WHOLEARCHIVE:yadrakova_core")
    else()
        target_link_options(${test_name} PRIVATE "-Wl,--whole-archive" yadrakova_core "-Wl,--no-whole-archive")
    endif()

    set_target_properties(${test_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )

    # Windows no tiene rpath -- el .exe busca DLLs en su propio directorio o
    # en el PATH del sistema. cuDNN via zip (a diferencia del CUDA Toolkit,
    # que si se agrega solo al PATH con su instalador) no toca el PATH, asi
    # que sin esto el .exe compila y linkea bien pero explota en runtime con
    # STATUS_DLL_NOT_FOUND (0xC0000135) apenas intenta cargar cudnn64_9.dll.
    if(WIN32 AND CUDNN_RUNTIME_DLLS)
        add_custom_command(TARGET ${test_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CUDNN_RUNTIME_DLLS} $<TARGET_FILE_DIR:${test_name}>
                COMMENT "Copiando DLLs de cuDNN junto a ${test_name}"
        )
    endif()

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