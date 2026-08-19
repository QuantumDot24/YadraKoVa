cmake_minimum_required(VERSION 4.3)
project(YadraKoVa LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# 1. Dependencies
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(PahoMqttCpp CONFIG REQUIRED)
find_program(PYTHON3_EXECUTABLE NAMES python3 python REQUIRED)

# --- cuDNN ---
find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h cudnn_version.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} $ENV{CUDNN_PATH} ${CUDNN_ROOT}
        PATH_SUFFIXES include
)
find_library(CUDNN_LIBRARY NAMES cudnn
        HINTS ${CUDAToolkit_LIBRARY_DIR} $ENV{CUDNN_PATH} ${CUDNN_ROOT}
        PATH_SUFFIXES lib lib/x64 lib64
)
if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cuDNN not found. Please set CUDNN_ROOT or ensure it is installed in the CUDA Toolkit path.")
endif()
add_library(CUDNN::cudnn UNKNOWN IMPORTED)
set_target_properties(CUDNN::cudnn PROPERTIES
        IMPORTED_LOCATION "${CUDNN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}"
)

# --- cuTENSOR (CUDA 12/13 aware - FORZADO) ---
set(CUTENSOR_ROOT_HINTS "C:/Program Files/NVIDIA cuTENSOR/v2.7" $ENV{CUTENSOR_PATH})
find_path(CUTENSOR_INCLUDE_DIR NAMES cutensor.h
        HINTS ${CUTENSOR_ROOT_HINTS}
        PATH_SUFFIXES include
)

# DETECCIÓN AUTOMÁTICA: Usar la versión de CUDA instalada
if(CUDAToolkit_VERSION_MAJOR EQUAL 13)
    set(_cutensor_lib_suffix "lib/13")
    set(_cutensor_bin_suffix "bin/13")
    message(STATUS "cuTENSOR: Detected CUDA 13.x, using lib/13 and bin/13")
elseif(CUDAToolkit_VERSION_MAJOR EQUAL 12)
    set(_cutensor_lib_suffix "lib/12")
    set(_cutensor_bin_suffix "bin/12")
    message(STATUS "cuTENSOR: Detected CUDA 12.x, using lib/12 and bin/12")
else()
    message(FATAL_ERROR "cuTENSOR: Unsupported CUDA version ${CUDAToolkit_VERSION_MAJOR}. Only CUDA 12 and 13 are supported.")
endif()

find_library(CUTENSOR_LIBRARY NAMES cutensor
        HINTS ${CUTENSOR_ROOT_HINTS}
        PATH_SUFFIXES ${_cutensor_lib_suffix}
        NO_DEFAULT_PATH  # CRÍTICO: No buscar en otras ubicaciones
)

if(NOT CUTENSOR_INCLUDE_DIR OR NOT CUTENSOR_LIBRARY)
    message(FATAL_ERROR "cuTENSOR not found in ${CUTENSOR_ROOT_HINTS}/${_cutensor_lib_suffix}. Please verify the installation.")
endif()

add_library(CUTENSOR::cutensor UNKNOWN IMPORTED)
set_target_properties(CUTENSOR::cutensor PROPERTIES
        IMPORTED_LOCATION "${CUTENSOR_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_INCLUDE_DIR}"
)

message(STATUS "cuTENSOR include: ${CUTENSOR_INCLUDE_DIR}")
message(STATUS "cuTENSOR lib:     ${CUTENSOR_LIBRARY}")
# ---------------------------------------------------------------------------
# 2. Language Standards & Flags
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
# 3. Source Discovery
# ---------------------------------------------------------------------------
file(GLOB_RECURSE YADRAKOVA_CORE_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/src/core/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/core/*.cu"
        "${CMAKE_SOURCE_DIR}/src/backends/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/backends/*.cu"
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

# Excluir el test standalone del glob principal (se compila aparte)
list(REMOVE_ITEM YADRAKOVA_TEST_SOURCES
        "${CMAKE_SOURCE_DIR}/tests/test_cutensor_standalone.cpp"
)

file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/kernels/*.cu")
file(GLOB_RECURSE KERNEL_YAML_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/kernels/*.yaml")
file(GLOB SHARED_KERNEL_HEADERS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/include/kernels/*.cuh")

# ---------------------------------------------------------------------------
# 4. Kernel Codegen
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
# 5. Core Library
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
        CUTENSOR::cutensor
        PahoMqttCpp::paho-mqttpp3
)

# ---------------------------------------------------------------------------
# 6. Tests
# ---------------------------------------------------------------------------
enable_testing()

# Pre-fetch DLL paths for Windows runtime (cuTENSOR 2.x aware)
if(WIN32)
    # --- cuDNN DLLs ---
    get_filename_component(_cudnn_lib_dir "${CUDNN_LIBRARY}" DIRECTORY)
    set(_cudnn_bin_dir "${_cudnn_lib_dir}/../bin")
    if(EXISTS "${_cudnn_bin_dir}/cudnn64_9.dll")
        set(CUDNN_RUNTIME_DLLS "${_cudnn_bin_dir}/cudnn64_9.dll")
    else()
        file(GLOB CUDNN_RUNTIME_DLLS "${_cudnn_bin_dir}/cudnn*.dll" "${_cudnn_lib_dir}/cudnn*.dll")
    endif()

    # --- cuTENSOR DLLs (USANDO LA VERSIÓN CORRECTA) ---
    get_filename_component(_cutensor_root_dir "${CUTENSOR_ROOT_HINTS}" DIRECTORY)
    set(_cutensor_bin_dir "${CUTENSOR_ROOT_HINTS}/${_cutensor_bin_suffix}")

    if(EXISTS "${_cutensor_bin_dir}/cutensor.dll")
        set(CUTENSOR_RUNTIME_DLLS
                "${_cutensor_bin_dir}/cutensor.dll"
                "${_cutensor_bin_dir}/cutensorMg.dll"
        )
        message(STATUS "cuTENSOR DLLs found in: ${_cutensor_bin_dir}")
    else()
        message(FATAL_ERROR "cuTENSOR DLLs not found in ${_cutensor_bin_dir}")
    endif()
endif()   # <-- ESTE endif() FALTABA (cierra el if(WIN32))

# Helper function para configurar tests (evita duplicación)
function(configure_yadrakova_test test_name test_src)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE yadrakova_core)

    # WHOLEARCHIVE necesario para que el linker no descarte los registros estáticos
    if(MSVC)
        target_link_options(${test_name} PRIVATE "/WHOLEARCHIVE:yadrakova_core")
        # BUG FIX: cuTENSOR 2.x en Windows necesita stack > 1 MB
        # (el planificador usa recursión profunda para selección de kernels)
        target_link_options(${test_name} PRIVATE "/STACK:16777216")
    else()
        target_link_options(${test_name} PRIVATE "-Wl,--whole-archive" yadrakova_core "-Wl,--no-whole-archive")
    endif()

    set_target_properties(${test_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )

    # Copiar DLLs necesarias junto al ejecutable en Windows
    if(WIN32)
        set(_dlls_to_copy "")
        if(CUDNN_RUNTIME_DLLS)
            list(APPEND _dlls_to_copy ${CUDNN_RUNTIME_DLLS})
        endif()
        if(CUTENSOR_RUNTIME_DLLS)
            list(APPEND _dlls_to_copy ${CUTENSOR_RUNTIME_DLLS})
        endif()

        if(_dlls_to_copy)
            add_custom_command(TARGET ${test_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_dlls_to_copy} $<TARGET_FILE_DIR:${test_name}>
                    COMMENT "Copying CUDA runtime DLLs next to ${test_name}"
            )
        endif()
    endif()

    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Crear todos los tests usando el helper
foreach(test_src ${YADRAKOVA_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    configure_yadrakova_test(${test_name} ${test_src})
endforeach()

# --- Test standalone de cuTENSOR (aislado, sin yadrakova_core) ---
add_executable(test_cutensor_standalone
        ${CMAKE_SOURCE_DIR}/tests/test_cutensor_standalone.cpp
)
target_link_libraries(test_cutensor_standalone PRIVATE
        CUDA::cudart
        CUTENSOR::cutensor
)
target_include_directories(test_cutensor_standalone PRIVATE
        ${CUTENSOR_INCLUDE_DIR}
)
set_target_properties(test_cutensor_standalone PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
)
if(MSVC)
    # Mismo fix de stack para el standalone
    target_link_options(test_cutensor_standalone PRIVATE "/STACK:16777216")
endif()

# Copiar DLLs de cuTENSOR al standalone también
if(WIN32 AND CUTENSOR_RUNTIME_DLLS)
    add_custom_command(TARGET test_cutensor_standalone POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CUTENSOR_RUNTIME_DLLS} $<TARGET_FILE_DIR:test_cutensor_standalone>
            COMMENT "Copying cuTENSOR DLLs next to test_cutensor_standalone"
    )
endif()

add_test(NAME test_cutensor_standalone COMMAND test_cutensor_standalone)

# ---------------------------------------------------------------------------
# 7. Diagnostics
# ---------------------------------------------------------------------------
message(STATUS "YadraKoVa -- Core sources detected: ${YADRAKOVA_CORE_SOURCES}")
message(STATUS "YadraKoVa -- .cu kernels detected: ${KERNEL_SOURCES}")
message(STATUS "YadraKoVa -- Generated .cpp (registration + dispatch): ${KERNEL_GENERATED_SOURCES}")
message(STATUS "YadraKoVa -- Tests detected: ${YADRAKOVA_TEST_SOURCES}")
message(STATUS "YadraKoVa -- cuDNN include: ${CUDNN_INCLUDE_DIR}")
message(STATUS "YadraKoVa -- cuDNN lib:     ${CUDNN_LIBRARY}")
message(STATUS "YadraKoVa -- cuTENSOR include: ${CUTENSOR_INCLUDE_DIR}")
message(STATUS "YadraKoVa -- cuTENSOR lib:     ${CUTENSOR_LIBRARY}")
message(STATUS "YadraKoVa -- cuTENSOR lib suffix (CUDA version): ${_cutensor_lib_suffix}")