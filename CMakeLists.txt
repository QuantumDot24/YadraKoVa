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
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
# ---------------------------------------------------------------------------
# Auto-descubrimiento de fuentes e includes (todo lo que NO es autogenerado)
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

# NUEVO: glob de los .yaml de dispatch, con CONFIGURE_DEPENDS -- si
# agregas/quitas un .yaml, CMake tiene que reconfigurar, porque mas
# abajo decidimos EN TIEMPO DE CONFIGURACION (con EXISTS) si ese
# kernel tiene o no un *_dispatch.generated.cpp como output esperado.
# Sin este glob, agregar un .yaml nuevo no dispararia reconfiguracion
# y el build quedaria desactualizado hasta un cmake manual.
file(GLOB_RECURSE KERNEL_YAML_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/kernels/*.yaml")

# ---------------------------------------------------------------------------
# Codegen de kernels -- OUTPUT explicito, NO glob, para los .cpp de
# registro. CMake conoce los nombres de salida en tiempo de
# configuracion (se derivan 1:1 de los .cu de entrada), asi que Ninja
# los trata como dependencias reales sin importar timing/cache del
# script Python. Esto elimina la carrera que causaba "no hay kernel
# registrado" en builds subsecuentes.
#
# Los .cuh (solo datos) SI pueden vivir en include/ y descubrirse por
# glob normal, porque nadie los compila directamente -- solo se
# incluyen. El problema de la carrera solo aplica a unidades de
# compilacion reales (.cpp), no a headers.
#
# NUEVO: ademas de "{kernel}_registration.cpp" (siempre), agregamos
# "{kernel}_dispatch.generated.cpp" SOLO si existe un
# src/kernels/{kernel}.yaml -- migracion incremental, kernel por
# kernel. compile_kernel.py hace exactamente esta misma verificacion
# (find_dispatch_yaml) del lado de Python; aqui la replicamos del
# lado de CMake porque el custom_command necesita declarar sus
# OUTPUT de antemano, en tiempo de configuracion.
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
        message(STATUS "YadraKoVa -- '${kernel_name}' sin .yaml de dispatch, aun no migrado")
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
        COMMENT "Compilando kernels y generando registro + dispatch (.cu/.yaml -> .cubin -> .cuh + .cpp)"
        VERBATIM
)

# ---------------------------------------------------------------------------
# Libreria nucleo
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
)

# ---------------------------------------------------------------------------
# Tests -- un ejecutable por archivo en tests/. /WHOLEARCHIVE es
# necesario porque los .cpp de registro (tanto KernelRegistry como,
# ahora, DispatchRegistry) no exportan simbolos que nadie llama
# directamente -- sin esto, el linker los descarta de
# yadrakova_core.lib por "no usados", y el registro estatico nunca se
# ejecuta (efecto: "no hay kernel registrado para X" o "no hay regla
# de dispatch registrada para X" en runtime, con el build en verde).
# ---------------------------------------------------------------------------
enable_testing()

foreach(test_src ${YADRAKOVA_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE yadrakova_core)
    # NUEVO: cuBLAS para tests de matmul que usen cublas_gemm_bf16
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
message(STATUS "YadraKoVa -- .cpp generados (registro + dispatch, OUTPUT explicito):")
foreach(src ${KERNEL_GENERATED_SOURCES})
    message(STATUS "  ${src}")
endforeach()
message(STATUS "YadraKoVa -- tests detectados:")
foreach(src ${YADRAKOVA_TEST_SOURCES})
    message(STATUS "  ${src}")
endforeach()