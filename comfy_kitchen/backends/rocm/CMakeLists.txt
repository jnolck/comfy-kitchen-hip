cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#motherfucking nanobind is being an asshole
#set(CMAKE_CXX_COMPILER "/home/jnolck/Documents/src/ai/.venv/bin/clang++")

project(comfy_kitchen_rocm LANGUAGES CXX HIP)

# Find ROCm/HIP
find_package(hip REQUIRED)
find_package(rocblas REQUIRED)

# Find Python
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Find nanobind
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NB_RESULT
)
if(NOT NB_RESULT EQUAL 0)
        message(FATAL_ERROR "nanobind not found. Install with: pip install nanobind")
endif()

list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# HIP flags
set(HIP_FLAGS
    -fvisibility=hidden
    -fdiagnostics-color=always
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(HIP_FLAGS ${HIP_FLAGS} -g -O0)
else()
        set(HIP_FLAGS ${HIP_FLAGS} -O3)
endif()

set(CPP_SOURCES
    dlpack_bindings.cpp
)

# Source files
set(HIP_SOURCES
    ops/int8_linear.cu
    ops/cublas_gemm_int8.cu
    ops/apply_rope.cu
)

# Tell CMake which compiler to use for each source
set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE HIP)

# Create the nanobind module
nanobind_add_module(_C_hip NB_STATIC STABLE_ABI LTO ${CPP_SOURCES} ${HIP_SOURCES})
set_target_properties(_C_hip PROPERTIES SUFFIX ".abi3.so")
target_compile_definitions(_C_hip PRIVATE Py_LIMITED_API=0x030C0000 NB_STATIC)

# CXX-only flags
target_compile_options(_C_hip PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
    $<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-color=always>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# HIP-only flags
target_compile_options(_C_hip PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-fvisibility=hidden>
    $<$<COMPILE_LANGUAGE:HIP>:-fdiagnostics-color=always>
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1100>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
)

# Include directories
target_include_directories(_C_hip PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ops
    ${ROCM_PATH}/include
    ${ROCM_PATH}/include/hipblaslt
    ${ROCM_PATH}/include/hipblas
)

# Link libraries
target_link_libraries(_C_hip PRIVATE
    hip::host
    hip::device
    roc::rocblas
    ${CMAKE_DL_LIBS}
)

# RPATH
set_target_properties(_C_hip PROPERTIES
    LINKER_LANGUAGE HIP
    INSTALL_RPATH "$ORIGIN"
    BUILD_RPATH "${ROCM_PATH}/lib"
)

install(TARGETS _C_hip
    LIBRARY DESTINATION .
)
