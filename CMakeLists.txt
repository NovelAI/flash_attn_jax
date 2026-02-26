cmake_minimum_required(VERSION 3.18)

# Get CUDA architectures from environment or use default
if(DEFINED ENV{FLASH_ATTN_CUDA_ARCHS})
    set(FLASH_ATTN_CUDA_ARCHS $ENV{FLASH_ATTN_CUDA_ARCHS})
else()
    set(FLASH_ATTN_CUDA_ARCHS "90a")
endif()

message(STATUS "Using FLASH_ATTN_CUDA_ARCHS: ${FLASH_ATTN_CUDA_ARCHS}")

# set(CMAKE_CUDA_ARCHITECTURES "80;90;100;120")
set(CMAKE_CUDA_ARCHITECTURES ${FLASH_ATTN_CUDA_ARCHS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(flash_attn LANGUAGES CXX CUDA)

if(DEFINED ENV{MAX_JOBS} AND NOT "$ENV{MAX_JOBS}" STREQUAL "")
    set(MAX_JOBS $ENV{MAX_JOBS})
else()
    set(MAX_JOBS 6)
endif()
message(STATUS "Using MAX_JOBS: ${MAX_JOBS}")

set(CMAKE_JOB_POOLS cuda=${MAX_JOBS} cxx=${MAX_JOBS})
set(CMAKE_INSTALL_RPATH "$ORIGIN/nvidia/cuda_runtime/lib")
# Make sure RPATH is used instead of RUNPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

if(PYTHON_VENV AND EXISTS ${PYTHON_VENV}/bin/python)
    # Convert to absolute path in case a relative path like .venv was given
    get_filename_component(PYTHON_VENV "${PYTHON_VENV}" ABSOLUTE)
    execute_process(
        COMMAND ${PYTHON_VENV}/bin/python -c "import sysconfig; print(sysconfig.get_path('purelib'))"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Python site-packages: ${PYTHON_SITE_PACKAGES}")
    set(Python_EXECUTABLE ${PYTHON_VENV}/bin/python)
    # Tell FindPython to prefer the location we specified over highest version
    set(Python_FIND_STRATEGY LOCATION)
    set(CMAKE_PREFIX_PATH
        ${PYTHON_SITE_PACKAGES}
        ${PYTHON_VENV})
endif()

# == Find dependencies ==
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

message(STATUS "Python executable: ${Python_EXECUTABLE}")

execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(nanobind CONFIG REQUIRED)

# == Setup CUDA ==

message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

# Set up ccache
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()

# CUDA handling
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(CUDAToolkit REQUIRED)

if (${NO_OPTIONS_FILE})
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES "0")
    set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES "0")
endif()

# CUDA flags
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CUDA_FLAGS
    -O3
    --use_fast_math
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    -DFLASHATTENTION_DISABLE_DROPOUT=1
    -DFLASHATTENTION_DISABLE_ALIBI=1
)
set(CXX_FLAGS
    -DFLASHATTENTION_DISABLE_DROPOUT=1
    -DFLASHATTENTION_DISABLE_ALIBI=1
)

# Collect source files
file(GLOB CUDA_SOURCES 
    "csrc/flash_attn/src/flash_fwd_hdim*.cu"
    "csrc/flash_attn/src/flash_bwd_hdim*.cu"
    "csrc/flash_attn/src/flash_fwd_split_hdim*.cu"
)

file(GLOB CC_SOURCES 
    "csrc/flash_attn/*.cpp"
)

# Create CUDA extension
nanobind_add_module(flash_api
    ${CC_SOURCES}
    ${CUDA_SOURCES}
)

set_property(TARGET flash_api PROPERTY JOB_POOL_COMPILE cuda)

target_compile_options(flash_api PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
)

target_include_directories(flash_api PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/src
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
)

target_link_libraries(flash_api PRIVATE
    CUDA::cudart
)

if(FLASH_ATTENTION_FORCE_CXX11_ABI)
    target_compile_definitions(flash_api PRIVATE
        _GLIBCXX_USE_CXX11_ABI=1
    )
endif()

# Installation
install(TARGETS flash_api
        DESTINATION ${SKBUILD_PLATLIB_DIR}/flash_attn_jax_lib
)

## === Hopper ===

file(GLOB HOPPER_CUDA_SOURCES_SM80
    "csrc/hopper/gpu/instantiations/*_sm80.cu"
)
file(GLOB HOPPER_CUDA_SOURCES_SM90
    "csrc/hopper/gpu/instantiations/*_sm90.cu"
)
file(GLOB HOPPER_CUDA_SOURCES
    "csrc/hopper/gpu/flash_prepare_scheduler.cu"
    "csrc/hopper/gpu/flash_fwd_combine.cu",
)

file(GLOB HOPPER_CC_SOURCES
    "csrc/hopper/flash_ffi_api.cpp"
    "csrc/hopper/flash_ffi_common.cpp"
    "csrc/hopper/mha_fwd_ffi.cpp"
)

set(FLASHATTENTION_DISABLE_SOFTCAP 1)
set(FLASHATTENTION_DISABLE_PAGEDKV 1)
set(FLASHATTENTION_DISABLE_FP8 1)

# *all* files combine different files, don't need that I think
list(FILTER HOPPER_CUDA_SOURCES_SM80 EXCLUDE REGEX "all")
list(FILTER HOPPER_CUDA_SOURCES_SM90 EXCLUDE REGEX "all")

if(FLASHATTENTION_DISABLE_SOFTCAP)
    list(FILTER HOPPER_CUDA_SOURCES_SM80 EXCLUDE REGEX "softcap")
    list(FILTER HOPPER_CUDA_SOURCES_SM90 EXCLUDE REGEX "softcap")
endif()
if(FLASHATTENTION_DISABLE_PAGEDKV)
    list(FILTER HOPPER_CUDA_SOURCES_SM80 EXCLUDE REGEX "paged")
    list(FILTER HOPPER_CUDA_SOURCES_SM90 EXCLUDE REGEX "paged")
endif()
if(FLASHATTENTION_DISABLE_FP8)
    list(FILTER HOPPER_CUDA_SOURCES_SM80 EXCLUDE REGEX "e4m3")
    list(FILTER HOPPER_CUDA_SOURCES_SM90 EXCLUDE REGEX "e4m3")
endif()

# Extract SM8x architectures (80 and 86)
set(SM8x_CUDA_ARCHS ${FLASH_ATTN_CUDA_ARCHS})
list(FILTER SM8x_CUDA_ARCHS INCLUDE REGEX "^(80|86)$")
set(SM9x_CUDA_ARCHS ${FLASH_ATTN_CUDA_ARCHS})
list(FILTER SM9x_CUDA_ARCHS EXCLUDE REGEX "^(80|86)$")


if(NOT SM8x_CUDA_ARCHS)
  set(FLASHATTENTION_DISABLE_SM8x 1)
endif()



set(HOPPER_CUDA_FLAGS
    -O3
    --use_fast_math
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    -DFLASHATTENTION_DISABLE_DROPOUT=1
    -DFLASHATTENTION_DISABLE_ALIBI=1
    $<$<BOOL:${FLASHATTENTION_DISABLE_SOFTCAP}>:-DFLASHATTENTION_DISABLE_SOFTCAP=1>
    $<$<BOOL:${FLASHATTENTION_DISABLE_PAGEDKV}>:-DFLASHATTENTION_DISABLE_PAGEDKV=1>
)
set(HOPPER_CXX_FLAGS
    -DFLASHATTENTION_DISABLE_DROPOUT=1
    -DFLASHATTENTION_DISABLE_ALIBI=1
    $<$<BOOL:${FLASHATTENTION_DISABLE_FP8}>:-DFLASHATTENTION_DISABLE_FP8=1>
    $<$<BOOL:${FLASHATTENTION_DISABLE_SOFTCAP}>:-DFLASHATTENTION_DISABLE_SOFTCAP=1>
    $<$<BOOL:${FLASHATTENTION_DISABLE_PAGEDKV}>:-DFLASHATTENTION_DISABLE_PAGEDKV=1>
    $<$<BOOL:${FLASHATTENTION_DISABLE_SM8x}>:-DFLASHATTENTION_DISABLE_SM8x=1>
)

# SM80 objects
if(NOT FLASHATTENTION_DISABLE_SM8x)
    add_library(hopper_sm80_objs OBJECT ${HOPPER_CUDA_SOURCES_SM80})
    set_target_properties(hopper_sm80_objs PROPERTIES
        CUDA_ARCHITECTURES ${SM8x_CUDA_ARCHS}
        JOB_POOL_COMPILE cuda
        POSITION_INDEPENDENT_CODE ON)
    target_compile_options(hopper_sm80_objs PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${HOPPER_CUDA_FLAGS}>
    )
    target_include_directories(hopper_sm80_objs PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass/include
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/hopper/gpu
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    )
endif()

# SM90 objects
add_library(hopper_sm90_objs OBJECT ${HOPPER_CUDA_SOURCES_SM90})
set_target_properties(hopper_sm90_objs PROPERTIES
    CUDA_ARCHITECTURES ${SM9x_CUDA_ARCHS}
    JOB_POOL_COMPILE cuda
    POSITION_INDEPENDENT_CODE ON)

# Apply common compile options to both
target_compile_options(hopper_sm90_objs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${HOPPER_CUDA_FLAGS}>
)
target_include_directories(hopper_sm90_objs PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/hopper/gpu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
)

add_library(hopper_common_objs OBJECT ${HOPPER_CUDA_SOURCES})
set_target_properties(hopper_common_objs PROPERTIES
    CUDA_ARCHITECTURES ${FLASH_ATTN_CUDA_ARCHS}
    JOB_POOL_COMPILE cuda
    POSITION_INDEPENDENT_CODE ON)
target_compile_options(hopper_common_objs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${HOPPER_CUDA_FLAGS}>
)
target_include_directories(hopper_common_objs PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/hopper/gpu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
)


nanobind_add_module(flash_hopper_ffi
    ${HOPPER_CC_SOURCES}
    $<$<NOT:$<BOOL:${FLASHATTENTION_DISABLE_SM8x}>>:$<TARGET_OBJECTS:hopper_sm80_objs>>
    $<TARGET_OBJECTS:hopper_sm90_objs>
    $<TARGET_OBJECTS:hopper_common_objs>
)
set_property(TARGET flash_hopper_ffi PROPERTY JOB_POOL_COMPILE cuda)

target_compile_options(flash_hopper_ffi PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${HOPPER_CUDA_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${HOPPER_CXX_FLAGS}>
)
target_include_directories(flash_hopper_ffi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/hopper/gpu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
)
target_link_libraries(flash_hopper_ffi PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
)

install(TARGETS flash_hopper_ffi
        DESTINATION ${SKBUILD_PLATLIB_DIR}/flash_attn_jax_lib
)