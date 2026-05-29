cmake_minimum_required(VERSION 3.16)
project(oracle CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "llama.cpp repo root")
set(LLAMA_BUILD_DIR "${REPO_ROOT}/build" CACHE PATH "Pre-built llama.cpp build/ directory")
set(LLAMA_LIB_DIR "${LLAMA_BUILD_DIR}/bin" CACHE PATH "Directory containing libllama.so and libggml*.so")

if(NOT EXISTS "${REPO_ROOT}/include/llama.h")
    message(FATAL_ERROR "llama.h not found under ${REPO_ROOT}/include — set REPO_ROOT correctly")
endif()
if(NOT EXISTS "${LLAMA_LIB_DIR}/libllama.so")
    message(FATAL_ERROR "libllama.so not found under ${LLAMA_LIB_DIR} — build llama.cpp first or set LLAMA_LIB_DIR")
endif()

find_library(LLAMA_LIB     llama     PATHS "${LLAMA_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_LIB      ggml      PATHS "${LLAMA_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_BASE_LIB ggml-base PATHS "${LLAMA_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_CPU_LIB  ggml-cpu  PATHS "${LLAMA_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)

add_executable(oracle oracle.cpp)

target_include_directories(oracle PRIVATE
    "${REPO_ROOT}/include"
    "${REPO_ROOT}/ggml/include"
)

target_link_libraries(oracle PRIVATE
    "${LLAMA_LIB}"
    "${GGML_LIB}"
    "${GGML_BASE_LIB}"
    "${GGML_CPU_LIB}"
)

set_target_properties(oracle PROPERTIES
    BUILD_RPATH       "${LLAMA_LIB_DIR}"
    INSTALL_RPATH     "${LLAMA_LIB_DIR}"
    SKIP_BUILD_RPATH  FALSE
    BUILD_WITH_INSTALL_RPATH FALSE
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(oracle PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
