# Track B CUDA runtime-API remoting (Phase 3). Opt-in: it needs the CUDA
# runtime headers and a Python with libclang + PyYAML for codegen.
#
# Provides (when TUNA_BUILD_CUDA_SHIM=ON):
#   tuna_cuda_dispatch   server-side dispatcher (links into tuna_server)
#   tuna_cuda_client     client shim object (the replacement libcudart)
#   tuna_add_cuda_tests()  registers the marshalling round-trip test
#
# Only the headers are needed to build and test marshalling (the dispatcher
# calls the real runtime through a resolved pointer, mocked in tests), so this
# builds and its test runs on a box with no GPU.
include_guard(GLOBAL)

option(TUNA_BUILD_CUDA_SHIM "Build the Track B CUDA runtime-API shim (needs CUDA headers)" OFF)
if(NOT TUNA_BUILD_CUDA_SHIM)
    return()
endif()

set(TUNA_CUDA_DIR "${CMAKE_CURRENT_LIST_DIR}/../cuda")
cmake_path(NORMAL_PATH TUNA_CUDA_DIR)
set(TUNA_CODEGEN_DIR "${CMAKE_CURRENT_LIST_DIR}/../tools/codegen")
cmake_path(NORMAL_PATH TUNA_CODEGEN_DIR)

set(TUNA_CUDA_COMMITTED_GEN "${TUNA_CUDA_DIR}/generated")

# CUDA runtime headers. Default to the toolkit; override with -DTUNA_CUDA_INCLUDE_DIR.
if(NOT TUNA_CUDA_INCLUDE_DIR)
    # Tuna targets the CUDA 12 runtime ABI (cudaGetDeviceProperties_v2 etc.), so
    # a 12.x include dir is required; 11.x is rejected below.
    find_path(TUNA_CUDA_INCLUDE_DIR cuda_runtime_api.h
        PATHS
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include"
            "/usr/local/cuda-12/include"
            "/usr/local/cuda/include"
            "$ENV{CUDA_PATH}/include"
        NO_DEFAULT_PATH)
endif()
if(NOT TUNA_CUDA_INCLUDE_DIR)
    message(FATAL_ERROR "cuda_runtime_api.h not found; set -DTUNA_CUDA_INCLUDE_DIR=<cuda-12>/include")
endif()
# Reject CUDA 11: the generated stubs assume the CUDA 12 runtime ABI.
file(READ "${TUNA_CUDA_INCLUDE_DIR}/cuda_runtime_api.h" _tuna_rt_header LIMIT 8000)
string(REGEX MATCH "define CUDART_VERSION +([0-9]+)" _ "${_tuna_rt_header}")
if(CMAKE_MATCH_1 AND CMAKE_MATCH_1 LESS 12000)
    message(FATAL_ERROR "TUNA_CUDA_INCLUDE_DIR is CUDART ${CMAKE_MATCH_1} (<12000); point it at a CUDA 12 include dir")
endif()

# Codegen needs python + libclang + the CUDA headers, so by default the build
# uses the committed stubs in cuda/generated/ and no codegen runs. Regenerate
# them (into the build tree) with -DTUNA_REGEN_CUDA=ON; commit the result by
# re-running tools/codegen/gen_cuda.py into cuda/generated/.
option(TUNA_REGEN_CUDA "Regenerate the CUDA stubs with libclang instead of using the committed ones" OFF)
if(TUNA_REGEN_CUDA)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(TUNA_CUDA_GEN_DIR "${CMAKE_BINARY_DIR}/generated/cuda")
    set(TUNA_CUDA_GENERATED
        "${TUNA_CUDA_GEN_DIR}/cuda_api_ids.h"
        "${TUNA_CUDA_GEN_DIR}/cuda_client_stubs.inc"
        "${TUNA_CUDA_GEN_DIR}/cuda_server_dispatch.inc"
        "${TUNA_CUDA_GEN_DIR}/cuda_coverage.inc")
    add_custom_command(
        OUTPUT ${TUNA_CUDA_GENERATED}
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${TUNA_CUDA_GEN_DIR}"
        COMMAND "${Python3_EXECUTABLE}" "${TUNA_CODEGEN_DIR}/gen_cuda.py"
                "${TUNA_CUDA_INCLUDE_DIR}" "${TUNA_CODEGEN_DIR}/annotations.yaml" "${TUNA_CUDA_GEN_DIR}"
        DEPENDS "${TUNA_CODEGEN_DIR}/gen_cuda.py" "${TUNA_CODEGEN_DIR}/annotations.yaml"
        COMMENT "Generating CUDA runtime remoting stubs"
        VERBATIM)
    add_custom_target(tuna_cuda_codegen DEPENDS ${TUNA_CUDA_GENERATED})
else()
    set(TUNA_CUDA_GEN_DIR "${TUNA_CUDA_COMMITTED_GEN}")
    add_custom_target(tuna_cuda_codegen)  # nothing to do; stubs are committed
endif()

# Server-side dispatcher.
add_library(tuna_cuda_dispatch STATIC ${TUNA_CUDA_DIR}/src/dispatch.cpp)
add_dependencies(tuna_cuda_dispatch tuna_cuda_codegen)
target_include_directories(tuna_cuda_dispatch PUBLIC
    ${TUNA_CUDA_DIR}/include ${TUNA_CUDA_GEN_DIR} ${TUNA_CUDA_INCLUDE_DIR})
target_link_libraries(tuna_cuda_dispatch PUBLIC tuna_tunnel tuna_proto)

# Client shim (the replacement libcudart). Built as an object library so it can
# go into either the test or a standalone shared library.
add_library(tuna_cuda_client OBJECT ${TUNA_CUDA_DIR}/src/client.cpp)
add_dependencies(tuna_cuda_client tuna_cuda_codegen)
target_include_directories(tuna_cuda_client PUBLIC
    ${TUNA_CUDA_DIR}/include ${TUNA_CUDA_GEN_DIR} ${TUNA_CUDA_INCLUDE_DIR})
target_link_libraries(tuna_cuda_client PUBLIC tuna_tunnel tuna_proto)

# Production backend: resolves symbols from the real libcudart at runtime.
add_library(tuna_cuda_backend_real STATIC ${TUNA_CUDA_DIR}/src/backend_real.cpp)
add_dependencies(tuna_cuda_backend_real tuna_cuda_codegen)
target_include_directories(tuna_cuda_backend_real PUBLIC
    ${TUNA_CUDA_DIR}/include ${TUNA_CUDA_GEN_DIR} ${TUNA_CUDA_INCLUDE_DIR})
target_link_libraries(tuna_cuda_backend_real PUBLIC tuna_cuda_dispatch)
if(UNIX)
    target_link_libraries(tuna_cuda_backend_real PUBLIC ${CMAKE_DL_LIBS})
endif()

# Real-GPU probe: drives the GPU through the generated dispatch in-process.
# Builds without a GPU (libcudart is dlopen'd); running it needs one, so it is
# not registered as a ctest.
add_executable(tuna_cuda_real_probe ${TUNA_CUDA_DIR}/tests/real_gpu_probe.cpp
               $<TARGET_OBJECTS:tuna_cuda_client>)
target_link_libraries(tuna_cuda_real_probe PRIVATE tuna_cuda_backend_real tuna_cuda_client)

function(tuna_add_cuda_tests)
    if(TARGET tuna_cuda_tests)
        return()
    endif()
    add_executable(tuna_cuda_tests ${TUNA_CUDA_DIR}/tests/test_marshal.cpp $<TARGET_OBJECTS:tuna_cuda_client>)
    target_link_libraries(tuna_cuda_tests PRIVATE Catch2::Catch2WithMain tuna_cuda_dispatch tuna_cuda_client)
    catch_discover_tests(tuna_cuda_tests TEST_PREFIX "cuda: ")
endfunction()
