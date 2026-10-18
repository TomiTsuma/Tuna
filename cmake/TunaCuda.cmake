# Track B CUDA runtime-API remoting (Phase 3). Opt-in: it needs the CUDA 12
# runtime headers (not a GPU, and not the CUDA compiler).
#
# Provides (when TUNA_BUILD_CUDA_SHIM=ON):
#   tuna_cuda_dispatch      server-side CudaSession (links into tuna_server)
#   tuna_cuda_backend_real  dlopen of NVIDIA's libcudart + libcuda (server)
#   tuna_cuda_client        client shim objects
#   tuna_cudart             the client shim as a shared library:
#                           <build>/shim/libcudart.so.12 (Linux) or
#                           <build>/shim/cudart64_12.dll (Windows)
#   tuna_cublas             the cuBLAS shim (Phase 3d), when the cuBLAS headers
#                           are found: <build>/shim/libcublas.so.12 or
#                           <build>/shim/cublas64_12.dll; it sends its calls
#                           through tuna_cudart's session
#   tuna_mock_cuda          a mock CUDA stack the server can load (tests)
#   tuna_add_cuda_tests()   marshalling test + GPU-free end-to-end test
#
# Before including this file, set CMAKE_POSITION_INDEPENDENT_CODE ON (the
# shim links the tunnel and protobuf statically into a shared library).
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

# cuBLAS headers (cublas_v2.h): in the toolkit's include dir, or from NVIDIA's
# redistributable / pip wheel (nvidia-cublas-cu12). Optional: without them the
# runtime shim builds as before and cuBLAS calls report "not supported".
if(NOT TUNA_CUBLAS_INCLUDE_DIR)
    find_path(TUNA_CUBLAS_INCLUDE_DIR cublas_v2.h PATHS "${TUNA_CUDA_INCLUDE_DIR}" NO_DEFAULT_PATH)
endif()
if(TUNA_CUBLAS_INCLUDE_DIR)
    set(TUNA_CUDA_HAVE_CUBLAS ON)
    message(STATUS "Tuna CUDA shim: cuBLAS remoting enabled (${TUNA_CUBLAS_INCLUDE_DIR})")
else()
    set(TUNA_CUDA_HAVE_CUBLAS OFF)
    message(STATUS "Tuna CUDA shim: cublas_v2.h not found, building without cuBLAS (set TUNA_CUBLAS_INCLUDE_DIR)")
endif()

# Codegen needs python + libclang + the CUDA headers, so by default the build
# uses the committed stubs in cuda/generated/ and no codegen runs. Regenerate
# them (into the build tree) with -DTUNA_REGEN_CUDA=ON; commit the result by
# re-running tools/codegen/gen_cuda.py into cuda/generated/.
option(TUNA_REGEN_CUDA "Regenerate the CUDA stubs with libclang instead of using the committed ones" OFF)
set(_tuna_cuda_gen_names
    cuda_api_ids.h cuda_client_stubs.inc cuda_server_dispatch.inc cuda_unsupported_stubs.inc
    cuda_error_strings.inc cuda_coverage.inc)
set(_tuna_codegen_cublas_args)
if(TUNA_CUDA_HAVE_CUBLAS)
    list(APPEND _tuna_cuda_gen_names cublas_client_stubs.inc cublas_server_dispatch.inc cublas_unsupported_stubs.inc)
    set(_tuna_codegen_cublas_args --cublas "${TUNA_CUBLAS_INCLUDE_DIR}")
endif()
if(TUNA_REGEN_CUDA)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(TUNA_CUDA_GEN_DIR "${CMAKE_BINARY_DIR}/generated/cuda")
    list(TRANSFORM _tuna_cuda_gen_names PREPEND "${TUNA_CUDA_GEN_DIR}/" OUTPUT_VARIABLE TUNA_CUDA_GENERATED)
    add_custom_command(
        OUTPUT ${TUNA_CUDA_GENERATED}
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${TUNA_CUDA_GEN_DIR}"
        COMMAND "${Python3_EXECUTABLE}" "${TUNA_CODEGEN_DIR}/gen_cuda.py"
                "${TUNA_CUDA_INCLUDE_DIR}" "${TUNA_CODEGEN_DIR}/annotations.yaml" "${TUNA_CUDA_GEN_DIR}"
                ${_tuna_codegen_cublas_args}
        DEPENDS "${TUNA_CODEGEN_DIR}/gen_cuda.py" "${TUNA_CODEGEN_DIR}/annotations.yaml"
                "${TUNA_CODEGEN_DIR}/cudart12_exports.txt" "${TUNA_CODEGEN_DIR}/annotations_cublas.yaml"
                "${TUNA_CODEGEN_DIR}/cublas12_exports.txt"
        COMMENT "Generating CUDA runtime remoting stubs"
        VERBATIM)
    add_custom_target(tuna_cuda_codegen DEPENDS ${TUNA_CUDA_GENERATED})
else()
    set(TUNA_CUDA_GEN_DIR "${TUNA_CUDA_COMMITTED_GEN}")
    add_custom_target(tuna_cuda_codegen)  # nothing to do; stubs are committed
endif()

set(_tuna_cuda_includes ${TUNA_CUDA_DIR}/include ${TUNA_CUDA_GEN_DIR} ${TUNA_CUDA_INCLUDE_DIR})
if(TUNA_CUDA_HAVE_CUBLAS)
    list(APPEND _tuna_cuda_includes ${TUNA_CUBLAS_INCLUDE_DIR})
endif()
set(_tuna_cuda_warnings $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>)

# Server-side dispatcher.
add_library(tuna_cuda_dispatch STATIC ${TUNA_CUDA_DIR}/src/dispatch.cpp)
if(TUNA_CUDA_HAVE_CUBLAS)
    target_sources(tuna_cuda_dispatch PRIVATE ${TUNA_CUDA_DIR}/src/dispatch_cublas.cpp)
    target_compile_definitions(tuna_cuda_dispatch PUBLIC TUNA_CUDA_HAVE_CUBLAS)
endif()
add_dependencies(tuna_cuda_dispatch tuna_cuda_codegen)
target_include_directories(tuna_cuda_dispatch PUBLIC ${_tuna_cuda_includes})
target_link_libraries(tuna_cuda_dispatch PUBLIC tuna_tunnel tuna_proto)
target_compile_options(tuna_cuda_dispatch PRIVATE ${_tuna_cuda_warnings})

# Production backend: resolves symbols from the real libcudart/libcuda at runtime.
add_library(tuna_cuda_backend_real STATIC ${TUNA_CUDA_DIR}/src/backend_real.cpp)
add_dependencies(tuna_cuda_backend_real tuna_cuda_codegen)
target_link_libraries(tuna_cuda_backend_real PUBLIC tuna_cuda_dispatch)
target_compile_options(tuna_cuda_backend_real PRIVATE ${_tuna_cuda_warnings})
if(UNIX)
    target_link_libraries(tuna_cuda_backend_real PUBLIC ${CMAKE_DL_LIBS})
endif()

# Client shim objects (the replacement libcudart). An object library, so the
# same code goes into the shared library and into the in-process tests.
add_library(tuna_cuda_client OBJECT
    ${TUNA_CUDA_DIR}/src/client.cpp
    ${TUNA_CUDA_DIR}/src/client_runtime.cpp
    ${TUNA_CUDA_DIR}/src/client_launch.cpp
    ${TUNA_CUDA_DIR}/src/client_unsupported.cpp)
add_dependencies(tuna_cuda_client tuna_cuda_codegen)
target_include_directories(tuna_cuda_client PUBLIC ${_tuna_cuda_includes})
target_link_libraries(tuna_cuda_client PUBLIC tuna_tunnel tuna_proto)
target_compile_options(tuna_cuda_client PRIVATE ${_tuna_cuda_warnings})

# The shim as the runtime library itself. Programs built against NVIDIA's
# runtime (`nvcc -cudart shared`, PyTorch) load it in place of the real one
# via LD_LIBRARY_PATH / the DLL search path; `tuna run` sets that up.
add_library(tuna_cudart SHARED $<TARGET_OBJECTS:tuna_cuda_client>)
target_link_libraries(tuna_cudart PRIVATE tuna_tunnel tuna_proto)
set_target_properties(tuna_cudart PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/shim"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/shim")
if(WIN32)
    set_target_properties(tuna_cudart PROPERTIES OUTPUT_NAME cudart64_12 PREFIX "")
else()
    # libcudart.so.12 with NVIDIA's symbol version node; only the CUDA API is
    # exported (see cuda/libcudart.map).
    set_target_properties(tuna_cudart PROPERTIES OUTPUT_NAME cudart VERSION 12 SOVERSION 12)
    target_link_options(tuna_cudart PRIVATE
        "LINKER:--version-script=${TUNA_CUDA_DIR}/libcudart.map" "LINKER:--no-undefined")
    set_property(TARGET tuna_cudart APPEND PROPERTY LINK_DEPENDS "${TUNA_CUDA_DIR}/libcudart.map")
endif()

# The cuBLAS shim: libcublas.so.12 / cublas64_12.dll next to the runtime shim.
# It carries no transport of its own; every call goes through tuna_cudart's
# session (tuna_cudart_call), in order with the app's runtime calls.
if(TUNA_CUDA_HAVE_CUBLAS)
    add_library(tuna_cublas_client OBJECT
        ${TUNA_CUDA_DIR}/src/cublas_client.cpp
        ${TUNA_CUDA_DIR}/src/cublas_unsupported.cpp)
    add_dependencies(tuna_cublas_client tuna_cuda_codegen)
    target_include_directories(tuna_cublas_client PUBLIC ${_tuna_cuda_includes})
    target_compile_options(tuna_cublas_client PRIVATE ${_tuna_cuda_warnings})

    add_library(tuna_cublas SHARED $<TARGET_OBJECTS:tuna_cublas_client>)
    target_link_libraries(tuna_cublas PRIVATE tuna_cudart)
    set_target_properties(tuna_cublas PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/shim"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/shim")
    if(WIN32)
        set_target_properties(tuna_cublas PROPERTIES OUTPUT_NAME cublas64_12 PREFIX "")
    else()
        set_target_properties(tuna_cublas PROPERTIES OUTPUT_NAME cublas VERSION 12 SOVERSION 12)
        target_link_options(tuna_cublas PRIVATE
            "LINKER:--version-script=${TUNA_CUDA_DIR}/libcublas.map" "LINKER:--no-undefined")
        set_property(TARGET tuna_cublas APPEND PROPERTY LINK_DEPENDS "${TUNA_CUDA_DIR}/libcublas.map")
    endif()
endif()

# A mock CUDA stack for tuna_server (runtime_library in its config), so the
# end-to-end test runs on machines with no GPU.
add_library(tuna_mock_cuda SHARED ${TUNA_CUDA_DIR}/tests/mock_cuda.cpp)
target_include_directories(tuna_mock_cuda PRIVATE ${TUNA_CUDA_INCLUDE_DIR})
if(TUNA_CUDA_HAVE_CUBLAS)
    target_include_directories(tuna_mock_cuda PRIVATE ${TUNA_CUBLAS_INCLUDE_DIR})
    target_compile_definitions(tuna_mock_cuda PRIVATE TUNA_CUDA_HAVE_CUBLAS)
endif()
set_target_properties(tuna_mock_cuda PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_options(tuna_mock_cuda PRIVATE ${_tuna_cuda_warnings})

# Real-GPU probe: drives the GPU through the dispatch in-process. Builds
# without a GPU (libcudart is dlopen'd); running it needs one, so it is not a
# ctest.
add_executable(tuna_cuda_real_probe ${TUNA_CUDA_DIR}/tests/real_gpu_probe.cpp
               $<TARGET_OBJECTS:tuna_cuda_client>)
target_link_libraries(tuna_cuda_real_probe PRIVATE tuna_cuda_backend_real tuna_cuda_client)

function(tuna_add_cuda_tests)
    if(TARGET tuna_cuda_tests)
        return()
    endif()
    add_executable(tuna_cuda_tests ${TUNA_CUDA_DIR}/tests/test_marshal.cpp ${TUNA_CUDA_DIR}/tests/mock_cuda.cpp
                   $<TARGET_OBJECTS:tuna_cuda_client>)
    target_link_libraries(tuna_cuda_tests PRIVATE Catch2::Catch2WithMain tuna_cuda_dispatch tuna_cuda_client)
    if(TUNA_CUDA_HAVE_CUBLAS)
        # The cuBLAS shim's objects in the same process: tuna_cudart_call is
        # then the in-process runtime client, so the whole chain is exercised.
        target_sources(tuna_cuda_tests PRIVATE ${TUNA_CUDA_DIR}/tests/test_cublas.cpp
                       $<TARGET_OBJECTS:tuna_cublas_client>)
    endif()
    catch_discover_tests(tuna_cuda_tests TEST_PREFIX "cuda: ")

    # A program linked against the shim, exactly as an nvcc-built app links
    # libcudart: it registers a fatbin and launches kernels over the network.
    add_executable(tuna_cuda_e2e_client ${TUNA_CUDA_DIR}/tests/e2e_client.cpp)
    target_include_directories(tuna_cuda_e2e_client PRIVATE ${TUNA_CUDA_INCLUDE_DIR})
    target_link_libraries(tuna_cuda_e2e_client PRIVATE tuna_cudart)

    # The same for cuBLAS: a program linked against libcublas + libcudart.
    set(_tuna_cublas_e2e_args)
    if(TUNA_CUDA_HAVE_CUBLAS)
        add_executable(tuna_cublas_e2e_client ${TUNA_CUDA_DIR}/tests/e2e_cublas_client.cpp)
        target_include_directories(tuna_cublas_e2e_client PRIVATE ${TUNA_CUDA_INCLUDE_DIR} ${TUNA_CUBLAS_INCLUDE_DIR})
        target_link_libraries(tuna_cublas_e2e_client PRIVATE tuna_cublas tuna_cudart)
        set(_tuna_cublas_e2e_args --cublas-client "$<TARGET_FILE:tuna_cublas_e2e_client>")
    endif()

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND AND TARGET tuna_server)
        add_test(NAME tuna_cuda_e2e
            COMMAND "${Python3_EXECUTABLE}" "${TUNA_CUDA_DIR}/tests/integration/test_cuda_e2e.py"
                "$<TARGET_FILE:tuna_server>" "$<TARGET_FILE:tuna_cuda_e2e_client>"
                "$<TARGET_FILE:tuna_mock_cuda>" "$<TARGET_FILE_DIR:tuna_cudart>" ${_tuna_cublas_e2e_args})
        set_tests_properties(tuna_cuda_e2e PROPERTIES TIMEOUT 120)
    endif()
endfunction()
