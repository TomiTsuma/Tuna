cmake_minimum_required(VERSION 3.21)
project(TunaServer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
include(FetchContent)
include(GNUInstallDirs)

# One `tuna_server` binary serves both tracks (see src/main.cpp):
#   Track A - explicit-workload gRPC/mTLS API (src/workload/). Needs gRPC +
#             protobuf dev packages; the production server is Linux.
#   Track B - driver-API remoting server core (src/core/), `--foreground`.
if(UNIX)
    set(_tuna_workload_default ON)
else()
    set(_tuna_workload_default OFF)
endif()
option(TUNA_SERVER_WORKLOAD_API "Build the Track A gRPC/mTLS workload API" ${_tuna_workload_default})
option(TUNA_ENABLE_CUDA "Build the production CUDA workload backend" OFF)
if(TUNA_ENABLE_CUDA AND NOT TUNA_SERVER_WORKLOAD_API)
    message(FATAL_ERROR "TUNA_ENABLE_CUDA requires TUNA_SERVER_WORKLOAD_API=ON.")
endif()

if(TUNA_SERVER_WORKLOAD_API)
    set(TUNA_SYSTEM_PROTOBUF ON)  # gRPC is in the build; see cmake/TunaProtobuf.cmake
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/TunaProtobuf.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/TunaTunnel.cmake)

# --- Track B core libraries (config, logging) ---
FetchContent_Declare(
    nlohmann_json
    # Release tarball (~110 KB) instead of a full clone of the repo's history.
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.14.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(spdlog)

add_library(tuna_server_core STATIC
    src/core/config_manager.cpp
    src/core/logger.cpp
)
target_include_directories(tuna_server_core PUBLIC src)
target_link_libraries(tuna_server_core PUBLIC nlohmann_json::nlohmann_json spdlog::spdlog)

# --- Track A workload API (gRPC/mTLS) ---
if(TUNA_SERVER_WORKLOAD_API)
    find_package(gRPC CONFIG REQUIRED)

    if(TUNA_ENABLE_CUDA)
        enable_language(CUDA)
        find_package(CUDAToolkit 12.8 REQUIRED)
        set(TUNA_GPU_BACKEND src/workload/gpu_matrix.cu)
    else()
        set(TUNA_GPU_BACKEND src/workload/gpu_matrix_unavailable.cpp)
    endif()

    set(TUNA_WORKLOAD_PROTO_FILE "${TUNA_PROTO_ROOT}/tuna/v1/workload.proto")
    set(TUNA_WORKLOAD_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/workload")
    set(TUNA_WORKLOAD_GENERATED
        "${TUNA_WORKLOAD_GEN_DIR}/tuna/v1/workload.pb.cc"
        "${TUNA_WORKLOAD_GEN_DIR}/tuna/v1/workload.pb.h"
        "${TUNA_WORKLOAD_GEN_DIR}/tuna/v1/workload.grpc.pb.cc"
        "${TUNA_WORKLOAD_GEN_DIR}/tuna/v1/workload.grpc.pb.h"
    )
    add_custom_command(
        OUTPUT ${TUNA_WORKLOAD_GENERATED}
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${TUNA_WORKLOAD_GEN_DIR}/tuna/v1"
        COMMAND $<TARGET_FILE:protobuf::protoc>
            "--proto_path=${TUNA_PROTO_ROOT}"
            "--cpp_out=${TUNA_WORKLOAD_GEN_DIR}"
            "--grpc_out=${TUNA_WORKLOAD_GEN_DIR}"
            "--plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
            "${TUNA_WORKLOAD_PROTO_FILE}"
        DEPENDS "${TUNA_WORKLOAD_PROTO_FILE}" protobuf::protoc gRPC::grpc_cpp_plugin
        VERBATIM
    )

    # Service, GPU backend and the server entry point, shared by tuna_server and
    # the workload tests.
    add_library(tuna_workload STATIC
        src/workload/workload_server.cpp
        src/workload/workload_service.cpp
        ${TUNA_GPU_BACKEND}
        ${TUNA_WORKLOAD_GENERATED}
    )
    target_include_directories(tuna_workload PUBLIC src/workload "${TUNA_WORKLOAD_GEN_DIR}")
    target_link_libraries(tuna_workload PUBLIC gRPC::grpc++ protobuf::libprotobuf)
    target_compile_options(tuna_workload PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wall>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wextra>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wpedantic>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wconversion>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wsign-conversion>
    )
    if(TUNA_ENABLE_CUDA)
        set_target_properties(tuna_workload PROPERTIES
            CUDA_ARCHITECTURES 86
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        target_link_libraries(tuna_workload PUBLIC CUDA::cudart)
        target_compile_definitions(tuna_workload PUBLIC TUNA_ENABLE_CUDA=1)
    endif()
endif()

add_executable(tuna_server
    src/main.cpp
    src/remoting/remoting_server.cpp
    src/remoting/worker_process.cpp
    src/remoting/worker_session.cpp
    src/remoting/dispatch.cpp
)
target_link_libraries(tuna_server PRIVATE tuna_server_core tuna_tunnel)
target_compile_options(tuna_server PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>)
if(TUNA_SERVER_WORKLOAD_API)
    target_link_libraries(tuna_server PRIVATE tuna_workload)
    target_compile_definitions(tuna_server PRIVATE TUNA_SERVER_WORKLOAD_API=1)
endif()

install(TARGETS tuna_server RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES README.md DESTINATION "${CMAKE_INSTALL_DOCDIR}")
if(TUNA_SERVER_WORKLOAD_API)
    install(FILES deploy/tuna-server.service DESTINATION "lib/systemd/system")
endif()

# --- Tests ---
if(BUILD_TESTING)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.7.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(Catch)

    # Track B (Catch2)
    add_executable(tuna_server_tests
        tests/test_frame_proto.cpp
        tests/test_config.cpp
        tests/test_logger.cpp
    )
    target_link_libraries(tuna_server_tests PRIVATE Catch2::Catch2WithMain tuna_proto tuna_server_core)
    catch_discover_tests(tuna_server_tests)
    tuna_add_tunnel_tests()

    # Real binaries end to end: --foreground listener + tuna_ping sessions.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME tuna_foreground_session
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/test_foreground_session.py"
                "$<TARGET_FILE:tuna_server>" "$<TARGET_FILE:tuna_ping>")
        set_tests_properties(tuna_foreground_session PROPERTIES TIMEOUT 90)
    endif()

    # Track A (plain executables, as authored in PR #1)
    if(TUNA_SERVER_WORKLOAD_API)
        add_executable(tuna_workload_tests tests/workload_service_test.cpp)
        target_link_libraries(tuna_workload_tests PRIVATE tuna_workload)
        add_test(NAME tuna_server_workload COMMAND tuna_workload_tests)

        if(TUNA_ENABLE_CUDA)
            add_executable(tuna_gpu_matrix_tests
                tests/gpu_matrix_test.cpp
                src/workload/gpu_matrix.cu
            )
            set_target_properties(tuna_gpu_matrix_tests PROPERTIES
                CUDA_ARCHITECTURES 86
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED ON
            )
            target_include_directories(tuna_gpu_matrix_tests PRIVATE src/workload)
            target_link_libraries(tuna_gpu_matrix_tests PRIVATE CUDA::cudart)
            add_test(NAME tuna_cuda_matrix_multiply COMMAND tuna_gpu_matrix_tests)
        endif()

        # Client <-> server mTLS round trip; needs the guest-agent superbuild
        # (TUNA_BUILD_LINUX_SERVER=ON) so tuna_sample_app exists.
        if(UNIX AND TARGET tuna_sample_app)
            find_package(Python3 COMPONENTS Interpreter QUIET)
            find_program(OPENSSL_EXECUTABLE openssl)
            if(Python3_Interpreter_FOUND AND OPENSSL_EXECUTABLE)
                set(TUNA_MTLS_TEST_COMMAND
                    "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../tuna-guest-agent/tests/integration/test_mtls_round_trip.py"
                    "$<TARGET_FILE:tuna_server>"
                    "$<TARGET_FILE:tuna_sample_app>"
                    "${OPENSSL_EXECUTABLE}"
                )
                if(TUNA_ENABLE_CUDA)
                    list(APPEND TUNA_MTLS_TEST_COMMAND --gpu-enabled)
                endif()
                add_test(NAME tuna_mtls_integration COMMAND ${TUNA_MTLS_TEST_COMMAND})
                if(TUNA_ENABLE_CUDA)
                    set_tests_properties(tuna_mtls_integration PROPERTIES TIMEOUT 120)
                else()
                    set_tests_properties(tuna_mtls_integration PROPERTIES TIMEOUT 45)
                endif()
            endif()
        endif()
    endif()
endif()
