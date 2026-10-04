cmake_minimum_required(VERSION 3.21)
project(TunaServer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
include(GNUInstallDirs)

find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)

option(TUNA_ENABLE_CUDA "Build the production CUDA workload backend" OFF)
if(TUNA_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit 12.8 REQUIRED)
    set(TUNA_GPU_BACKEND src/gpu_matrix.cu)
else()
    set(TUNA_GPU_BACKEND src/gpu_matrix_unavailable.cpp)
endif()

set(TUNA_PROTO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../protocol")
set(TUNA_PROTO_FILE "${TUNA_PROTO_ROOT}/tuna/v1/workload.proto")
set(TUNA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(TUNA_PROTO_SOURCE "${TUNA_GENERATED_DIR}/tuna/v1/workload.pb.cc")
set(TUNA_PROTO_HEADER "${TUNA_GENERATED_DIR}/tuna/v1/workload.pb.h")
set(TUNA_GRPC_SOURCE "${TUNA_GENERATED_DIR}/tuna/v1/workload.grpc.pb.cc")
set(TUNA_GRPC_HEADER "${TUNA_GENERATED_DIR}/tuna/v1/workload.grpc.pb.h")

add_custom_command(
    OUTPUT
        "${TUNA_PROTO_SOURCE}"
        "${TUNA_PROTO_HEADER}"
        "${TUNA_GRPC_SOURCE}"
        "${TUNA_GRPC_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${TUNA_GENERATED_DIR}/tuna/v1"
    COMMAND $<TARGET_FILE:protobuf::protoc>
        "--proto_path=${TUNA_PROTO_ROOT}"
        "--cpp_out=${TUNA_GENERATED_DIR}"
        "--grpc_out=${TUNA_GENERATED_DIR}"
        "--plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
        "${TUNA_PROTO_FILE}"
    DEPENDS "${TUNA_PROTO_FILE}" protobuf::protoc gRPC::grpc_cpp_plugin
    VERBATIM
)

add_executable(tuna_server
    src/main.cpp
    src/workload_service.cpp
    ${TUNA_GPU_BACKEND}
    "${TUNA_PROTO_SOURCE}"
    "${TUNA_GRPC_SOURCE}"
)
target_include_directories(tuna_server PRIVATE "${TUNA_GENERATED_DIR}")
target_link_libraries(tuna_server PRIVATE gRPC::grpc++ protobuf::libprotobuf)
if(TUNA_ENABLE_CUDA)
    set_target_properties(tuna_server PROPERTIES
        CUDA_ARCHITECTURES 86
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(tuna_server PRIVATE CUDA::cudart)
    target_compile_definitions(tuna_server PRIVATE TUNA_ENABLE_CUDA=1)
endif()
target_compile_options(tuna_server PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wall>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wextra>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wpedantic>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wconversion>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wsign-conversion>
)

install(TARGETS tuna_server RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES README.md DESTINATION "${CMAKE_INSTALL_DOCDIR}")
install(FILES deploy/tuna-server.service DESTINATION "lib/systemd/system")

if(BUILD_TESTING)
    add_executable(tuna_server_tests
        tests/workload_service_test.cpp
        src/workload_service.cpp
        ${TUNA_GPU_BACKEND}
        "${TUNA_PROTO_SOURCE}"
        "${TUNA_GRPC_SOURCE}"
    )
    target_include_directories(tuna_server_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${TUNA_GENERATED_DIR}"
    )
    target_link_libraries(tuna_server_tests PRIVATE gRPC::grpc++ protobuf::libprotobuf)
    if(TUNA_ENABLE_CUDA)
        set_target_properties(tuna_server_tests PROPERTIES
            CUDA_ARCHITECTURES 86
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        target_link_libraries(tuna_server_tests PRIVATE CUDA::cudart)
        target_compile_definitions(tuna_server_tests PRIVATE TUNA_ENABLE_CUDA=1)

        add_executable(tuna_gpu_matrix_tests
            tests/gpu_matrix_test.cpp
            src/gpu_matrix.cu
        )
        set_target_properties(tuna_gpu_matrix_tests PROPERTIES
            CUDA_ARCHITECTURES 86
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        target_include_directories(tuna_gpu_matrix_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(tuna_gpu_matrix_tests PRIVATE CUDA::cudart)
        add_test(NAME tuna_cuda_matrix_multiply COMMAND tuna_gpu_matrix_tests)
    endif()
    add_test(NAME tuna_server_workload COMMAND tuna_server_tests)

    if(UNIX AND TARGET tuna_sample_app)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        find_program(OPENSSL_EXECUTABLE openssl)
        if(Python3_Interpreter_FOUND AND OPENSSL_EXECUTABLE)
            set(TUNA_MTLS_TEST_COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/../tests/integration/test_mtls_round_trip.py"
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
