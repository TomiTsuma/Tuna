cmake_minimum_required(VERSION 3.21)
project(TunaSampleClient LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)

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

add_library(tuna_workload_client
    src/workload_client.cpp
    "${TUNA_PROTO_SOURCE}"
    "${TUNA_GRPC_SOURCE}"
)
target_compile_features(tuna_workload_client PUBLIC cxx_std_20)
target_include_directories(tuna_workload_client
    PUBLIC include "${TUNA_GENERATED_DIR}"
)
target_link_libraries(tuna_workload_client
    PUBLIC gRPC::grpc++ protobuf::libprotobuf
)

add_executable(tuna_sample_app src/sample_app.cpp)
target_link_libraries(tuna_sample_app PRIVATE tuna_workload_client)
install(TARGETS tuna_sample_app RUNTIME DESTINATION bin)
