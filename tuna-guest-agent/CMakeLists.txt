cmake_minimum_required(VERSION 3.16)
project(TunaGuestAgent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Point to your Qt installation
set(CMAKE_PREFIX_PATH "C:/Qt/6.10.0/mingw_64/lib/cmake")

find_package(Qt6 REQUIRED COMPONENTS Widgets Network)

add_executable(tuna_gui
    src/gui/main.cpp
    src/gui/mainwindow.cpp
    src/gui/mainwindow.h
    src/gui/mainwindow.ui
    src/gui/settings_dialog.cpp
)

target_link_libraries(tuna_gui PRIVATE Qt6::Widgets Qt6::Network)

include(FetchContent)

# --- Shared tunnel frame schema (protobuf) ---
FetchContent_Declare(
    protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG v21.12
)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(protobuf)

set(TUNA_PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../proto)
set(TUNA_PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${TUNA_PROTO_GEN_DIR})

add_custom_command(
    OUTPUT ${TUNA_PROTO_GEN_DIR}/tuna_frames.pb.cc ${TUNA_PROTO_GEN_DIR}/tuna_frames.pb.h
    COMMAND $<TARGET_FILE:protobuf::protoc> --cpp_out=${TUNA_PROTO_GEN_DIR} -I ${TUNA_PROTO_DIR} ${TUNA_PROTO_DIR}/tuna_frames.proto
    DEPENDS ${TUNA_PROTO_DIR}/tuna_frames.proto
    COMMENT "Generating C++ sources from tuna_frames.proto"
    VERBATIM
)

add_library(tuna_proto STATIC
    ${TUNA_PROTO_GEN_DIR}/tuna_frames.pb.cc
    ${TUNA_PROTO_GEN_DIR}/tuna_frames.pb.h
)
add_dependencies(tuna_proto protobuf::protoc)
target_include_directories(tuna_proto PUBLIC ${TUNA_PROTO_GEN_DIR})
target_link_libraries(tuna_proto PUBLIC protobuf::libprotobuf)

# --- Core libraries (config, logging, error handling) ---
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

add_library(tuna_core STATIC
    src/core/config_manager.cpp
    src/core/logger.cpp
    src/core/error_handler.cpp
)
target_include_directories(tuna_core PUBLIC src)
target_link_libraries(tuna_core PUBLIC nlohmann_json::nlohmann_json spdlog::spdlog)

# --- Test harness (Catch2 + CTest) ---
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
)
FetchContent_MakeAvailable(Catch2)

enable_testing()
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

add_executable(tuna_tests
    tests/unit/test_config.cpp
    tests/unit/test_logger.cpp
    tests/unit/test_error_handler.cpp
    tests/unit/test_hooks.cpp
    tests/unit/test_tunnel.cpp
    tests/unit/test_frame_proto.cpp
    tests/integration/test_full_agent.cpp
)
target_link_libraries(tuna_tests PRIVATE Catch2::Catch2WithMain tuna_proto tuna_core)
catch_discover_tests(tuna_tests)
