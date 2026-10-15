cmake_minimum_required(VERSION 3.21)
project(TunaClient LANGUAGES CXX)

include(CTest)
include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${configuration} "${CMAKE_BINARY_DIR}/bin")
endforeach()

# Track A (explicit-workload gRPC/mTLS MVP) targets are opt-in/opt-out here.
# Track B (driver-API remoting) core libraries and tests are always built.
option(TUNA_BUILD_GUI "Build the Qt management GUI" OFF)
option(TUNA_BUILD_SAMPLE_CLIENT "Build the gRPC sample client and workload API" ON)
option(TUNA_BUILD_LINUX_SERVER "Build ../tuna-server (with its workload API) in this tree" OFF)

if(TUNA_BUILD_SAMPLE_CLIENT OR TUNA_BUILD_LINUX_SERVER)
    set(TUNA_SYSTEM_PROTOBUF ON)  # gRPC is in the build; see cmake/TunaProtobuf.cmake
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/TunaProtobuf.cmake)

if(TUNA_BUILD_GUI)
    if(NOT WIN32)
        message(FATAL_ERROR "The Qt management GUI is a Windows client target; disable TUNA_BUILD_GUI on Linux.")
    endif()
    find_package(Qt6 6.5 REQUIRED COMPONENTS Widgets)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)

    add_executable(tuna_gui
        src/gui/main.cpp
        src/gui/mainwindow.cpp
        src/gui/mainwindow.h
        src/gui/settings_dialog.cpp
        src/gui/settings_dialog.h
    )
    target_link_libraries(tuna_gui PRIVATE Qt6::Widgets)
    install(TARGETS tuna_gui RUNTIME DESTINATION bin)
endif()

if(TUNA_BUILD_SAMPLE_CLIENT)
    add_subdirectory(client)
endif()

# --- Track B core libraries (config, logging, error handling) ---
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

    add_executable(tuna_tests
        tests/unit/test_config.cpp
        tests/unit/test_logger.cpp
        tests/unit/test_error_handler.cpp
        tests/unit/test_frame_proto.cpp
    )
    target_link_libraries(tuna_tests PRIVATE Catch2::Catch2WithMain tuna_proto tuna_core)
    catch_discover_tests(tuna_tests)
endif()

# Added last so the server's mTLS integration test can see tuna_sample_app.
if(TUNA_BUILD_LINUX_SERVER)
    if(WIN32)
        message(FATAL_ERROR "The Linux server must be built on Linux; use its standalone CMake project.")
    endif()
    set(TUNA_SERVER_WORKLOAD_API ON CACHE BOOL "" FORCE)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../tuna-server ${CMAKE_BINARY_DIR}/tuna-server)
endif()
