cmake_minimum_required(VERSION 3.16)
project(TunaServer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(tuna_server src/main.cpp)

# --- Test harness (Catch2 + CTest) ---
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
)
FetchContent_MakeAvailable(Catch2)

enable_testing()
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

add_executable(tuna_server_tests tests/test_smoke.cpp)
target_link_libraries(tuna_server_tests PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(tuna_server_tests)
