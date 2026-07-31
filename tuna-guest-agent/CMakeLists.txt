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

add_executable(tuna_tests
    tests/unit/test_config.cpp
    tests/unit/test_hooks.cpp
    tests/unit/test_tunnel.cpp
    tests/integration/test_full_agent.cpp
)
target_link_libraries(tuna_tests PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(tuna_tests)
