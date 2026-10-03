cmake_minimum_required(VERSION 3.21)
project(TunaClient LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${configuration} "${CMAKE_BINARY_DIR}/bin")
endforeach()

option(TUNA_BUILD_GUI "Build the Qt management GUI" OFF)
option(TUNA_BUILD_SAMPLE_CLIENT "Build the gRPC sample client and workload API" ON)
option(TUNA_BUILD_LINUX_SERVER "Build the Linux server in this repository" OFF)

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
        src/gui/mainwindow.ui
        src/gui/settings_dialog.cpp
        src/gui/settings_dialog.h
    )
    target_link_libraries(tuna_gui PRIVATE Qt6::Widgets)
    install(TARGETS tuna_gui RUNTIME DESTINATION bin)
endif()

if(TUNA_BUILD_SAMPLE_CLIENT)
    add_subdirectory(client)
endif()

if(TUNA_BUILD_LINUX_SERVER)
    if(WIN32)
        message(FATAL_ERROR "The Linux server must be built on Linux; use its standalone CMake project.")
    endif()
    add_subdirectory(tuna-server)
endif()
