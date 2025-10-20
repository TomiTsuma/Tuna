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
