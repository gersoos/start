cmake_minimum_required (VERSION 2.8)

get_filename_component(a_dir ${CMAKE_CURRENT_SOURCE_DIR} NAME)
message(STATUS "Building ${a_dir}")

project(${a_dir})

find_package (Qt5Widgets)

set (MyProjectLib_src mainwindow.cpp)
set (MyProjectLib_hdr mainwindow.h)
set (MyProjectLib_ui  mainwindow.ui)
set (MyProjectBin_src main.cpp)

qt5_wrap_cpp(MyProjectLib_hdr_moc ${MyProjectLib_hdr})
qt5_wrap_ui (MyProjectLib_ui_moc  ${MyProjectLib_ui})

include_directories (${PROJECT_SOURCE_DIR})
include_directories (${PROJECT_BINARY_DIR})

add_library ("lib${a_dir}" SHARED 
    ${MyProjectLib_src}
    ${MyProjectLib_hdr_moc}
    ${MyProjectLib_ui_moc}
)
target_link_libraries ("lib${a_dir}" Qt5::Widgets)

add_executable(${a_dir} ${MyProjectBin_src})
target_link_libraries (${a_dir} "lib${a_dir}")
