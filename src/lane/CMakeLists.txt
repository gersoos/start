cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

get_filename_component(a_dir ${CMAKE_CURRENT_SOURCE_DIR} NAME)
message(STATUS "Building ${a_dir}")

project(${a_dir})

file(GLOB SOURCES "*.cpp")

find_package( OpenCV REQUIRED)

add_executable(${a_dir} ${SOURCES})
target_link_libraries(${a_dir} ${OpenCV_LIBS} )