cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

if(NOT OpenCV_DIR)
  set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/build_rel/opencv/share/OpenCV  CACHE STRING
      "opencv directory"
      FORCE)
endif(NOT OpenCV_DIR)

add_subdirectory(src bin)
