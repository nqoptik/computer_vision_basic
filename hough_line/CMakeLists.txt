cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(hough_line)

find_package(OpenCV REQUIRED)

add_executable(hough_line src/hough_line.cpp)
target_link_libraries(hough_line ${OpenCV_LIBS})
