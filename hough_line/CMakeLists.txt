cmake_minimum_required(VERSION 3.1)

project(hough_line)

## Compile as C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Compile with the highest warning level
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

## System dependencies
find_package(OpenCV REQUIRED)

## Declare a C++ executable
add_executable(hough_line src/hough_line.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(hough_line ${OpenCV_LIBS})
