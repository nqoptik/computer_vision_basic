cmake_minimum_required(VERSION 3.1)

project(canny_edge_detection)

## Compile as C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Compile with the highest warning level
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

## System dependencies
find_package(OpenCV REQUIRED)
if(NOT ${OpenCV_VERSION} STRGREATER "2.4")
    message(FATAL_ERROR "OpenCV_VERSION has to > 2.4")
endif()

## Declare a C++ executable
add_executable(canny_edge_detection src/canny_edge_detection.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(canny_edge_detection ${OpenCV_LIBS})
