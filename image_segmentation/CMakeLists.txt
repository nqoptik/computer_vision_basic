cmake_minimum_required(VERSION 3.1)

project(image_segmentation)

## Compile as C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Compile with the highest warning level
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

## System dependencies
find_package(OpenCV 3 REQUIRED)

## Declare C++ executables
add_executable(watershed_segmentation src/watershed_segmentation.cpp)

add_executable(grabcut_segmentation src/grabcut_segmentation.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(watershed_segmentation ${OpenCV_LIBS})

target_link_libraries(grabcut_segmentation ${OpenCV_LIBS})
