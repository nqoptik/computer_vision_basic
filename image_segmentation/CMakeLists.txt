cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(image_segmentation)

find_package(OpenCV REQUIRED)

add_executable(watershed_segmentation src/watershed_segmentation.cpp)
target_link_libraries(watershed_segmentation ${OpenCV_LIBS})

add_executable(graphcuts_segmentation src/graphcuts_segmentation.cpp)
target_link_libraries(graphcuts_segmentation ${OpenCV_LIBS})
