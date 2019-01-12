cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(sudoku_detection)

find_package(OpenCV REQUIRED)

add_executable(sudoku_detection src/sudoku_detection.cpp)
target_link_libraries(sudoku_detection ${OpenCV_LIBS})
