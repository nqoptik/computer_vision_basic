cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(brother_earth)

find_package(OpenCV REQUIRED)

add_executable(cross_light_detection src/cross_light_detection.cpp)
target_link_libraries(cross_light_detection ${OpenCV_LIBS})

add_executable(fast_hough_circle src/fast_hough_circle.cpp)
target_link_libraries(fast_hough_circle ${OpenCV_LIBS})

add_executable(orientation_fft src/orientation_fft.cpp)
target_link_libraries(orientation_fft ${OpenCV_LIBS} fftw3)
