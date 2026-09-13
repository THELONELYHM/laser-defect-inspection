cmake_minimum_required(VERSION 3.20)
project(laser_defect_inspection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4.8 REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio dnn)

add_executable(laser_inspection
    src/main.cpp
    src/yolo_detector.cpp
)
target_include_directories(laser_inspection PRIVATE include)
target_link_libraries(laser_inspection PRIVATE ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(laser_inspection PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(laser_inspection PRIVATE -Wall -Wextra -Wpedantic)
endif()

