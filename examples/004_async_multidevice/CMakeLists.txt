cmake_minimum_required(VERSION 3.10)
project(async_multidevice)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCL REQUIRED)

add_executable(async_multidevice main.cpp)
target_link_libraries(async_multidevice OpenCL::OpenCL)