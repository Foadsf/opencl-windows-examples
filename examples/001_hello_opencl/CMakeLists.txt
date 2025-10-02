cmake_minimum_required(VERSION 3.15)
project(HelloOpenCL CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

add_executable(hello_opencl main.cpp)
target_link_libraries(hello_opencl OpenCL::OpenCL)

# Copy kernel file to build directory
configure_file(hello.cl ${CMAKE_BINARY_DIR}/hello.cl COPYONLY)

message(STATUS "OpenCL Include: ${OpenCL_INCLUDE_DIRS}")
message(STATUS "OpenCL Library: ${OpenCL_LIBRARIES}")