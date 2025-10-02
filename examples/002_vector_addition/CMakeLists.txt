cmake_minimum_required(VERSION 3.15)
project(VectorAddition CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

add_executable(vector_addition main.cpp)
target_link_libraries(vector_addition OpenCL::OpenCL)

configure_file(vector_add.cl ${CMAKE_BINARY_DIR}/vector_add.cl COPYONLY)

message(STATUS "OpenCL Include: ${OpenCL_INCLUDE_DIRS}")
message(STATUS "OpenCL Library: ${OpenCL_LIBRARIES}")