cmake_minimum_required(VERSION 3.15)
project(BreakevenAnalysis CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

add_executable(breakeven_analysis main.cpp)
target_link_libraries(breakeven_analysis OpenCL::OpenCL)

configure_file(vector_add.cl ${CMAKE_BINARY_DIR}/vector_add.cl COPYONLY)