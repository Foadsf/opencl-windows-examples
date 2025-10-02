cmake_minimum_required(VERSION 3.15)
project(ParallelizationComparison CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(parallelization_comparison main.cpp)

target_link_libraries(parallelization_comparison 
    OpenCL::OpenCL 
    OpenMP::OpenMP_CXX
)

configure_file(matvec.cl ${CMAKE_BINARY_DIR}/matvec.cl COPYONLY)