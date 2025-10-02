cmake_minimum_required(VERSION 3.15)
project(MatrixMultiply CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(matrix_multiply main.cpp)

target_link_libraries(matrix_multiply 
    OpenCL::OpenCL 
    OpenMP::OpenMP_CXX
)

configure_file(matmul.cl ${CMAKE_BINARY_DIR}/matmul.cl COPYONLY)