cmake_minimum_required(VERSION 3.15)
project(ImageConvolution CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(image_convolution main.cpp)

target_link_libraries(image_convolution 
    OpenCL::OpenCL 
    OpenMP::OpenMP_CXX
)

configure_file(convolution.cl ${CMAKE_BINARY_DIR}/convolution.cl COPYONLY)