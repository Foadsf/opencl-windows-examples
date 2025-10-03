cmake_minimum_required(VERSION 3.15)
project(NBodySimulation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(nbody_simulation main.cpp)

target_link_libraries(nbody_simulation 
    OpenCL::OpenCL 
    OpenMP::OpenMP_CXX
)

configure_file(nbody.cl ${CMAKE_BINARY_DIR}/nbody.cl COPYONLY)