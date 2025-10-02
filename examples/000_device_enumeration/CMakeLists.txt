cmake_minimum_required(VERSION 3.15)
project(DeviceEnumeration CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

add_executable(device_enumeration main.cpp)
target_link_libraries(device_enumeration OpenCL::OpenCL)

message(STATUS "OpenCL Include: ${OpenCL_INCLUDE_DIRS}")
message(STATUS "OpenCL Library: ${OpenCL_LIBRARIES}")