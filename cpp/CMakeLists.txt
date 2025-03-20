cmake_minimum_required(VERSION 3.4)
project(call_python_from_cpp)

set(CMAKE_CXX_STANDARD 14)

find_package(pybind11 REQUIRED)

# Add optimization flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(main main.cpp)
target_link_libraries(main PRIVATE pybind11::embed)
