cmake_minimum_required(VERSION 3.31)
project(pl_qcqp)

set(CMAKE_CXX_STANDARD 20)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
        ${EIGEN3_INCLUDE_DIRS}
)

pybind11_add_module(pl_qcqp bindings.cpp)

