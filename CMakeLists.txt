cmake_minimum_required(VERSION 3.28)
project(Starlab)

find_package(Eigen3 3.4 REQUIRED)

set(CMAKE_CXX_STANDARD 17)

add_executable(Starlab main.cpp
        QPSolver.h
        QPSolver.cpp)

target_link_libraries(Starlab Eigen3::Eigen)

# Optionally set up compile definitions to print Eigen version
target_compile_definitions(Starlab    PRIVATE EIGEN_VERSION=3.4)