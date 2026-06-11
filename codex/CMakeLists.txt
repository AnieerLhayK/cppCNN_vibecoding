cmake_minimum_required(VERSION 3.20)

project(CppCNNTrafficSign
    VERSION 0.1.0
    DESCRIPTION "Pure C++ CNN traffic sign recognition course project"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CPPCNN_BUILD_TESTS "Build unit tests" ON)

message(STATUS "Project skeleton initialized. Source targets are added in later stages.")
