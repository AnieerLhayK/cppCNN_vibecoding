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

find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc highgui)

add_library(cppcnn_core
    src/cnn/Activation.cpp
    src/cnn/CNN.cpp
    src/cnn/ConvLayer.cpp
    src/cnn/FCLayer.cpp
    src/cnn/FlattenLayer.cpp
    src/cnn/Loss.cpp
    src/cnn/PoolingLayer.cpp
    src/cnn/Tensor.cpp
    src/cnn/Trainer.cpp
    src/data/DataLoader.cpp
    src/image/ImageProcessor.cpp
)

target_include_directories(cppcnn_core PUBLIC src)

if(OpenCV_FOUND)
    target_compile_definitions(cppcnn_core PUBLIC CPP_CNN_WITH_OPENCV=1)
    target_include_directories(cppcnn_core PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(cppcnn_core PUBLIC ${OpenCV_LIBS})
    message(STATUS "OpenCV found: common image formats are enabled.")
else()
    message(STATUS "OpenCV not found: the portable PPM image loader remains available.")
endif()

if(MSVC)
    target_compile_options(cppcnn_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(cppcnn_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CPPCNN_BUILD_TESTS)
    enable_testing()

    add_executable(cppcnn_basic_tests tests/test_basic.cpp)
    target_link_libraries(cppcnn_basic_tests PRIVATE cppcnn_core)

    add_test(NAME cppcnn_basic_tests COMMAND cppcnn_basic_tests)
endif()
