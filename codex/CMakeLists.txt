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
option(CPPCNN_BUILD_DEMO_GENERATOR "Build the synthetic demo dataset generator" ON)
option(CPPCNN_BUILD_SUBSET_TOOL "Build the GTSRB subset creation tool" ON)
option(CPPCNN_BUILD_GUI "Build the Qt Quick desktop application" ON)

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

add_executable(cppcnn_app
    src/main.cpp
    src/app/App.cpp
    src/ui/SimpleUI.cpp
)
target_link_libraries(cppcnn_app PRIVATE cppcnn_core)

if(MSVC)
    target_compile_options(cppcnn_app PRIVATE /W4 /permissive-)
else()
    target_compile_options(cppcnn_app PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CPPCNN_BUILD_GUI)
    find_package(Qt6 QUIET COMPONENTS Core Gui Quick Qml QuickControls2 Concurrent)
    if(Qt6_FOUND)
        qt_standard_project_setup(REQUIRES 6.8)

        qt_add_executable(cppcnn_gui WIN32
            src/gui/GuiMain.cpp
            src/gui/AppController.cpp
            src/gui/AppController.h
            src/gui/ImageBridge.cpp
            src/gui/ImageBridge.h
            src/gui/PredictionResult.h
        )

        qt_add_qml_module(cppcnn_gui
            URI CppCNNGui
            VERSION 1.0
            QML_FILES
                qml/Main.qml
        )

        target_link_libraries(cppcnn_gui PRIVATE
            cppcnn_core
            Qt6::Core
            Qt6::Gui
            Qt6::Quick
            Qt6::Qml
            Qt6::QuickControls2
            Qt6::Concurrent
        )

        if(MSVC)
            target_compile_options(cppcnn_gui PRIVATE /W4 /permissive-)
        else()
            target_compile_options(cppcnn_gui PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    else()
        message(WARNING
            "Qt 6 was not found. cppcnn_gui will not be built; "
            "set CMAKE_PREFIX_PATH to a Qt 6 MSVC installation.")
    endif()
endif()

if(CPPCNN_BUILD_DEMO_GENERATOR)
    add_executable(cppcnn_demo_generator tools/generate_demo_dataset.cpp)
    if(MSVC)
        target_compile_options(cppcnn_demo_generator PRIVATE /W4 /permissive-)
    else()
        target_compile_options(cppcnn_demo_generator PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

if(CPPCNN_BUILD_SUBSET_TOOL)
    add_executable(cppcnn_create_subset scripts/create_subset.cpp)
    if(MSVC)
        target_compile_options(cppcnn_create_subset PRIVATE /W4 /permissive-)
    else()
        target_compile_options(cppcnn_create_subset PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

if(CPPCNN_BUILD_TESTS)
    enable_testing()

    add_executable(cppcnn_basic_tests tests/test_basic.cpp)
    target_link_libraries(cppcnn_basic_tests PRIVATE cppcnn_core)

    add_test(NAME cppcnn_basic_tests COMMAND cppcnn_basic_tests)
endif()
