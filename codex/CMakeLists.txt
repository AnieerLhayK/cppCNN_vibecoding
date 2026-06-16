cmake_minimum_required(VERSION 3.20)

project(CppCNNTrafficSign
    VERSION 2.0.0
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
option(CPPCNN_WITH_LIBTORCH "Build with LibTorch GPU backend (requires CUDA + LibTorch 2.x)" OFF)

find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc highgui)

add_library(cppcnn_core
    src/cnn/Activation.cpp
    src/cnn/Augmenter.cpp
    src/cnn/CNN.cpp
    src/cnn/ConvLayer.cpp
    src/cnn/DropoutLayer.cpp
    src/cnn/FCLayer.cpp
    src/cnn/FlattenLayer.cpp
    src/cnn/Loss.cpp
    src/cnn/LRScheduler.cpp
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
install(TARGETS cppcnn_app RUNTIME DESTINATION .)

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
            src/gui/InferenceEngine.cpp
            src/gui/InferenceEngine.h
            src/gui/PredictionResult.h
            src/gui/ResourceLocator.cpp
            src/gui/ResourceLocator.h
        )

        qt_add_qml_module(cppcnn_gui
            URI CppCNNGui
            VERSION 1.0
            QML_FILES
                qml/Main.qml
                qml/components/AppButton.qml
                qml/components/InfoChip.qml
                qml/components/Panel.qml
                qml/components/ProbabilityBar.qml
                qml/components/StatusPill.qml
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
        target_compile_definitions(cppcnn_gui PRIVATE
            CPPCNN_VERSION="${PROJECT_VERSION}"
        )
        install(TARGETS cppcnn_gui RUNTIME DESTINATION .)

        qt_generate_deploy_qml_app_script(
            TARGET cppcnn_gui
            OUTPUT_SCRIPT cppcnn_gui_deploy_script
            NO_TRANSLATIONS
        )
        install(SCRIPT ${cppcnn_gui_deploy_script})

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

if(CPPCNN_WITH_LIBTORCH)
    # LibTorch from the pre-installed SDK (set Torch_DIR if different)
    set(Torch_DIR "D:/SDK/libtorch-2.12.0-cu130/share/cmake/Torch"
        CACHE PATH "Path to TorchConfig.cmake")
    find_package(Torch REQUIRED)

    # LibTorch CNN components (depends on cppcnn_core for data loading)
    add_library(cppcnn_libtorch
        src/cnn_libtorch/LibTorchModel.cpp
        src/cnn_libtorch/GTSRBDataset.cpp
        src/cnn_libtorch/LibTorchTrainer.cpp
    )
    target_include_directories(cppcnn_libtorch PUBLIC src)
    target_link_libraries(cppcnn_libtorch PUBLIC
        cppcnn_core
        ${TORCH_LIBRARIES}
    )
    if(MSVC)
        target_compile_options(cppcnn_libtorch PRIVATE /W4 /permissive-)
    else()
        target_compile_options(cppcnn_libtorch PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # GPU-accelerated CLI
    add_executable(cppcnn_app_gpu src/main_libtorch.cpp)
    target_link_libraries(cppcnn_app_gpu PRIVATE
        cppcnn_libtorch
        cppcnn_core
        ${TORCH_LIBRARIES}
    )
    if(MSVC)
        target_compile_options(cppcnn_app_gpu PRIVATE /W4 /permissive-)
    else()
        target_compile_options(cppcnn_app_gpu PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    install(TARGETS cppcnn_app_gpu RUNTIME DESTINATION .)

    message(STATUS "LibTorch ${TORCH_VERSION} found at ${TORCH_INSTALL_PREFIX}")
    message(STATUS "CUDA enabled: ${TORCH_CUDA}")
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

    if(TARGET cppcnn_gui)
        add_executable(cppcnn_gui_tests
            tests/test_gui_controller.cpp
            src/gui/AppController.cpp
            src/gui/AppController.h
            src/gui/ImageBridge.cpp
            src/gui/ImageBridge.h
            src/gui/InferenceEngine.cpp
            src/gui/InferenceEngine.h
            src/gui/PredictionResult.h
            src/gui/ResourceLocator.cpp
            src/gui/ResourceLocator.h
        )
        target_compile_definitions(cppcnn_gui_tests PRIVATE
            CPPCNN_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
            CPPCNN_VERSION="${PROJECT_VERSION}"
        )
        target_link_libraries(cppcnn_gui_tests PRIVATE
            cppcnn_core
            Qt6::Core
            Qt6::Gui
            Qt6::Concurrent
        )
        add_test(NAME cppcnn_gui_tests COMMAND cppcnn_gui_tests)
        set_tests_properties(cppcnn_gui_tests PROPERTIES
            ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
            ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:Qt6::Core>"
        )
    endif()
endif()


