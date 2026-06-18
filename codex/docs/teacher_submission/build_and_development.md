# 编译、测试与开发者训练

## 1. CPU CLI 与核心测试

依赖：

- Windows 10/11 x64
- Visual Studio 2022，Desktop development with C++
- CMake 3.20+

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCPPCNN_BUILD_GUI=OFF `
  -DCPPCNN_BUILD_TESTS=ON `
  -DCPPCNN_WITH_LIBTORCH=OFF

cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

没有 OpenCV 时仍可读取 PPM；安装 OpenCV 后会自动启用 PNG、JPEG、BMP 等格式。

## 2. Qt GUI

额外依赖 Qt 6.8+ MSVC 2022 x64：

```powershell
cmake -S . -B build_gui -G "Visual Studio 17 2022" -A x64 `
  -DCPPCNN_BUILD_GUI=ON `
  -DCPPCNN_BUILD_TESTS=ON `
  -DCMAKE_PREFIX_PATH="C:\Qt\6.11.1\msvc2022_64"

cmake --build build_gui --config Release
ctest --test-dir build_gui -C Release --output-on-failure
```

## 3. LibTorch GPU 后端

额外依赖：

- NVIDIA GPU
- 与 LibTorch 匹配的 CUDA Toolkit
- LibTorch C++ CUDA 发行包

示例：

```powershell
cmake -S . -B build_libtorch -G "Visual Studio 17 2022" -A x64 `
  -T "cuda=D:\SDK\CUDA\v13.0" `
  -DCPPCNN_WITH_LIBTORCH=ON `
  -DCPPCNN_BUILD_GUI=OFF `
  -DTorch_DIR="D:\SDK\libtorch-2.12.0-cu130\share\cmake\Torch"

cmake --build build_libtorch --config Release --target cppcnn_app_gpu
```

CMake 中的关键链接关系：

```text
cppcnn_core
  -> cppcnn_app
  -> cppcnn_gui + Qt6
  -> cppcnn_libtorch + LibTorch
       -> cppcnn_app_gpu
```

GPU 可用性自检：

```powershell
.\build_libtorch\Release\cppcnn_app_gpu.exe cuda-test
```

## 4. 使用提交包样本训练

在 `source/` 目录执行：

```powershell
.\build\Release\cppcnn_app.exe train `
  ..\sample_dataset `
  ..\models\sample_retrained.bin `
  10 2 100 16 0.01 0.0001 42
```

样本包只用于验证训练接口。正式指标必须使用完整 GTSRB。

## 5. 评估与预测

```powershell
.\build\Release\cppcnn_app.exe evaluate `
  ..\sample_dataset `
  ..\models\gtsrb_v2_subset10.bin 20

.\build\Release\cppcnn_app.exe predict `
  ..\sample_dataset\test\00001\00001.ppm `
  ..\models\gtsrb_v2_subset10.bin `
  ..\models\gtsrb_v2_subset10.labels.txt
```

## 6. 开发边界

- GUI 不承担训练任务，保持用户端简单稳定。
- CPU 手写 CNN 用于课程验证、调试和推理。
- GPU 后端用于正式长训练，但不改变纯 C++ GUI 的运行依赖。
- 完整数据集、构建缓存和新训练权重不应提交到 Git。
