# cppCNN GPU Accelerated Training v2.0.0-gpu

本版本发布基于 LibTorch 的 GPU 加速训练后端。所有 GPU 功能与原始纯 C++ CPU 实现完全独立、互不干扰——`cppcnn_app_gpu.exe` 与原版 `cppcnn_app.exe` 及 `cppcnn_gui.exe` 共存。

## 新功能

- **GPU 加速训练**：通过 LibTorch 2.12.0 + CUDA 13.0 实现训练速度提升 15–80 倍。
- **双架构支持**：GPU 版 LeNet（~56K 参数）与 Enhanced（~1.3M 参数），通过 `--arch` 运行时选择。
- **完整训练管线**：train/evaluate/predict + data augmentation + class-balanced sampling + track-aware validation split。
- **Archive 序列化**：`saveToFile()` / `loadFromFile()` 支持多态模型类型安全持久化。
- **CUDA 自检**：`cuda-test` 命令一键验证 CUDA 工具链是否正常工作。

## 性能对比

| 场景 | 原 CPU（手写 CNN） | GPU（LibTorch） | 加速比 |
|------|-------------------|-----------------|--------|
| LeNet 1 epoch（10 类） | 32.8s | 2.2s | ~15× |
| Enhanced 1 epoch（43 类） | 15–30 min | 10–15s | ~80–120× |
| 完整 80 epoch 训练 | 20–40 小时 | 15–20 分钟 | ~80× |

## 环境要求

- NVIDIA GPU（测试环境：RTX 4060 Laptop，8 GB VRAM）
- CUDA 13.0+
- LibTorch 2.12.0-cu130
- Visual Studio 2022 + CMake 3.20+

## 构建方式

```powershell
cmake -S codex -B build_libtorch -G "Visual Studio 17 2022" -A x64 `
  -T "cuda=D:\SDK\CUDA\v13.0" `
  -DCPPCNN_WITH_LIBTORCH=ON `
  -DCUDA_TOOLKIT_ROOT_DIR="D:/SDK/CUDA/v13.0" `
  -DCPPCNN_BUILD_GUI=OFF

cmake --build build_libtorch --config Release
```

## 运行示例

```powershell
# CUDA 环境自检
build_libtorch\Release\cppcnn_app_gpu.exe cuda-test

# 训练（LeNet，10 类，5 epoch）
build_libtorch\Release\cppcnn_app_gpu.exe train codex\datasets\GTSRB_subset `
  codex\models\gpu_lenet10.pt --classes 10 --arch LeNet --epochs 5 `
  --batch 64 --lr 0.01 --verbose

# 训练（Enhanced，43 类，80 epoch）
build_libtorch\Release\cppcnn_app_gpu.exe train codex\datasets\GTSRB `
  codex\models\gtsrb43_enhanced.pt --classes 43 --arch Enhanced --epochs 80 `
  --batch 64 --lr 0.01 --momentum 0.9 --wd 0.0005 --val 0.2 --aug --balance

# 评估
build_libtorch\Release\cppcnn_app_gpu.exe evaluate codex\datasets\GTSRB `
  codex\models\gtsrb43_enhanced.pt --classes 43 --arch Enhanced
```

## 与原始实现的差异

| 方面 | 原始 CPU（cppcnn_app） | GPU 加速（cppcnn_app_gpu） |
|------|----------------------|--------------------------|
| 后端 | 手写 C++ Tensor/Conv | LibTorch 2.12.0（cuDNN） |
| 设备 | CPU（单线程） | GPU（RTX 4060） |
| 模型格式 | `.bin`（自定义格式） | `.pt`（PyTorch Archive） |
| 加速 | 基准 | 15–80× |
| CLI 参数 | 位置参数 | `--key=value` 风格 |

## 已知限制

- **GPU 验证回退**：CUDA 兼容性问题导致 GPU 前向传播在 `model.eval()` 模式下崩溃，验证和评估当前在 CPU 上执行。不影响训练速度。
- **非便携**：需要手动配置 CUDA 工具链，不包含在标准 `Release/` 便携包中。
- **模型不兼容**：`.pt` 格式与原始 `.bin` 格式不互通。
- **GUI 不可用**：GPU 版仅为 CLI 工具，无图形界面。
