# 纯 C++ CNN 交通标志识别系统

## 项目简介

这是 `cppCNN_vibecoding` 仓库中的 Codex 实现：一个面向本科 C++ 课程设计、可训练和可推理的交通标志分类系统。CNN 数值核心全部使用 C++17 编写，OpenCV 只负责常见图片格式读取和可选的预测窗口。

## 功能列表

- 连续内存 `Tensor` 与三维索引
- Conv、ReLU、MaxPooling、Flatten、Fully Connected、Softmax
- 交叉熵损失、Top-1 预测和准确率统计
- 小批量 SGD、L2 权重衰减
- 二进制模型保存、加载和架构检查
- GTSRB 类别目录扫描、类别/样本数量限制、固定种子洗牌
- 单图预测、交互预测、测试集评估
- OpenCV 图片窗口；无 OpenCV 时保留 PPM 和控制台保底模式
- 合成二分类演示数据生成器和 CTest 回归测试

当前版本同时支持训练与推理。

## 环境依赖

必需：

- Windows 10/11
- Visual Studio 2022，“使用 C++ 的桌面开发”
- CMake 3.20+

可选：

- OpenCV 4.x：启用 PNG/JPEG/BMP 和图片显示窗口

没有 OpenCV 时项目仍可编译，支持 PPM 图片、训练、评估和控制台预测。

## 数据集准备

推荐使用 GTSRB，并整理为：

```text
dataset/
|-- train/
|   |-- 00000/
|   |-- 00001/
|   `-- ...
`-- test/
    |-- 00000/
    |-- 00001/
    `-- ...
```

每个类别目录中放该类图片。程序按数字目录排序，将选中的原始类别映射为连续的 `0..N-1` 标签。默认取前 10 类，传入 `43` 即可扩展到全部类别。完整说明见 [docs/dataset_guide.md](docs/dataset_guide.md)。

不要将完整 GTSRB、生成模型或构建目录提交到 Git。

## 编译方式

在 Visual Studio Developer PowerShell 或普通 PowerShell 中：

```powershell
cd codex
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

启用 OpenCV：

```powershell
cmake -S . -B build -DOpenCV_DIR="C:\opencv\build"
cmake --build build --config Release
```

CMake 会生成 Visual Studio 解决方案 `build/CppCNNTrafficSign.sln`，也可直接用 Visual Studio 打开。

## 运行方式

以下命令从 `codex/` 目录运行，Release 可执行文件默认位于 `build/Release/`。

查看帮助：

```powershell
.\build\Release\cppcnn_app.exe help
```

训练默认 10 类、5 个 epoch：

```powershell
.\build\Release\cppcnn_app.exe train `
  D:\datasets\gtsrb\train models\gtsrb10.bin 10 5 0
```

最后一个参数是每类样本上限，`0` 表示不限制。快速调试可设为 `50`。

评估：

```powershell
.\build\Release\cppcnn_app.exe evaluate `
  D:\datasets\gtsrb\test models\gtsrb10.bin 0
```

单图预测：

```powershell
.\build\Release\cppcnn_app.exe predict `
  D:\datasets\gtsrb\test\00000\00000_00000.ppm `
  models\gtsrb10.bin assets\labels.txt
```

交互模式：

```powershell
.\build\Release\cppcnn_app.exe interactive `
  models\gtsrb10.bin assets\labels.txt
```

## 无数据集演示

先生成轻量合成数据，再跑通训练、评估和预测：

```powershell
.\build\Release\cppcnn_demo_generator.exe demo_data
.\build\Release\cppcnn_app.exe train demo_data\train models\demo.bin 2 3 0
.\build\Release\cppcnn_app.exe evaluate demo_data\test models\demo.bin 0
.\build\Release\cppcnn_app.exe predict `
  demo_data\test\00000\sample_0.ppm models\demo.bin
```

合成数据仅用于软件演示，不代表真实交通标志精度。

## 项目结构

```text
codex/
|-- assets/                 # 默认标签
|-- docs/                   # 报告、设计和数据集指南
|-- include/                # 预留公共头文件目录
|-- models/                 # 模型说明，二进制模型被忽略
|-- src/
|   |-- app/                # 命令行应用流程
|   |-- cnn/                # CNN 层、网络、损失和训练器
|   |-- data/               # 数据集扫描和标签映射
|   |-- image/              # 图片读取、缩放和归一化
|   |-- ui/                 # OpenCV 可选显示窗口
|   `-- main.cpp
|-- tests/                  # 基础回归测试
|-- tools/                  # 合成演示数据生成器
`-- CMakeLists.txt
```

## 当前限制

- CPU 单线程朴素循环实现，完整 43 类训练耗时较长。
- 网络固定接收 `3 x 32 x 32` RGB 输入。
- 当前 SGD 没有动量、Adam、批归一化或数据增强。
- 默认标签文件列出 GTSRB 前 10 类；扩展类别时需补充标签文本。
- 无 OpenCV 构建仅支持 PPM/PNM，不能弹出图片窗口。
- 仓库未包含真实 GTSRB 或预训练模型，因此真实准确率需在本地训练后获得。

## 后续改进方向

- OpenMP、SIMD 或 im2col 加速卷积
- 动量 SGD/Adam 和学习率衰减
- 随机裁剪、旋转、亮度扰动等数据增强
- 训练验证集划分、混淆矩阵和 Top-k 指标
- 更完整的 Windows GUI
- 完整 43 类预训练模型与可复现实验配置
