# 设计说明

## 1. 设计目标

项目优先保证核心 CNN 算法可读、可调试、可在 Visual Studio 中展示。图像和界面依赖被限制在边缘模块，数值核心不依赖 OpenCV，也不使用 PyTorch、TensorFlow 或 Keras。

## 2. 模块划分

- `cnn/Tensor`：以 CHW 顺序保存连续 `float` 数据。
- `cnn/Layer`：统一前向、反向、清梯度和参数更新接口。
- `cnn/ConvLayer`：多输入/输出通道二维卷积，支持 stride 和 padding。
- `cnn/Activation`：ReLU 和数值稳定 Softmax。
- `cnn/PoolingLayer`：最大池化并保存最大值位置用于反向传播。
- `cnn/FlattenLayer`：仅改变张量逻辑形状。
- `cnn/FCLayer`：稠密线性变换及参数梯度。
- `cnn/Loss`：交叉熵、argmax、Accuracy。
- `cnn/CNN`：固定 LeNet 风格拓扑和模型持久化。
- `cnn/Trainer`：小批量 SGD、评估和进度输出。
- `image/ImageProcessor`：图片读取、双线性缩放、`[0,1]` 归一化。
- `data/DataLoader`：数字类别目录扫描和连续标签映射。
- `app/App`：命令行参数和训练/评估/预测流程。
- `ui/SimpleUI`：可选 OpenCV 图片窗口。

## 3. 数据流

```text
图片路径
  -> ImageProcessor(load, resize, normalize)
  -> Tensor(3 x 32 x 32)
  -> Conv/ReLU/Pool
  -> Conv/ReLU/Pool
  -> Flatten
  -> FC/ReLU/FC
  -> Softmax
  -> 类别概率、Top-1、交叉熵和准确率
```

网络维度：

```text
3x32x32
 -> Conv(3,6,5)   = 6x28x28
 -> MaxPool(2,2)  = 6x14x14
 -> Conv(6,16,5)  = 16x10x10
 -> MaxPool(2,2)  = 16x5x5
 -> Flatten       = 400
 -> FC             = 120
 -> FC             = class_count
 -> Softmax
```

## 4. 训练设计

每个样本完成前向传播、交叉熵求值和反向传播，梯度在一个 mini-batch 内累积。批次结束后按批大小缩放梯度，使用 SGD 更新，并对权重施加 L2 衰减。偏置不做权重衰减。

Softmax 和交叉熵保持为两个独立模块，便于课程展示各自公式。反向传播通过 Softmax Jacobian-vector product 实现，避免显式构造平方矩阵。

## 5. 模型格式

模型采用本项目自定义二进制格式，包含：

- 魔数 `CPPCNN1`
- 格式版本
- 类别数
- 可训练层数量和类型
- 各卷积/全连接层的权重和偏置

加载时会检查版本、类别数、层顺序和参数数量，错误模型不会静默读入。

## 6. 可移植性

OpenCV 通过 CMake `find_package` 可选发现。没有 OpenCV 时：

- PPM/PNM 图片仍可读取；
- 纯 C++ 双线性缩放仍可用；
- 训练、评估、模型保存和控制台预测仍可运行；
- 仅 PNG/JPEG/BMP 和弹窗显示不可用。

## 7. 错误处理

对不存在的目录/图片、空数据集、错误类别数、无效模型、尺寸不匹配和错误命令参数均抛出带上下文的异常，由 `main.cpp` 统一转为用户可读错误信息和非零退出码。
