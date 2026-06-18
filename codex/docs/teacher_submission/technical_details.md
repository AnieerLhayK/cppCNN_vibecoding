# CNN 网络参数与技术细节

## 1. 输入与预处理

| 项目 | 参数 |
| --- | --- |
| 输入尺寸 | `3 x 32 x 32` |
| 通道顺序 | RGB，Tensor 内部为 CHW |
| 像素范围 | `[0, 1]` |
| 图片处理 | 解码、缩放、颜色转换、归一化 |
| 数据增强 | 旋转、平移、缩放、亮度、对比度、高斯噪声 |

## 2. LeNet 基线

适用于纯 C++ 10 类训练与早期基线实验。

| 顺序 | 层 | 主要参数 | 输出尺寸 | 可训练参数 |
| ---: | --- | --- | --- | ---: |
| 1 | Conv2D | `3 -> 6`，`5x5`，stride 1，padding 0 | `6x28x28` | 456 |
| 2 | ReLU | 逐元素 | `6x28x28` | 0 |
| 3 | MaxPool | `2x2`，stride 2 | `6x14x14` | 0 |
| 4 | Conv2D | `6 -> 16`，`5x5`，stride 1，padding 0 | `16x10x10` | 2,416 |
| 5 | ReLU | 逐元素 | `16x10x10` | 0 |
| 6 | MaxPool | `2x2`，stride 2 | `16x5x5` | 0 |
| 7 | Flatten | `16x5x5 -> 400` | `400` | 0 |
| 8 | Fully Connected | `400 -> 120` | `120` | 48,120 |
| 9 | ReLU | 逐元素 | `120` | 0 |
| 10 | Fully Connected | `120 -> N` | `N` | `121N + N` 的展开结果 |
| 11 | Softmax / LogSoftmax | 类别概率 | `N` | 0 |

10 类 LeNet 总参数量为 52,202。

## 3. Enhanced 正式网络

用于完整 43 类正式训练和最终模型。

| 顺序 | 层 | 主要参数 | 输出尺寸 | 可训练参数 |
| ---: | --- | --- | --- | ---: |
| 1 | Conv2D | `3 -> 32`，`5x5`，stride 1，padding 2 | `32x32x32` | 2,432 |
| 2 | ReLU | 逐元素 | `32x32x32` | 0 |
| 3 | MaxPool | `2x2`，stride 2 | `32x16x16` | 0 |
| 4 | Conv2D | `32 -> 64`，`3x3`，stride 1，padding 1 | `64x16x16` | 18,496 |
| 5 | ReLU | 逐元素 | `64x16x16` | 0 |
| 6 | MaxPool | `2x2`，stride 2 | `64x8x8` | 0 |
| 7 | Conv2D | `64 -> 128`，`3x3`，stride 1，padding 1 | `128x8x8` | 73,856 |
| 8 | ReLU | 逐元素 | `128x8x8` | 0 |
| 9 | MaxPool | `2x2`，stride 2 | `128x4x4` | 0 |
| 10 | Flatten | `128x4x4 -> 2048` | `2048` | 0 |
| 11 | Fully Connected | `2048 -> 512` | `512` | 1,049,088 |
| 12 | ReLU | 逐元素 | `512` | 0 |
| 13 | Dropout | `p=0.5` | `512` | 0 |
| 14 | Fully Connected | `512 -> 256` | `256` | 131,328 |
| 15 | ReLU | 逐元素 | `256` | 0 |
| 16 | Dropout | `p=0.5` | `256` | 0 |
| 17 | Fully Connected | `256 -> 43` | `43` | 11,051 |
| 18 | Softmax / LogSoftmax | 43 类概率 | `43` | 0 |

Enhanced 43 类总参数量为 1,286,251。

## 4. 损失、优化与评估

| 项目 | 实现 |
| --- | --- |
| 损失函数 | Cross Entropy / Negative Log Likelihood |
| 基础优化 | mini-batch SGD |
| 正式训练 | SGD + Momentum 0.9 + Weight Decay 0.0005 |
| 初始学习率 | 0.01 |
| 调度策略 | StepLR，每 24 epoch 乘 0.1 |
| Batch size | 64 |
| Epoch | 80 |
| 验证集 | 20% track-aware split |
| 类别处理 | class-balanced sampling |
| 主要指标 | Top-1、mean class accuracy、min class accuracy |

## 5. 模型文件格式

纯 C++ `.bin` 使用自定义 `CPPCNN1` 二进制格式：

| 字段 | 内容 |
| --- | --- |
| 魔数 | `CPPCNN1\0`，8 字节 |
| 格式版本 | `uint32`，当前为 2 |
| 类别数 | `uint64` |
| 网络架构 | `uint32`，0 为 LeNet，1 为 Enhanced |
| 可训练层数 | `uint32` |
| 层类型 | `uint32`，卷积为 1，全连接为 2 |
| 权重 | 数量 `uint64` + 连续 `float32` |
| 偏置 | 数量 `uint64` + 连续 `float32` |

加载时会检查魔数、版本、类别数、网络架构、层顺序、参数数量和文件完整性。

## 6. 代码对应关系

| 技术模块 | 源码 |
| --- | --- |
| Tensor | `source/src/cnn/Tensor.*` |
| 卷积 | `source/src/cnn/ConvLayer.*` |
| ReLU / Softmax | `source/src/cnn/Activation.*` |
| 最大池化 | `source/src/cnn/PoolingLayer.*` |
| Flatten | `source/src/cnn/FlattenLayer.*` |
| 全连接 | `source/src/cnn/FCLayer.*` |
| Dropout | `source/src/cnn/DropoutLayer.*` |
| Loss | `source/src/cnn/Loss.*` |
| 网络编排与模型格式 | `source/src/cnn/CNN.*` |
| CPU 训练 | `source/src/cnn/Trainer.*` |
| GPU 模型 | `source/src/cnn_libtorch/LibTorchModel.*` |
| GPU 训练 | `source/src/cnn_libtorch/LibTorchTrainer.*` |
| CMake 链接 | `source/CMakeLists.txt` |
