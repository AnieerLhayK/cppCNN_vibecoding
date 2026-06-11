# 纯 C++ CNN 交通标志识别系统

这是一个面向本科 C++ 课程设计的交通标志分类项目。项目在不依赖深度学习框架的前提下，用 C++ 实现 Tensor、卷积、ReLU、最大池化、全连接、Softmax、交叉熵损失、训练和评估流程。OpenCV 仅用于图片读写、预处理和简单交互显示。

## 当前状态

项目骨架已经建立，后续阶段将依次补充：

- Tensor 与基础数学运算
- 图片读取、缩放和归一化
- LeNet 风格 CNN 各基础层
- GTSRB 目录数据加载
- 训练、推理、模型保存和测试集评估
- 控制台与 OpenCV 简单前端
- 完整设计文档和实验报告

## 计划功能

- 默认使用 32 x 32 RGB 输入
- 默认支持可配置的 10 类 GTSRB 子集
- 类别数和每类样本数可配置，可扩展到完整 43 类
- 纯 C++ CNN 前向与反向传播
- 模型参数保存和加载
- 单图 Top-1 预测与置信度输出
- 测试集准确率统计

## 环境依赖

- Windows 10/11
- Visual Studio 2022，安装“使用 C++ 的桌面开发”
- CMake 3.20 或更高版本
- OpenCV 4.x

## 数据集

推荐使用 GTSRB。完整数据集不会提交到 Git；请将整理后的训练集和测试集放到仓库外或 `codex/data/` 下。详细目录约定将在 `docs/dataset_guide.md` 中给出。

## 构建

最终版本将支持以下 CMake 工作流：

```powershell
cd codex
cmake -S . -B build -DOpenCV_DIR="C:\path\to\opencv\build"
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## 项目结构

```text
codex/
|-- assets/
|-- docs/
|-- include/
|-- models/
|-- src/
|   |-- app/
|   |-- cnn/
|   |-- data/
|   |-- image/
|   `-- ui/
|-- tests/
|-- CMakeLists.txt
`-- README.md
```

## 当前限制

当前提交仅完成项目初始化，尚不能编译运行。后续稳定阶段会分别提交并推送。

## 后续改进方向

- 数据增强与更深的网络结构
- 多线程数据加载
- SIMD 或 OpenMP 加速
- 更完整的桌面 GUI
- 完整 43 类模型和更系统的超参数搜索
