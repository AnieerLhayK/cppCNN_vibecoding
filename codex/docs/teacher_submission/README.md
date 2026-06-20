# 基于 C++ 的 CNN 交通标志识别系统

> **GitHub 仓库：** [github.com/AnieerLhayK/cppCNN_vibecoding](https://github.com/AnieerLhayK/cppCNN_vibecoding)
>
> 本项目全部源代码、CMake 构建配置、Git 版本历史、Release 下载以及 AI 辅助开发记录均托管于此。提交本包时请以此仓库为准。

## 一、项目工程性总结

本项目面向 GTSRB 德国交通标志数据集，完成了从数据读取、图像预处理、CNN 训练与评估、模型持久化，到 Qt 图形界面推理和 Windows 便携部署的完整闭环。

项目主体以 C++17 实现。核心 CNN 不依赖 PyTorch、TensorFlow 或 Keras，手工实现了 Tensor、卷积、ReLU、最大池化、Flatten、全连接、Softmax、交叉熵、反向传播、参数更新和准确率统计。由此验证了仅使用 C++ 完成小型卷积神经网络训练和推理的可行性。

为了进行完整 43 类、80 epoch 的正式训练，项目另外提供 C++ LibTorch GPU 后端。该后端用于提高训练效率，训练得到的 Enhanced 模型可导出为项目自定义的 `.bin` 格式，再由纯 C++ GUI 和 CPU CLI 加载推理。最终用户端不需要安装 CUDA、LibTorch、Qt、Visual Studio、Python 或数据集。

项目的主要工程特点包括：

- 核心算法可读：主要神经网络层均有独立 C++ 类和前向、反向实现。
- 训练与交付分离：开发者可使用 CPU 或 GPU 训练，用户只运行便携 GUI。
- 双网络结构：保留 LeNet 基线，并实现 3 Conv、3 FC、双 Dropout 的 Enhanced 网络。
- 模型可追溯：保留多个训练阶段的 checkpoint、标签、元数据、训练日志和版本说明。
- 数据管理规范：完整数据集不进入 Git；提交包仅附带可运行的小型样本。
- 自动化构建：CMake 统一管理核心库、CLI、Qt GUI、测试、子集工具和可选 LibTorch 后端。
- 可验证交付：包含单元测试、模型校验、SHA-256 清单和可直接运行的 Windows x64 应用。
- 协作过程可审计：项目使用 Git/GitHub 管理版本和发布，并附有 AI 辅助开发对话记录。

项目最初从课程中较简单的 MNIST CNN 作业思路出发，随后扩展到真实彩色数据、43 类识别、数据增强、类别均衡、GPU 训练、Qt GUI 和便携发布。复杂度增长后，项目使用 AI 辅助进行代码生成、调试、文档整理和工程维护；关键结果均通过编译、测试、训练日志或实际推理进行验证。

## 二、最终成果

| 项目 | 结果 |
| --- | --- |
| 数据集 | GTSRB，43 类 |
| 正式训练图像 | 39,209 张 |
| 官方测试图像 | 12,630 张 |
| 输入 | RGB，`3 x 32 x 32`，归一化到 `[0,1]` |
| 正式网络 | Enhanced，3 Conv + 3 FC + 2 Dropout |
| 参数量 | 1,286,251 |
| 正式训练 | 80 epoch，LibTorch GPU |
| 官方测试集 Top-1 | 95.26% |
| 最佳验证准确率 | 97.50% |
| 纯 C++ 10 类模型 | 89.63% |
| 纯 C++ 语义均衡 10 类模型 | 92.63% |
| 用户界面 | Qt Quick 桌面 GUI，Top-1、置信度、Top-3、推理耗时 |

## 三、提交包结构

> **GitHub 仓库：** [github.com/AnieerLhayK/cppCNN_vibecoding](https://github.com/AnieerLhayK/cppCNN_vibecoding) ← 请以此为准

```text
cppCNN-Codex-Teacher-Submission-v2.0.0/
├── README.md                    # 本文件：工程性报告与总说明
├── application/                 # 可直接运行的 Windows x64 GUI
├── source/                      # Codex 版本完整 C++ 源码与 CMake 工程
├── sample_dataset/              # 10 类小型训练/测试数据
├── models/                      # CPU、GPU 各阶段模型与版本材料
├── ai_records/                  # AI 对话 HTML、PDF 及说明
├── docs/
│   ├── technical_details.md     # 网络逐层参数表
│   ├── model_versions.md        # 模型版本与用途
│   ├── build_and_development.md # 编译、测试、训练方法
│   └── application_usage.md     # GUI 使用说明
├── SHA256SUMS.txt               # 关键文件校验值
└── PACKAGE_MANIFEST.txt         # 文件清单
```

## 四、直接演示

进入 `application/`，双击：

```text
run_demo.bat
```

程序自带 Qt 运行库、两个可加载的纯 C++ `.bin` 模型、类别标签和 50 张演示图片。默认加载 10 类模型；按 `Ctrl+M` 可切换到 `models/gtsrb_v5_full43.bin` 完整 43 类模型。

GUI 操作详见 `docs/application_usage.md`。

## 五、源码与构建

`source/` 仅包含 `codex/` 实现，不包含仓库中的其他平台原型。核心入口如下：

- `source/src/cnn/`：纯 C++ CNN、Tensor、训练器和损失函数。
- `source/src/cnn_libtorch/`：C++ LibTorch GPU 训练后端。
- `source/src/data/`：GTSRB 数据加载。
- `source/src/gui/` 与 `source/qml/`：Qt Quick GUI。
- `source/CMakeLists.txt`：所有目标、依赖和链接关系。
- `source/tests/`：核心与 GUI 控制器测试。

完整构建命令及 GPU 环境说明见 `docs/build_and_development.md`。

## 六、数据与模型

`sample_dataset/` 提供 10 类、每类 100 张训练图和 20 张测试图，共 1,200 张 PPM 图片，可直接用于 DataLoader、训练流程和评估流程验证。它不是正式实验使用的完整数据集。

正式实验使用完整 GTSRB。因数据集许可、体积和机器学习项目管理规范，完整数据集未放入提交包；来源与目录约定保留在源码文档中。

`models/` 包含：

- 纯 C++ `.bin` 模型：供 GUI 与 CPU CLI 推理。
- LibTorch `.pt` 模型：供开发者继续 GPU 实验。
- 标签、训练日志、模型卡和元数据。

各模型用途和指标见 `docs/model_versions.md`。

## 七、技术路线

系统推理流程为：

```text
图片选择
  -> 解码与缩放
  -> RGB 3x32x32 Tensor
  -> CNN 前向传播
  -> Softmax 概率
  -> Top-1 / Top-3
  -> Qt GUI 展示
```

训练流程为：

```text
GTSRB 目录扫描
  -> 预处理与数据增强
  -> mini-batch
  -> 前向传播
  -> 交叉熵
  -> 反向传播
  -> SGD + Momentum + Weight Decay
  -> 验证与 checkpoint
  -> .pt / .bin 模型
```

网络逐层输出、卷积核、步长、Padding 和参数量见 `docs/technical_details.md`。

## 八、已知限制

- 纯 C++ 卷积采用朴素循环，适合教学和推理验证，不适合长时间的大规模训练。
- GPU 后端依赖 NVIDIA CUDA 与匹配的 LibTorch，仅面向开发者，不属于用户端依赖。
- 便携 GUI 目前只提供图片分类，不在界面中提供训练功能。
- 小样本数据只用于验证流程，不能代表正式模型精度。
- 正式模型在少样本类别上的准确率仍低于高频类别，后续可通过 BatchNorm、更强重采样和更高分辨率改进。

## 九、结论

本项目不仅实现了一个能够运行的交通标志识别程序，也形成了较完整的机器学习工程：算法主体可解释、训练过程可复现、模型版本可追踪、GUI 可直接使用、构建与测试自动化、交付材料完整。纯 C++ 实现证明了课程要求中的 CNN 核心链路可以独立完成；GPU 后端和工程化工具则解决了完整 43 类正式训练与最终交付的效率问题。
