# Changelog

本文件记录 `codex/` 实现面向使用者的稳定版本变化。

## 1.0.0 - 2026-06-13

### Added

- 纯 C++ LeNet 风格 CNN：Tensor、Conv、ReLU、MaxPooling、Flatten、FC、Softmax、Loss。
- mini-batch SGD 训练、测试集评估、模型保存与加载。
- GTSRB 官方目录和 10 类开发子集读取。
- Qt Quick 桌面界面，支持打开、拖放、预览和演示图片。
- Top-1、置信度、Top-3 概率与推理耗时显示。
- 模型、图片和结果三阶段状态提示。
- 键盘快捷键、工具提示、运行时信息和可读错误提示。
- Windows CMake/Qt GitHub Actions 持续集成。
- Windows x64 便携发布包、版本文件和 SHA-256 校验文件。

### Architecture

- 将运行时资源查找集中到 `ResourceLocator`。
- 将 Top-K 推理和计时集中到 `InferenceEngine`。
- `AppController` 仅协调 GUI 状态、异步任务和错误反馈。

### Model

- 默认发布模型为 10 类 GTSRB 子集模型。
- 训练样本 10,000，测试样本 5,670。
- 5 个 Epoch 后测试集 Top-1 准确率为 89.63%。

