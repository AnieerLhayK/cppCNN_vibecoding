# cppCNN Traffic Sign Studio v1.0.0

这是 Codex 实现的首个可直接演示版本。CNN 的 Tensor、卷积、池化、激活、全连接、Softmax、损失、训练和模型持久化均由 C++17 实现，不依赖 PyTorch、TensorFlow 或 Keras。

## 下载与运行

1. 下载 `cppCNN-Traffic-Sign-Studio-v1.0.0-windows-x64.zip`。
2. 完整解压 ZIP。
3. 双击 `run_demo.bat` 或 `cppcnn_gui.exe`。
4. 点击底部任意演示图片即可运行识别。

便携包已包含 Qt 运行库、10 类模型权重、类别标签和五张演示图片，无需安装开发工具或重新训练。

## 本版本内容

- Qt Quick 深色桌面界面。
- 图片选择、拖放、预览和演示图片快捷入口。
- Top-1 类别、置信度、Top-3 概率和推理耗时。
- 模型、图片、结果三阶段状态反馈。
- `Ctrl+O` 打开图片、`Ctrl+Enter` 识别、`Esc` 清除。
- 模型缺失或文件无效时给出明确提示，不直接崩溃。
- CLI 训练、评估、单图预测与交互预测。

## 模型指标

- GTSRB 10 类子集。
- 10,000 张训练图，5,670 张测试图。
- 5 个 Epoch。
- 测试集 Top-1 准确率：89.63%。

## 已知限制

- 仅提供 Windows x64 便携包。
- 默认模型为 10 类而非完整 43 类。
- CNN 使用 CPU 朴素循环，完整训练速度有限。
- GUI 用于推理演示，训练通过 CLI 完成。

压缩包完整性可使用同名 `.sha256` 文件校验。

