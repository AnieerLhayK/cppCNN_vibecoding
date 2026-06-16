# cppCNN Traffic Sign Studio v2.0.0

本版本是课程交付用的完整发布版，覆盖纯 C++ CNN 主程序、Qt GUI、完整 43 类模型、GPU 加速训练工具链、报告资料包和 AI 使用记录。

## 主要变化

- 新增完整 43 类 GTSRB Enhanced 模型 `gtsrb_v5_full43.bin`，可在 GUI 中通过 `Ctrl+M` 手动加载。
- 默认保留 10 类模型 `gtsrb_v2_subset10.bin`，便于打开程序后直接演示内置图片。
- GUI 增加模型切换能力，可在多个 `.bin` 模型之间切换并自动匹配同名标签文件。
- 完成 Phase B 基线对比：LeNet 与 Enhanced、无增强与有增强四组实验。
- 完成 Phase C 正式训练：Enhanced + augmentation + class-balanced sampling，80 epoch。
- 完成 `.pt -> .bin` 导出流程，使 GPU 训练得到的权重可进入纯 C++ GUI 推理链路。
- Release 包新增 `ai_records/codex_session_export.html`，用于提交 AI 使用记录。
- Report Kit 保留宝宝级指南，并将运行、报告和答辩说明合并到一个综合指南中；代码阅读指南单独保留。

## 模型结果

| 模型 | 类别 | 用途 | 指标 |
| --- | ---: | --- | --- |
| `gtsrb_v2_subset10.bin` | 10 | 默认快速演示 | 测试 Top-1 89.63% |
| `gtsrb_v4_semantic10.bin` | 10 | 语义均衡开发模型 | 测试 Top-1 92.63% |
| `gtsrb_v5_full43.bin` | 43 | 完整 GTSRB 展示模型 | 官方测试集 Top-1 95.26% |

43 类模型来自 `gtsrb_v5_enhanced_full_best.pt`，通过 `cppcnn_app_gpu export` 导出为项目自定义 `CPPCNN1` `.bin` 格式。最终 GUI 推理仍使用项目自己的 C++ CNN 结构加载 `.bin` 权重。

## 发布包内容

`cppCNN-Traffic-Sign-Studio-v2.0.0-windows-x64.zip` 包含：

- `cppcnn_gui.exe` Qt 桌面界面。
- `cppcnn_app.exe` CPU CLI。
- Qt 运行库、QML 模块和插件。
- 默认 10 类模型与完整 43 类模型。
- 多类别 GTSRB 演示图片。
- AI 使用记录 HTML。
- 使用说明与启动脚本。

`cppCNN-Codex-Report-Kit-v2.0.0.zip` 包含：

- 可运行应用副本。
- Codex 版本源码。
- 项目报告、设计说明、数据集说明、训练结果和模型报告。
- AI 使用记录 `codex_session_export.html`。
- 宝宝级资料包指南和代码阅读指南。

## 注意事项

- 完整 GTSRB 数据集不进入 Git，也不进入 Release ZIP。
- 模型权重不进入 Git 历史，只通过 Release 资产分发。
- 教师便携包默认不包含 LibTorch/CUDA 运行库；普通演示不需要配置 GPU 环境。
- `claude/` 目录是对照原型，不包含在 Codex Report Kit 中。
