# cppCNN Vibe Coding

本仓库用于比较不同 AI 平台或协作方式如何完成“使用纯 C++ 实现 CNN”的工程任务。每个一级目录代表一种独立实现：

| 目录 | 实现 | 状态 |
| --- | --- | --- |
| [`codex/`](codex/) | Codex 协作完成的 GTSRB 交通标志识别系统 | `v1.0.0`，可训练、可推理、含 Qt GUI |
| `claude/` | 预留给 Claude 或其他工作流 | 待实现 |

## 当前发布

- [下载最新 GitHub Release](https://github.com/f32797653-beep/cppCNN_vibecoding/releases/latest)
- [查看 Codex 实现说明](codex/README.md)
- [查看项目报告](codex/docs/project_report.md)
- [查看版本变化](codex/CHANGELOG.md)

GitHub Release 的 Windows x64 ZIP 包含可执行程序、Qt 运行库、训练好的 10 类模型、标签和演示图片。源码仓库本身不提交数据集或模型权重。

共同约定：

- CNN 数值核心使用 C++ 实现，不调用深度学习框架。
- 数据集、训练权重和临时构建产物不进入 Git。
- 每种实现独立维护源码、构建配置、测试、文档和实验记录。
- 稳定阶段使用清晰的 Git commit，并保持主分支可构建。

Codex 版本的完整说明、数据集准备、编译和运行方式见 [`codex/README.md`](codex/README.md)。
