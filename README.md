# cppCNN Vibe Coding

本仓库比较不同 AI 协作平台完成 CNN 项目的实现方式。各一级目录是相互独立的技术路线，不共享数据集、模型格式或运行时依赖。

| 目录 | 技术路线 | 定位 | 状态 |
| --- | --- | --- | --- |
| [`codex/`](codex/) | C++17、手写 CNN、Qt Quick | 课程主交付：GTSRB 交通标志识别 | `v1.1.0`，可训练、评估、推理和直接演示 |
| [`claude/`](claude/) | Python、PyTorch、控制台 | 对照原型：车标分类 | 源码与 23 项单元测试完整，尚无数据集和训练权重 |

## 当前发布

- [下载最新 GitHub Release](https://github.com/f32797653-beep/cppCNN_vibecoding/releases/latest)
- [Codex 主实现说明](codex/README.md)
- [Claude 对照原型说明](claude/README.md)
- [Codex 项目报告](codex/docs/project_report.md)
- [版本变化](codex/CHANGELOG.md)

GitHub Release 的 Windows x64 ZIP 是 `codex/` 主实现的教师演示包，包含可执行程序、Qt 运行库、训练好的 10 类模型、标签和多类别演示图片。`claude/` 随源码标签发布，不提供预训练权重或独立可执行包。

## 仓库约定

- 数据集、训练权重、缓存和临时构建产物不进入 Git。
- 每种实现独立维护源码、配置、测试和文档。
- `codex/` 满足纯 C++ CNN 课程要求，不调用深度学习框架。
- `claude/` 使用 PyTorch，仅用于技术路线比较，不能替代纯 C++ 主交付。

## 快速验证

Codex：

```powershell
cmake -S codex -B codex/build -DCPPCNN_BUILD_GUI=OFF
cmake --build codex/build --config Release
ctest --test-dir codex/build -C Release --output-on-failure
```

Claude：

```powershell
python -m pip install -r claude/requirements.txt
python -m pytest -q claude/tests
```
