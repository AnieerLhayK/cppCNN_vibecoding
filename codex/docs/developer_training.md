# 开发者训练指南

## UI 与训练职责

Qt GUI 只负责加载模型和执行推理，不提供训练按钮。这是有意设计：

- 教师或普通用户直接运行 Release，不需要数据集或开发环境。
- 训练需要完整数据、较长 CPU 时间和明确的参数记录，应由开发者工具完成。
- GUI 和训练流程分离，可避免误操作覆盖可用权重。

## 推荐入口

在仓库根目录执行：

```powershell
.\codex\scripts\train_model.ps1
```

默认工作流会：

1. 构建 `cppcnn_app`、子集工具和核心测试。
2. 运行 CTest。
3. 从完整 GTSRB 生成语义均衡 10 类子集。
4. 使用纯 C++ CNN 训练 5 Epoch。
5. 在官方测试图片上评估。
6. 保存模型、训练日志、参数元数据和 SHA-256。

默认输出：

```text
codex/datasets/GTSRB_semantic10/
codex/models/gtsrb_v4_semantic10.bin
codex/models/gtsrb_v4_semantic10.labels.txt
codex/models/gtsrb_semantic10.training.log
codex/models/gtsrb_semantic10.metadata.json
```

权重、日志和元数据被 Git 忽略；轻量的同名标签文件可以进入 Git，确保模型输出索引有明确含义。

## 默认语义均衡类别

| GTSRB ID | 类别 |
| ---: | --- |
| 2 | 50 km/h speed limit |
| 9 | No passing |
| 11 | Right-of-way at the next intersection |
| 13 | Yield |
| 14 | Stop |
| 17 | No entry |
| 18 | General caution |
| 25 | Road work |
| 28 | Children crossing |
| 38 | Keep right |

训练集每类固定 500 张，共 5,000 张。测试集保留这些类别在官方测试集中的全部 4,710 张图片，因此测试分布不是人为均衡的。

## 常用参数

```powershell
.\codex\scripts\train_model.ps1 `
  -ClassIds 2,9,11,13,14,17,18,25,28,38 `
  -ImagesPerClass 500 `
  -Epochs 8 `
  -BatchSize 16 `
  -LearningRate 0.01 `
  -WeightDecay 0.0001 `
  -Seed 42 `
  -ModelPath models\gtsrb_semantic10_e8.bin
```

跳过重复构建：

```powershell
.\codex\scripts\train_model.ps1 -SkipBuild
```

使用已经生成的子集：

```powershell
.\codex\scripts\train_model.ps1 -SkipBuild -SkipSubset
```

若目标模型文件已经存在，脚本会把旧模型、标签、日志和元数据移动到：

```text
codex/models/archive/<模型名>.<时间戳>/
```

因此重复训练不会静默覆盖历史权重。

## 底层 CLI

需要直接控制底层程序时：

```powershell
.\build\Release\cppcnn_app.exe train `
  <dataset> <model> `
  <class_count> <epochs> <samples_per_class> `
  <batch_size> <learning_rate> <weight_decay> <seed>
```

示例：

```powershell
.\build\Release\cppcnn_app.exe train `
  datasets\GTSRB_semantic10 `
  models\gtsrb_v4_semantic10.bin `
  10 5 0 16 0.01 0.0001 42

.\build\Release\cppcnn_app.exe evaluate `
  datasets\GTSRB_semantic10 `
  models\gtsrb_v4_semantic10.bin 0
```

底层子集工具支持显式类别：

```powershell
.\build\Release\cppcnn_create_subset.exe `
  datasets\GTSRB datasets\GTSRB_semantic10 `
  10 500 "2,9,11,13,14,17,18,25,28,38"
```

## 模型切换

GUI 默认继续加载稳定的：

```text
models/gtsrb_v2_subset10.bin
```

新模型训练完成后，可以在 GUI 的 Settings/模型选择入口手动加载，也可以在确认质量后修改默认模型配置并生成新的 Release。GUI 会优先读取与模型同名的 `.labels.txt`，例如 `gtsrb_v4_semantic10.labels.txt`，避免不同类别映射之间发生标签错位。不要直接用未经评估的新模型替换稳定模型。

## 当前保留模型

| 模型 | 类别策略 | 训练图 | 测试图 | Top-1 |
| --- | --- | ---: | ---: | ---: |
| `gtsrb_v2_subset10.bin` | 7 个限速类别的原开发基线 | 10,000 | 5,670 | 89.63% |
| `gtsrb_v4_semantic10.bin` | 10 个语义分散类别 | 5,000 | 4,710 | 92.63% |

两个模型均为 10 类，但输出索引含义不同，不能交换标签文件。
