# 模型版本声明

本项目同时保留纯 C++ 模型和 LibTorch GPU 训练 checkpoint。两类文件用途不同：

- `.bin`：项目自定义格式，供纯 C++ GUI 和 CPU CLI 推理。
- `.pt`：LibTorch checkpoint，供开发者继续训练、评估和复现实验。

## 1. 纯 C++ 模型

| 文件 | 架构 | 类别 | 用途 | 测试 Top-1 |
| --- | --- | ---: | --- | ---: |
| `gtsrb_v2_subset10.bin` | LeNet | 10 | 默认 GUI 模型，原开发子集 | 89.63% |
| `gtsrb_v4_semantic10.bin` | LeNet | 10 | 语义更均衡的 10 类实验 | 92.63% |
| `gtsrb_v5_full43.bin` | Enhanced | 43 | 正式 GUI/CLI 完整类别模型 | 95.26% |

每个 `.bin` 应与同名 `.labels.txt` 配套使用。

## 2. GPU 基线与正式训练模型

| 阶段 | 文件 | 架构 | 数据增强 | Epoch | Val | Test |
| --- | --- | --- | --- | ---: | ---: | ---: |
| B1 | `gtsrb_v1_lenet_plain*.pt` | LeNet | 否 | 5 | 89.69% | 87.99% |
| B2 | `gtsrb_v2_lenet_aug*.pt` | LeNet | 是 | 5 | 89.90% | 87.01% |
| B3 | `gtsrb_v3_enhanced_plain*.pt` | Enhanced | 否 | 5 | 94.53% | 93.34% |
| B4 | `gtsrb_v4_enhanced_aug*.pt` | Enhanced | 是 | 5 | 95.12% | 92.69% |
| C | `gtsrb_v5_enhanced_full_best.pt` | Enhanced | 是 | 80 训练中最佳 | 97.50% | 95.26% |
| C | `gtsrb_v5_enhanced_full.pt` | Enhanced | 是 | 80 最终 | 97.20% | 以正式评估记录为准 |

带 `_best` 的文件是验证集最优 checkpoint；不带 `_best` 的文件是该次训练最后一个 epoch 的状态。

## 3. 正式模型说明

正式发布模型为：

```text
gtsrb_v5_full43.bin
```

它由 Enhanced 43 类模型导出，包含 1,286,251 个参数。正式训练使用 39,209 张训练图、80 epoch、数据增强和类别均衡采样，在 GTSRB 官方 12,630 张测试图上达到 95.26% Top-1。

## 4. 版本号说明

`v1` 至 `v5` 在不同实验系列中表达训练迭代阶段。纯 C++ 10 类系列与 GPU 43 类系列曾并行推进，因此不能只依据文件名中的数字比较模型优劣；应同时查看架构、类别数、标签文件和本表指标。

模型二进制均附带于教师提交包，但按机器学习项目管理规范不进入 Git 历史。
