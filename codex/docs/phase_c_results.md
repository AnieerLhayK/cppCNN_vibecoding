# Phase C 正式训练结果报告

> 日期：2026-06-16  
> 架构：Enhanced，3 Conv + 3 FC + 双 Dropout  
> 后端：LibTorch GPU，NVIDIA RTX 4060 Laptop，CUDA 13.0，LibTorch 2.12.0  

## 1. 实验配置

| 参数 | 值 |
| --- | --- |
| 数据集 | GTSRB 43 类 |
| 训练图像 | 39,209 |
| 官方测试图像 | 12,630 |
| 输入尺寸 | 32 x 32 RGB |
| 模型架构 | Enhanced，1,286,251 参数 |
| Epochs | 80 |
| Batch size | 64 |
| Initial LR | 0.01 |
| LR 调度 | StepLR，每 24 epoch 衰减 0.1 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| 数据增强 | 旋转、平移、缩放、亮度、对比度、高斯噪声 |
| 类别均衡 | class-balanced sampling |
| 验证集 | 20% track-aware split |
| 最佳 checkpoint | `gtsrb_v5_enhanced_full_best.pt` |

## 2. 最终结果

| 指标 | 值 |
| --- | ---: |
| 官方测试集 Top-1 | **95.26%** |
| 最佳验证集准确率 | **97.50%**，epoch 30 |
| 最终验证集准确率 | 97.20% |
| 最终训练准确率 | 98.42% |
| 测试集 mean class accuracy | 93.54% |
| 测试集 min class accuracy | 48.33% |
| 总训练时间 | 5,330.9s，约 88.8 分钟 |

## 3. 与 Phase B 对比

| 实验 | 架构 | Epoch | Val | Test | 说明 |
| --- | --- | ---: | ---: | ---: | --- |
| B1 | LeNet no-aug | 5 | 89.69% | 87.99% | 原始基线 |
| B2 | LeNet aug | 5 | 89.90% | 87.01% | 增强后验证略升 |
| B3 | Enhanced no-aug | 5 | 94.53% | 93.34% | 架构提升明显 |
| B4 | Enhanced aug | 5 | 95.12% | 92.69% | Phase C 选择基础 |
| C | Enhanced aug | 80 | **97.50%** | **95.26%** | 正式模型 |

## 4. 训练过程观察

| Epoch 范围 | LR | 现象 |
| --- | --- | --- |
| 1-24 | 0.01 | 快速上升到约 96% 验证准确率 |
| 25-48 | 0.001 | 达到最佳验证准确率，模型趋于稳定 |
| 49-71 | 0.0001 | 小幅微调，收益有限 |
| 72-80 | 0.00001 | 基本冻结，继续训练收益很小 |

关键节点：

- Epoch 1 受 CUDA 初始化和缓存影响，耗时明显偏高。
- Epoch 30 达到最佳验证准确率 97.50%。
- 80 epoch 后验证集没有继续明显突破，说明后续可以考虑早停。

## 5. 分析

Enhanced 架构在 43 类 GTSRB 上达到 95.26% 官方测试集准确率，相比 LeNet 的 87-89% 有明显提升。主要贡献来自更高容量的 3 层卷积结构、Dropout、数据增强和类别均衡采样。

当前瓶颈主要在极少样本类别。例如 class 00、class 19 等类别训练样本只有约 210 张，min class accuracy 仍较低。后续若继续提升，可以优先考虑：

- BatchNorm。
- 更大输入尺寸，如 48 x 48。
- 对少样本类别做更强的数据增强或重采样。
- 更合理的早停策略，避免无收益长训。
- 多 checkpoint 集成或投票。

## 6. 模型产物

| 文件 | 说明 |
| --- | --- |
| `gtsrb_v5_enhanced_full_best.pt` | 最佳 LibTorch checkpoint，epoch 30 |
| `gtsrb_v5_enhanced_full.pt` | 80 epoch 最终 LibTorch 模型 |
| `gtsrb_v5_full43.bin` | 导出后的纯 C++ GUI 可加载模型 |
| `gtsrb_v5_full43.labels.txt` | 43 类标签 |

`.pt` 文件用于开发者训练与复现实验；`.bin` 文件用于最终 GUI 和 CPU CLI 推理。
