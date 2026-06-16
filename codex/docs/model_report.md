# 模型综合报告

> **生成日期：** 2026-06-16
> **项目：** cppCNN Traffic Sign Recognition (GTSRB)

---

## 一、模型文件清单

### 1.1 Enhanced 架构 (~1.29M 参数)

| 文件名 | 大小 | 说明 |
| --- | --- | --- |
| gtsrb_v5_enhanced_full_best.pt | 5,156 KB | 当前最佳 - Phase C, 97.5% val, epoch 30 |
| gtsrb_v5_enhanced_full.pt | 5,156 KB | Phase C 最终模型, epoch 80 |
| gtsrb_v4_enhanced_aug_best.pt | 5,156 KB | Phase B4 best, 95.12% val |
| gtsrb_v4_enhanced_aug.pt | 5,156 KB | Phase B4 最终 |
| gtsrb_v3_enhanced_plain_best.pt | 5,156 KB | Phase B3 best, 94.53% val |
| gtsrb_v3_enhanced_plain.pt | 5,156 KB | Phase B3 最终 |

### 1.2 LeNet 架构 (~56K 参数)

| 文件名 | 大小 | 说明 |
| --- | --- | --- |
| gtsrb_v1_lenet_plain_best.pt | 227 KB | LeNet baseline, 89.69% val |
| gtsrb_v1_lenet_plain.pt | 227 KB | Phase B1 最终 |
| gtsrb_v2_lenet_aug_best.pt | 227 KB | LeNet + aug, 89.90% val |
| gtsrb_v2_lenet_aug.pt | 227 KB | Phase B2 最终 |

### 1.3 cppCNN 原生格式 (.bin)

| 文件名 | 大小 | 类数 | 说明 |
| --- | --- | --- | --- |
| gtsrb_v2_subset10.bin | 204 KB | 10 类 | UI 默认加载模型 |
| gtsrb_v4_semantic10.bin | 204 KB | 10 类 | 语义子集版本 |

---

## 二、训练集统计 (GTSRB 完整 43 类)

### 基本数据

| 指标 | 值 |
| --- | --- |
| 总训练图像 | **39,209** 张 |
| 测试图像 | **12,630** 张 |
| 类别数 | **43** |
| 图像格式 | PPM (RGB) |
| 原始尺寸 | 15x15 ~ 250x250 不等 |
| CNN 输入 | 32x32 RGB |
| 数据集大小 | 920 MB |
| 来源 | GTSRB (German Traffic Sign Recognition Benchmark) |

### 类别分布 (极不均衡)

| 范围 | 类别数 | 示例类 |
| --- | --- | --- |
| **极少** (210-300) | **10 类** | 类00 (限速20): 210; 类19 (危险左弯): 210; 类27/32/37/39-42 |
| 少量 (330-600) | **10 类** | 类16: 420; 类21: 330; 类26: 600 |
| 中等 (630-1500) | **12 类** | 类15: 630; 类25: 1500 |
| 充足 (1500-2250) | **11 类** | 类01 (限速30): 2220; 类02 (限速50): 2250 |

**不均衡倍率：10.7x** (最多 2250 vs 最少 210)

### 训练策略补偿

| 策略 | 描述 |
| --- | --- |
| Class-balanced sampling | 每类 ~1,800 样本 → ~77K samples/epoch |
| Data augmentation | 旋转 ±15 度, 平移 ±4px, 缩放 ±10%, 亮度/对比度, 高斯噪声 |

---

## 三、Phase C 最佳模型评估

### 性能指标

| 指标 | 值 | 评价 |
| --- | --- | --- |
| **Test Accuracy** | **95.26%** | 主指标 |
| Best Val Accuracy | 97.50% (epoch 30) | 验证集巅峰 |
| Mean Class Accuracy | 93.54% | 多数类表现优秀 |
| **Min Class Accuracy** | **48.33%** | 瓶颈在样本极少类 |
| 总参数量 | 1,286,251 | 适中 |
| 模型大小 | 5,156 KB | ~5 MB |
| 总训练时间 | 88.8 分钟 | RTX 4060 |

### 训练过程

| Epoch 范围 | LR | 训练损失 | 验证准确率 | 阶段 |
| --- | --- | --- | --- | --- |
| 1-24 | 0.01 | 3.38→0.10 | 23.76%→96.33% | 快速上升 |
| 25-48 | 0.001 | 0.07→0.06 | 97.29%→97.32% | 最佳稳定区 |
| 49-71 | 0.0001 | 0.055 | 97.19% | 微调区 |
| 72-80 | 0.00001 | 0.055 | 97.19% | 冻结区 |

### 瓶颈类别分析

| 类别 | 含义 | 训练样本 | 推断表现 |
| --- | --- | --- | --- |
| 00 | 限速20 km/h | 210 | ~50% |
| 19 | 危险左弯 | 210 | ~50% |
| 其他 <300 样本类 | 各类警告/限制 | 210-300 | ~50-70% |

---

## 四、模型大小对比

| 架构 | 参数 | 磁盘大小 | 架构概要 |
| --- | --- | --- | --- |
| **LeNet** | 56,195 | 227 KB | Conv(6) → Pool → Conv(16) → Pool → FC(120→84→43) |
| **Enhanced** | 1,286,251 | 5,156 KB | Conv(32) → Pool → Conv(64) → Pool → Conv(128) → Pool → FC(2048→512→256→43) |
| 差距 | **22.9x** | **22.7x** | Enhanced 更深更宽 + Dropout |

---

## 五、模型格式说明

| 格式 | 用途 | 加载方式 |
| --- | --- | --- |
| .pt | GPU 训练 / LibTorch 后端 | cppcnn_app_gpu.exe 使用 PyTorch 序列化 |
| .bin | GUI / 纯 C++ 推理 | cppcnn_gui.exe 使用 CNN 自定义序列化 |
| 转换关系 | 两者互不兼容 | 需要额外转换步骤 |

---

## 六、UI 模型切换能力

### 当前限制

**无法切换：** 当前 UI (cppcnn_gui.exe) 不能加载任何 .pt 格式模型 (包括 Phase C 的 Enhanced 最佳模型)。

原因：

1. GUI 使用纯 C++ CNN 后端，不支持 PyTorch/LibTorch 序列化格式
2. 文件对话框过滤器限制为 *.bin 文件
3. InferenceEngine 使用 cppcnn::CNN::forward()，而非 LibTorch 推理路径

### 能做什么

- 通过 Ctrl+M 或菜单选择任意 .bin 模型文件进行切换
- 当前可用：gtsrb_v2_subset10.bin (10类)、gtsrb_v4_semantic10.bin (10类)、**gtsrb_v5_full43.bin (43类, Enhanced)**
- 自动加载配套的 labels.txt
- 界面上实时显示类数、架构类型、参数信息

### 当前已完成

1. **导出到 .bin (已完成)**：通过 cppcnn_app_gpu export 命令将 gtsrb_v5_enhanced_full_best.pt 成功导出为 gtsrb_v5_full43.bin
   * 格式：Magic(CPPCNN1\0) + Version 2 + 43 classes + Enhanced + 6 层
   * 大小：4,906,800 字节 (4.9 MB)
   * 标签：gtsrb_v5_full43.labels.txt (43 行，GTSRB 标准命名)
   * GUI 通过 Ctrl+M 加载此 .bin 文件即可使用 Phase C 模型
2. **GUI 集成 LibTorch**：未进行
3. **CLI 推理**：始终可用，cppcnn_app_gpu predict