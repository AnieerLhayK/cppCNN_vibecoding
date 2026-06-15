# GTSRB 43 类高精度模型 — 数据审计与训练计划

> 生成日期：2026-06-15  
> 仓库：cppCNN_vibecoding  
> 分支：`claude_release`

---

## 1. 环境确认

| 项 | 状态 |
| --- | --- |
| Git 分支 | claude_release（阶段 A/B/C/D 开发分支） |
| 编译工具链 | MSVC 19.44 + Visual Studio 17 2022 x64 |
| CMake | 3.20+ |
| 构建目录 | D:\AI\data\codex\cache\staging\cppcnn-training-build |
| OpenCV | 未安装（PPM 加载器正常工作） |
| CTest（测试套件） | 全部通过（cppcnn_basic_tests） |
| 43 类训练管线 | 已验证（2 epoch, 1 样本/类, 可 save/load） |

## 2. 数据集审计

### 2.1 目录结构

codex/datasets/GTSRB/
+-- GT-final_test.csv（12630 条）
+-- GTSRB/
    +-- Final_Training/Images/00000..00042/（43 个目录）
    |   +-- GT-000xx.csv（每类一个 CSV）
    +-- Final_Test/Images/（12630 张 PPM）

### 2.2 训练集分布（43 类，共 39,209 张）

| Cls | GTSRB ID | 训练 | 测试 | 含义 |
| --: | :---: | ---: | ---: | :--- |

| 0 | 0 | 210 | 60 | 20 km/h |
| 1 | 1 | 2,220 | 720 | 30 km/h |
| 2 | 2 | 2,250 | 750 | 50 km/h |
| 3 | 3 | 1,410 | 450 | 60 km/h |
| 4 | 4 | 1,980 | 660 | 70 km/h |
| 5 | 5 | 1,860 | 630 | 80 km/h |
| 6 | 6 | 420 | 150 | End of 80 km/h |
| 7 | 7 | 1,440 | 450 | 100 km/h |
| 8 | 8 | 1,410 | 450 | 120 km/h |
| 9 | 9 | 1,470 | 480 | No passing |
| 10 | 10 | 2,010 | 660 | No passing over 3.5t |
| 11 | 11 | 1,320 | 420 | Right-of-way |
| 12 | 12 | 2,100 | 690 | Priority road |
| 13 | 13 | 2,160 | 720 | Yield |
| 14 | 14 | 780 | 270 | Stop |
| 15 | 15 | 630 | 210 | No vehicles |
| 16 | 16 | 420 | 150 | Vehicles over 3.5t prohibited |
| 17 | 17 | 1,110 | 360 | No entry |
| 18 | 18 | 1,200 | 390 | General caution |
| 19 | 19 | 210 | 60 | Dangerous curve left |
| 20 | 20 | 360 | 90 | Dangerous curve right |
| 21 | 21 | 330 | 90 | Double curve |
| 22 | 22 | 390 | 120 | Bumpy road |
| 23 | 23 | 510 | 150 | Slippery road |
| 24 | 24 | 270 | 90 | Road narrows right |
| 25 | 25 | 1,500 | 480 | Road work |
| 26 | 26 | 600 | 180 | Traffic signals |
| 27 | 27 | 240 | 60 | Pedestrians |
| 28 | 28 | 540 | 150 | Children crossing |
| 29 | 29 | 270 | 90 | Bicycles crossing |
| 30 | 30 | 450 | 150 | Beware ice/snow |
| 31 | 31 | 780 | 270 | Wild animals |
| 32 | 32 | 240 | 60 | End speed+passing limits |
| 33 | 33 | 689 | 210 | Turn right ahead |
| 34 | 34 | 420 | 120 | Turn left ahead |
| 35 | 35 | 1,200 | 390 | Ahead only |
| 36 | 36 | 390 | 120 | Go straight or right |
| 37 | 37 | 210 | 60 | Go straight or left |
| 38 | 38 | 2,070 | 690 | Keep right |
| 39 | 39 | 300 | 90 | Keep left |
| 40 | 40 | 360 | 90 | Roundabout mandatory |
| 41 | 41 | 240 | 60 | End of no passing |
| 42 | 42 | 240 | 90 | End no passing over 3.5t |

**关键统计：** 训练总数 39,209，测试总数 12,630，最小类 210，最大类 2,250，不均衡比 10.7x

### 2.3 数据完整性

| 检查项 | 结果 |
| --- | --- |
| 损坏图片 | 无（所有文件 > 1KB） |
| 空类别 | 无 |
| 标签映射 | 完整（assets/labels.txt 43 行） |
| 训练 CSV 与 PPM | 全部一致 |
| 测试 CSV 与图片 | 12630 条目 = 12630 图片 |

### 2.4 Track 分布

GTSRB 图片命名 XXXXX_YYYYY.ppm，XXXXX = track ID。同一 track 连续帧高度相似，不可分入 train/val 两方。
每类 track 数 7-74，每 track 约 30 帧。

## 3. 现有实现审计

### 3.1 网络架构（LeNet）

Input (3x32x32) 
  -> Conv2d(3->6, k=5, s=1, p=0) [456 params] -> ReLU -> MaxPool(2x2) [28->14]
  -> Conv2d(6->16, k=5, s=1, p=0) [2,416 params] -> ReLU -> MaxPool(2x2) [10->5]
  -> Flatten (400) -> FC(400->120) [48,120 params] -> ReLU -> FC(120->43) [5,203 params]
  -> Softmax -> Output (43)

**总参数量：56,195（约 220 KB float32）**

### 3.2 功能现状

| 功能 | 状态 |
| --- | --- |
| 前向/反向传播 | 工作正常 |
| Mini-batch SGD + Weight Decay | 工作正常 |
| He 初始化 | 正常 |
| 交叉熵 + Softmax | 正常（含 max-subtract 稳定化） |
| 模型序列化 | 正常（magic + version + shape check） |
| 43 类 DataLoader | 已验证 |
| Momentum SGD | 缺失 |
| 学习率衰减 | 缺失 |
| 验证集划分（track-based） | 缺失 |
| 最佳 checkpoint | 缺失 |
| 断点续训 | 缺失 |
| 每类准确率 / 混淆矩阵 | 缺失 |
| CSV 历史 | 缺失 |
| 数据增强 | 缺失 |
| 类别均衡采样 | 缺失 |

### 3.3 现有网络局限性

1. 卷积层太浅（仅 2 层）- 43 类复杂标志区分力不足
2. 卷积核太少（6->16）- 特征维度太低
3. 无 Dropout - 小类（210 样本）易过拟合
4. SGD 无动量 - 收敛慢
5. 无数据增强 - 小类泛化差

## 4. 改进网络设计

Input (3x32x32)
  -> Conv2d(3->32, k=5, s=1, p=2) -> ReLU -> MaxPool(2x2) [32->16]
  -> Conv2d(32->64, k=3, s=1, p=1) -> ReLU -> MaxPool(2x2) [16->8]
  -> Conv2d(64->128, k=3, s=1, p=1) -> ReLU -> MaxPool(2x2) [8->4]
  -> Flatten (2048) -> FC(2048->512) -> ReLU + Dropout(0.5)
  -> FC(512->256) -> ReLU + Dropout(0.5) -> FC(256->43) -> Softmax

**参数量：约 1,286,000（约 5 MB float32）**

## 5. 超参数计划

| 参数 | 原 LeNet | 增强网络 |
| --- | --- | --- |
| Batch size | 64 | 64 |
| Initial LR | 0.01 | 0.01 |
| Momentum | 0.9 | 0.9 |
| Weight decay | 1e-4 | 5e-4 |
| LR decay | StepLR(30, 0.1) | StepLR(20, 0.1) |
| Max epochs | 80 | 80 |
| Early stopping | patience=10 | patience=10 |
| Validation | 20% track-based | 20% track-based |

**数据增强：** 旋转 +/-15 度, 平移 +/-4px, 缩放 +/-10%, 亮度 +/-20%, 对比度 +/-20%, 高斯噪声 sigma<=0.05（禁止水平/垂直翻转）

## 6. 预计资源

| 项 | 原 LeNet | 增强网络 |
| --- | --- | --- |
| 模型 | ~220 KB | ~5 MB |
| 训练内存 | ~10 MB | ~100 MB |
| 1 epoch | ~90 秒 | ~15-30 分钟 |
| 80 epoch | ~2 小时 | ~20-40 小时 |

## 7. 分阶段计划

### 阶段 A：训练管线增强
- Momentum SGD, LR 调度器, 验证集, checkpoint, 断点续训
- 每类准确率, 混淆矩阵, CSV 历史
- 数据增强, 类别均衡采样
- 扩展新网络架构

### 阶段 B：基线实验（5-10 epoch）
- B1: 原 LeNet 无增强 / B2: 原 LeNet 全增强
- B3: 增强网络无增强 / B4: 增强网络全增强

### 阶段 C：正式训练（60-80 epoch, early stopping）

### 阶段 D：最终重训与发布（全 39,209 训练 + 测试集评估）

## 8. 预期目标

| 指标 | 最低 | 目标 | 挑战 |
| --- | ---: | ---: | ---: |
| 测试集 Top-1 | >= 96% | >= 98% | >= 99% |
| 单类最低准确率 | >= 90% | >= 90% | >= 95% |

## 9. 已完成步骤

1. 创建分支 feat/gtsrb43-training
2. 审查编译现有代码（CTest 全部通过）
3. 验证 43 类数据管线
4. 完成数据审计报告
5. 准备开始阶段 A 实现

## 11. 阶段 A 验证结果

> 验证时间：2026-06-15

### 测试套件
- 核心与 GUI 测试全部通过（Tensor, CNN, Dropout, LR Scheduler, Augmenter, 混淆矩阵, Track 分割等）
- 构建：Release, MSVC 19.44, 无警告

### 高级管线验证（2 epoch, LeNet, 43 类）
| 指标 | 值 |
| --- | --- |
| 训练样本 | 31,349（80%, track-based） |
| 验证样本 | 7,860（20%, track-based） |
| Epoch 1 val acc | 83.09% |
| Epoch 2 val acc | 88.32% |
| 平均类准确率 | 85.16% |
| 训练耗时 | 801 秒（含完整 39K 图片加载） |
| CSV 历史 | 正常输出 |
| Checkpoint | 模型 + 优化器状态 |
| 续训 | 支持（.opt 文件） |

### 阶段 A 交付物
| 文件 | 说明 |
| --- | --- |
| src/cnn/Augmenter.h/cpp | 数据增强（旋转/平移/缩放/亮度/对比度/噪声） |
| src/cnn/DropoutLayer.h/cpp | Dropout 层（train/eval 模式切换） |
| src/cnn/LRScheduler.h/cpp | 学习率调度（StepLR/Cosine/Warmup） |
| src/cnn/Loss.h/cpp | PerClassAccuracy + ConfusionMatrix |
| src/cnn/Layer.h | 基类扩展（动量、优化器状态序列化） |
| src/cnn/ConvLayer.h/cpp | 动量 velocity 缓存 + updateWithMomentum |
| src/cnn/FCLayer.h/cpp | 动量 velocity 缓存 + updateWithMomentum |
| src/cnn/CNN.h/cpp | Enhanced 架构 + setTraining + 优化器序列化 |
| src/data/DataLoader.h/cpp | Track 解析 + splitByTrack |
| src/cnn/Trainer.h/cpp | 验证集、checkpoint、CSV 历史、类均衡采样 |
| src/app/App.h/cpp | train-advanced CLI |
| CMakeLists.txt | 新增源文件 |
| tests/test_basic.cpp | 16 个测试 |
