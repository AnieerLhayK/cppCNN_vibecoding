# GTSRB Semantic10 Model Card

## 用途

`gtsrb_semantic10.bin` 是纯 C++ LeNet 风格 CNN 的语义均衡开发模型，用于验证比原限速牌集中子集更广的交通标志分类能力。

该模型不会替换 `gtsrb_subset10.bin`。当前教师演示 Release 仍默认加载原模型。

## 类别

输出索引顺序由 `gtsrb_semantic10.labels.txt` 定义：

1. 50 km/h speed limit
2. No passing
3. Right-of-way at the next intersection
4. Yield
5. Stop
6. No entry
7. General caution
8. Road work
9. Children crossing
10. Keep right

对应 GTSRB 原始 ID：

```text
2,9,11,13,14,17,18,25,28,38
```

## 训练配置

| 参数 | 值 |
| --- | ---: |
| 训练图片 | 5,000 |
| 每类训练图片 | 500 |
| 官方测试图片 | 4,710 |
| Epoch | 5 |
| Batch size | 16 |
| Learning rate | 0.01 |
| Weight decay | 0.0001 |
| Seed | 42 |

## 结果

| 指标 | 值 |
| --- | ---: |
| 最终训练损失 | 0.1164 |
| 最终训练准确率 | 97.96% |
| 测试损失 | 0.2851 |
| 测试 Top-1 | 92.63% |

模型 SHA-256：

```text
2fc5ee6cb30b53aab48461c8c7df77eb369b13ef9fc416d79fcdac2fbe99e241
```

## 使用

在 GUI 中手动选择 `gtsrb_semantic10.bin`。程序会自动读取同目录的 `gtsrb_semantic10.labels.txt`。

CLI：

```powershell
.\build\Release\cppcnn_app.exe evaluate `
  datasets\GTSRB_semantic10 `
  models\gtsrb_semantic10.bin 0
```

权重通过 GitHub Release 模型资产分发，不进入 Git。
