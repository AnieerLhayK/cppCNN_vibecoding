# 模型文件

训练权重保存在此目录，但所有 `.bin`、`.model` 和 `.weights` 文件均被 Git 忽略。

默认开发模型：

```text
models/gtsrb_v2_subset10.bin
```

语义均衡开发模型：

```text
models/gtsrb_v4_semantic10.bin
```

两个模型使用不同类别映射，必须配合各自同名的 `.labels.txt`。GUI 会优先查找模型同目录下的同名标签文件。训练日志与元数据分别使用 `.training.log` 和 `.metadata.json`，不进入 Git。

## 自定义格式

模型使用 `CPPCNN1` 二进制格式：

| 字段 | 类型 |
| --- | --- |
| 魔数 `CPPCNN1\0` | 8 字节 |
| 格式版本 | `uint32` |
| 类别数 | `uint64` |
| 网络架构 | `uint32`，LeNet 为 0，Enhanced 为 1（格式 v2） |
| 可训练层数 | `uint32` |
| 层类型 | `uint32`，卷积为 1，全连接为 2 |
| 权重数量和数据 | `uint64` + 连续 `float32` |
| 偏置数量和数据 | `uint64` + 连续 `float32` |

加载时会检查魔数、版本、类别数、网络架构、层顺序、参数数量和数据完整性。格式 v2 会保存网络架构；已有格式 v1 模型继续按 LeNet 兼容加载。不要手工编辑权重文件。

## 训练

推荐使用带命名参数、自动评估和归档功能的开发脚本：

```powershell
.\scripts\train_model.ps1
```

完整说明见 [`../docs/developer_training.md`](../docs/developer_training.md)。

底层 CLI：

```powershell
.\build\Release\cppcnn_app.exe train `
  datasets\GTSRB_subset `
  models\gtsrb_v2_subset10.bin 10 5 0 16 0.01 0.0001 42
```

模型缺失时，CLI 会给出训练提示；GUI 会安全打开、显示“Model missing”并允许选择模型，不会崩溃或使用随机权重预测。
