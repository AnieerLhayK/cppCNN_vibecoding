# 模型文件

训练权重保存在此目录，但所有 `.bin`、`.model` 和 `.weights` 文件均被 Git 忽略。

默认开发模型：

```text
models/gtsrb_subset10.bin
```

## 自定义格式

模型使用 `CPPCNN1` 二进制格式：

| 字段 | 类型 |
| --- | --- |
| 魔数 `CPPCNN1\0` | 8 字节 |
| 格式版本 | `uint32` |
| 类别数 | `uint64` |
| 可训练层数 | `uint32` |
| 层类型 | `uint32`，卷积为 1，全连接为 2 |
| 权重数量和数据 | `uint64` + 连续 `float32` |
| 偏置数量和数据 | `uint64` + 连续 `float32` |

加载时会检查魔数、版本、类别数、层顺序、参数数量和数据完整性。不要手工编辑权重文件。

## 训练

```powershell
.\build\Release\cppcnn_app.exe train `
  datasets\GTSRB_subset `
  models\gtsrb_subset10.bin 10 5 0
```

模型缺失时，CLI 会给出训练提示；GUI 会安全打开、显示“Model missing”并允许选择模型，不会崩溃或使用随机权重预测。
