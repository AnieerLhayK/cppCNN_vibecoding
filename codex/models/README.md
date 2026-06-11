# Models

训练生成的模型文件默认保存在此目录，但二进制模型不会提交到 Git。

模型采用项目自定义的 `CPPCNN1` 二进制格式，包含版本、类别数、层类型、权重和偏置。示例：

```powershell
.\build\Release\cppcnn_app.exe train `
  D:\datasets\gtsrb\train models\gtsrb10.bin 10 5 0
```

加载时会检查类别数、层顺序和参数数量。不要手工编辑模型文件。

## 文件格式

按写入顺序：

| 字段 | 类型 |
| --- | --- |
| 魔数 `CPPCNN1\0` | 8 字节 |
| 格式版本 | `uint32` |
| 类别数量 | `uint64` |
| 可训练层数量 | `uint32` |
| 层类型 | `uint32`，卷积为 1，全连接为 2 |
| 权重数量及数据 | `uint64` + 连续 `float32` |
| 偏置数量及数据 | `uint64` + 连续 `float32` |

当前格式版本为 1，参数使用运行机器的本地二进制字节序，主要面向本项目在 Windows/MSVC 环境下保存和恢复。

## Git 管理

本目录的 `.gitignore` 只允许提交：

- `.gitignore`
- `README.md`

所有模型权重和二进制文件均留在本地。程序启动时会检查默认模型 `models/gtsrb10.bin`；模型缺失时会提示训练命令，不会把底层文件打开错误直接暴露给用户。
