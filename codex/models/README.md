# Models

训练生成的模型文件默认保存在此目录，但二进制模型不会提交到 Git。

模型采用项目自定义的 `CPPCNN1` 二进制格式，包含版本、类别数、层类型、权重和偏置。示例：

```powershell
.\build\Release\cppcnn_app.exe train `
  D:\datasets\gtsrb\train models\gtsrb10.bin 10 5 0
```

加载时会检查类别数、层顺序和参数数量。不要手工编辑模型文件。
