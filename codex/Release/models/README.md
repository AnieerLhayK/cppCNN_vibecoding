# Model Placeholder

本目录包含已训练的 GTSRB 交通标志识别模型：

```text
gtsrb_v2_subset10.bin    10 类子集模型 — 默认自动加载
gtsrb_v5_full43.bin      完整 43 类模型 — 通过 Ctrl+M 手动加载
```

每个 `.bin` 文件对应同名的 `.labels.txt` 配套标签文件，使用项目自定义 `CPPCNN1` 二进制格式。

`gtsrb_v5_full43.bin` 由 80 epoch Enhanced 模型导出，官方 GTSRB 测试集 Top-1 为 95.26%。默认仍加载 10 类模型，是为了打开程序后能立即识别内置演示图；需要展示完整 43 类时，在 GUI 中按 `Ctrl+M` 选择 `models/gtsrb_v5_full43.bin`。

本地 Release 包现已放入经过实际训练和测试的模型，不是空权重或随机参数。模型二进制文件仍被 Git 忽略，不会进入源码仓库。

模型文件属于生成产物，不提交到 Git。训练完成后，由发布负责人将模型复制到本目录，再交付完整演示包。
