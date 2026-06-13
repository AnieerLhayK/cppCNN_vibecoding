# PyTorch 模型文件

训练脚本默认将 checkpoint 写入：

```text
claude/models/checkpoints/
├── best_model.pth
└── latest_model.pth
```

`.pth` 文件由 `torch.save` 生成，包含：

- 当前 epoch 和验证集准确率；
- `model_state_dict`；
- `optimizer_state_dict`；
- 模型配置；
- 与输出索引对应的 `class_names`。

模型权重、训练日志和派生产物不会进入 Git。单图推理默认读取 `best_model.pth`：

```powershell
python claude/scripts/predict.py `
  --image path\to\logo.jpg `
  --checkpoint claude\models\checkpoints\best_model.pth
```

checkpoint 依赖 PyTorch，不能由 `codex/` 的纯 C++ 程序加载。加载不可信来源的 PyTorch checkpoint 可能执行恶意序列化内容，只使用自己训练或可信发布方提供的文件。
