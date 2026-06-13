# cppCNN Traffic Sign Studio v1.0.1

这是一个聚焦演示质量的补丁版本。此前五张内置图片全部是连续限速牌，不能充分展示交通标志分类的类别广度。

## 主要变化

- 演示图片区改为 `Class showcase`。
- 五张图片现在对应五个不同模型类别：
  - 30 km/h speed limit
  - 100 km/h speed limit
  - No passing
  - No passing for vehicles over 3.5 t
  - Right-of-way at the next intersection
- 所有样本均来自 GTSRB 测试集，并经过当前模型逐张预测验证。
- 保留点击样本后自动识别、Top-1、置信度、Top-3 和推理耗时展示。

## 下载与运行

1. 下载 `cppCNN-Traffic-Sign-Studio-v1.0.1-windows-x64.zip`。
2. 完整解压 ZIP。
3. 双击 `run_demo.bat`。
4. 点击底部任意类别样本运行识别。

便携包包含 Qt 运行库、10 类模型权重、标签和演示图片，无需重新训练。

## 仍有的限制

当前 10 类训练子集中仍有 7 个限速类别。后续模型版本应改为语义更均衡的类别组合，例如限速、停车、让行、施工、行人、儿童、禁行、强制方向等。
