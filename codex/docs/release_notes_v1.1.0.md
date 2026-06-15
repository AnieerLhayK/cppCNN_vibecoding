# cppCNN Traffic Sign Studio v1.1.0

本版本发布 Codex 纯 C++ 交通标志识别主实现，并将 Claude Python/PyTorch 车标原型合入同一仓库用于技术路线比较。

## Codex 主实现

- 新增语义均衡 10 类数据集与独立模型工作流，实测 Top-1 为 92.63%。
- 开发者训练脚本支持命名参数、自动评估、日志、元数据和旧模型归档。
- GUI 可安全切换模型，并优先读取模型同名 `.labels.txt`，避免类别映射错位。
- 教师演示包继续使用经过逐张验证的稳定 10 类模型和五种不同类别演示图。

## Claude 对照原型

- 合入配置驱动的 PyTorch 车标 CNN、数据划分、训练、评估和单图推理源码。
- 补充数据集、checkpoint、环境与运行边界说明。
- 新增 Windows Python 3.11 CI；当前 23 项单元测试通过。
- 不包含车标数据集、训练权重或未经验证的准确率。

## 下载

下载 `cppCNN-Traffic-Sign-Studio-v1.1.0-windows-x64.zip`，完整解压后双击 `run_demo.bat`。SHA-256 文件可用于校验下载完整性。

课程报告协作者可下载 `cppCNN-Codex-Report-Kit-v1.1.0.zip`。该补充资料包只包含 Codex 纯 C++ 实现，提供同一便携程序、完整课程源码、报告材料、代码阅读指南和答辩指南，不包含 Claude/PyTorch 原型或完整数据集。
