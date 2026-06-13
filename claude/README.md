# Claude 车标 CNN 对照原型

这是 `cppCNN_vibecoding` 仓库中的 Python/PyTorch 对照实现，用于展示配置驱动的车标分类训练、评估和单图推理流程。它不是纯 C++ 实现，也不是当前课程主交付；主交付见 [`../codex/`](../codex/)。

## 当前状态

- 已实现数据扫描与训练/验证/测试集划分。
- 已实现可配置 CNN、数据增强、训练、断点保存、评估和 Top-K 推理。
- 已有 23 项单元测试。
- 未附带车标数据集、训练权重、实测准确率或 GUI。
- 程序在数据集或 checkpoint 缺失时会给出明确错误。

## 环境

推荐使用 Python 3.10-3.12 的独立虚拟环境。Python 3.13 与部分 PyTorch/NumPy 二进制组合可能不兼容。

```powershell
py -3.11 -m venv claude/.venv
.\claude\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r claude/requirements.txt
```

需要 GPU 训练时，应按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择与本机 CUDA 匹配的安装命令，再安装其余依赖。

## 快速开始

以下命令均从仓库根目录执行：

```powershell
# 将按品牌分目录的图片放入 claude/data/raw 后划分数据集
python claude/scripts/prepare_data.py

# 训练并在 claude/models/checkpoints 保存 checkpoint
python claude/scripts/train.py

# 测试集评估
python claude/scripts/evaluate.py

# 单图 Top-K 推理
python claude/scripts/predict.py --image path\to\logo.jpg

# 单元测试
python -m pytest -q claude/tests
```

训练参数可在 [`configs/training_config.json`](configs/training_config.json) 修改，也可用 `--epochs`、`--lr`、`--batch-size` 和 `--data-dir` 覆盖。网络结构位于 [`configs/model_config.json`](configs/model_config.json)。

## 项目结构

```text
claude/
├── configs/              # 模型结构与训练超参数
├── data/                 # 本地数据集和数据说明
├── models/               # 本地 checkpoint 和格式说明
├── scripts/              # 准备、训练、评估、推理模块
├── tests/                # Pytest 单元测试
├── .gitignore
└── requirements.txt
```

## 数据与模型

数据目录格式和划分方式见 [`data/README.md`](data/README.md)，checkpoint 内容及恢复方式见 [`models/README.md`](models/README.md)。数据和权重均被 Git 忽略，Release 也不声称包含 Claude 预训练模型。

## 当前限制

- 默认配置按约 50 类、`224 x 224` 输入设计，实际类别数取自数据目录。
- 当前是控制台工作流，没有桌面 GUI。
- 未在真实车标数据集上完成基准训练，因此不提供未经验证的准确率。
- PyTorch 模型格式与 `codex/` 的纯 C++ `CPPCNN1` 格式不兼容。
