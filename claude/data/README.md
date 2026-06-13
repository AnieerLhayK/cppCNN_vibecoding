# 车标数据集准备

本目录只保留说明文件，真实图片不会提交到 Git。请确认所选数据集的许可证允许课程使用、展示和重新分发；不要将第三方完整数据集打包进本仓库或 GitHub Release。

## 支持的目录

准备脚本默认读取按品牌名称分目录的图片：

```
claude/data/
  raw/
    brand_a/
      img1.jpg
      img2.jpg
    brand_b/
      ...
```text

运行脚本后生成：

```text
claude/data/
  train/
    brand_a/...
  val/
    brand_a/...
  test/
    brand_a/...
  manifests/
    train_manifest.txt
    val_manifest.txt
    test_manifest.txt
```

也可以手工准备已划分的数据：

```text
claude/data/
  train/
    brand_a/...
  val/
    brand_a/...
  test/
    brand_a/...
```

## 数据集选择

可使用公开车标数据集或自行收集的已授权图片，例如 XMU、VLD、CCLD 等常见研究数据。不同镜像的类别数、图片数和许可可能不同，下载前应以来源页面为准，并记录：

- 数据集名称、来源 URL 和下载日期；
- 使用许可证或课程使用授权；
- 类别数、总图片数和每类分布；
- 是否存在重复图、损坏图或明显类别不平衡。

## 运行

从仓库根目录执行：

```powershell
python claude/scripts/prepare_data.py --dry-run
python claude/scripts/prepare_data.py
```

默认按每个类别独立随机划分为 `70% / 15% / 15%`，随机种子为 `42`。可通过 `--train-ratio`、`--val-ratio`、`--seed`、`--raw-dir` 和 `--output-dir` 调整。

训练前应检查控制台输出的类别分布。极小类别可能无法在三个集合中都保留样本，建议每类至少准备数十张图片，并尽量保持语义和数量均衡。
