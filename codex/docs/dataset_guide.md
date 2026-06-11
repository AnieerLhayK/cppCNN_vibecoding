# GTSRB 数据集指南

## 1. 数据集来源

GTSRB（German Traffic Sign Recognition Benchmark）是常用的 43 类德国交通标志分类数据集。可从官方基准页面获取：

- 官方页面：<https://benchmark.ini.rub.de/gtsrb_dataset.html>
- TensorFlow Datasets 数据说明：<https://www.tensorflow.org/datasets/catalog/german_traffic_sign>

本项目只使用数据文件，不依赖 TensorFlow。

## 2. 推荐目录结构

`DataLoader` 要求训练集和测试集都整理为“类别目录包含图片”的结构：

```text
D:\datasets\gtsrb\
|-- train\
|   |-- 00000\
|   |   |-- image_001.ppm
|   |   `-- ...
|   |-- 00001\
|   `-- ...
`-- test\
    |-- 00000\
    |-- 00001\
    `-- ...
```

类别目录名必须是非负整数，可使用 `0`、`1` 或 GTSRB 常见的 `00000`、`00001`。支持 `.ppm`、`.pnm`、`.png`、`.jpg`、`.jpeg`、`.bmp`；未启用 OpenCV 时只能实际读取 PPM/PNM。

官方测试包可能通过 CSV 单独提供标签，而不是按类别分目录。请根据标签 CSV 将测试图片复制或链接到对应类别目录，或者自行写转换脚本。转换产物应放在数据集目录，不放入 Git。

## 3. 如何选择子集

命令：

```powershell
cppcnn_app train <train_dir> <model> <class_count> <epochs> <samples_per_class>
```

例如取前 10 类，每类最多 200 张：

```powershell
cppcnn_app train D:\datasets\gtsrb\train models\gtsrb10.bin 10 5 200
```

程序按数字类别 ID 升序选择前 N 个目录，并映射为连续标签。例如原始目录 `00002`、`00005` 被选择时，训练标签分别是 `0`、`1`。

课程演示建议：

- 快速烟雾测试：2 类，每类 20 至 50 张，1 至 2 个 epoch。
- 默认实验：10 类，尽量使用这些类别的全部训练图片。
- 计算资源允许时：逐步增加 epoch 和类别数。

## 4. 扩展到 43 类

1. 保留 `train/00000` 到 `train/00042`。
2. 将测试集同样整理为 `test/00000` 到 `test/00042`。
3. 训练命令的 `class_count` 改为 `43`。
4. 将 `assets/labels.txt` 扩展为 43 行，顺序与类别 ID 一致。
5. 适当增加 epoch，并考虑加入数据增强和学习率调度。

网络输出层会按运行时类别数创建，模型文件也记录类别数，因此代码本身不限定为 10 类。

## 5. Git 注意事项

不要提交：

- 完整 GTSRB 压缩包或解压图片
- `codex/data/`、`codex/dataset/`、`codex/datasets/`
- `models/*.bin`、`*.weights`、`*.model`
- 构建目录和运行日志

这些路径已写入 `codex/.gitignore`。大型数据集最好放在仓库之外，例如 `D:\datasets\gtsrb`。
