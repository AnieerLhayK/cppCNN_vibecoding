# GTSRB 数据集指南

## 1. 数据来源

GTSRB（German Traffic Sign Recognition Benchmark）是 43 类德国交通标志分类基准。项目优先使用官方页面链接的 ERDA 公共归档：

- 官方说明页：<https://benchmark.ini.rub.de/gtsrb_dataset.html>
- 官方归档目录：<https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/>
- Kaggle 备用镜像：<https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign>

项目只读取数据文件，不依赖 Kaggle SDK、TensorFlow 或其他深度学习框架。

## 2. 下载链接

官方直链：

- 训练图：<https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip>
- 测试图：<https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip>
- 测试标签：<https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip>

PowerShell 下载示例：

```powershell
cd codex
New-Item -ItemType Directory -Force datasets\GTSRB\downloads

$base = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370"
curl.exe -L -C - -o datasets\GTSRB\downloads\GTSRB_Final_Training_Images.zip `
  "$base/GTSRB_Final_Training_Images.zip"
curl.exe -L -C - -o datasets\GTSRB\downloads\GTSRB_Final_Test_Images.zip `
  "$base/GTSRB_Final_Test_Images.zip"
curl.exe -L -C - -o datasets\GTSRB\downloads\GTSRB_Final_Test_GT.zip `
  "$base/GTSRB_Final_Test_GT.zip"
```

`-C -` 支持断点续传。可用 `Get-FileHash -Algorithm SHA256` 对照 `datasets/README.md` 校验。

## 3. 数据规模

| 项目 | 数量 |
| --- | ---: |
| 类别 | 43 |
| 训练图片 | 39,209 |
| 测试图片 | 12,630 |
| 总图片 | 51,839 |

训练类别不均衡，每类约 210 至 2,250 张。测试标签位于 `GT-final_test.csv`，以分号分隔。

## 4. 原始目录结构

官方 ZIP 解压后保留原始结构：

```text
codex/datasets/GTSRB/
|-- downloads/
|-- GTSRB/
|   |-- Final_Training/
|   |   `-- Images/
|   |       |-- 00000/
|   |       |-- 00001/
|   |       `-- 00042/
|   `-- Final_Test/
|       `-- Images/
|           |-- 00000.ppm
|           `-- ...
`-- GT-final_test.csv
```

`DataLoader::loadDataset` 默认识别这个结构：

- 训练时自动进入 `GTSRB/Final_Training/Images`。
- 评估时自动读取 `GTSRB/Final_Test/Images` 和 `GT-final_test.csv`。
- 同时兼容已经整理好的 `train/<class>`、`test/<class>` 子集目录。

因此训练和评估都可以直接传数据集根目录：

```powershell
.\build\Release\cppcnn_app.exe train datasets\GTSRB models\gtsrb43.bin 43 5 0
.\build\Release\cppcnn_app.exe evaluate datasets\GTSRB models\gtsrb43.bin 0
```

## 5. 开发子集

构建后运行：

```powershell
.\build\Release\cppcnn_create_subset.exe `
  datasets\GTSRB datasets\GTSRB_subset 10 1000
```

参数依次为：

```text
<完整数据根目录> <输出目录> <类别数> <每类训练图片数> [类别ID列表]
```

类别 ID 列表是可选的逗号分隔值。省略时自动选择样本数充足且 ID 最小的类别；提供时会严格使用指定类别。

每类图片数必须在 500 到 1000 之间。工具会：

1. 按原始类别 ID 排序；
2. 跳过训练图片不足指定数量的类别；
3. 每类复制指定数量的训练图；
4. 根据官方 CSV 复制这些类别的全部测试图；
5. 生成 `subset_manifest.txt` 和顺序一致的 `labels.txt`。

默认结果：

```text
classes=10
training_images_per_class=1000
training_images=10000
test_images=5670
class_ids=1,2,3,4,5,7,8,9,10,11
```

语义均衡 10 类：

```powershell
.\build\Release\cppcnn_create_subset.exe `
  datasets\GTSRB datasets\GTSRB_semantic10 `
  10 500 "2,9,11,13,14,17,18,25,28,38"
```

结果为 5,000 张数量均衡的训练图和 4,710 张官方测试图。推荐直接使用 [`developer_training.md`](developer_training.md) 中的一体化训练脚本。

开发阶段推荐：

```powershell
.\build\Release\cppcnn_app.exe train `
  datasets\GTSRB_subset models\gtsrb_v2_subset10.bin `
  10 5 0 16 0.01 0.0001 42

.\build\Release\cppcnn_app.exe evaluate `
  datasets\GTSRB_subset models\gtsrb_v2_subset10.bin 0

.\build\Release\cppcnn_app.exe predict `
  datasets\GTSRB_subset\test\00001\00001.ppm `
  models\gtsrb_v2_subset10.bin datasets\GTSRB_subset\labels.txt
```

## 6. 切换到完整 43 类

代码的输出类别数由运行参数决定，模型文件也保存类别数。切换完整数据只需：

1. 使用 `datasets/GTSRB` 根目录；
2. 将 `class_count` 设为 `43`；
3. 使用包含 43 行的标签文件；
4. 根据 CPU 性能调整 epoch 和每类样本上限。

完整训练较慢时可先使用：

```powershell
.\build\Release\cppcnn_app.exe train datasets\GTSRB models\debug43.bin 43 1 10
```

这会使用每类最多 10 张图片验证 43 类流程。

## 7. Git 与缺失资源

- `datasets/.gitignore` 排除所有真实数据，只跟踪说明文件。
- `models/.gitignore` 排除所有模型权重，只跟踪格式说明。
- 程序无参数启动时会显示完整数据集、开发子集和默认模型状态。
- 显式传入不存在的数据或模型时，程序会给出下载或训练提示并正常返回错误码。

不要使用 `git add -f` 提交数据或权重。
