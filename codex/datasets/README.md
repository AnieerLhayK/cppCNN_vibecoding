# Datasets

此目录保存本地机器学习数据。图片、压缩包、CSV 标注和生成子集均不进入 Git；目录中的 `.gitignore` 只允许提交本 README 和忽略规则。

## GTSRB

- 名称：German Traffic Sign Recognition Benchmark
- 任务：多类别单图交通标志分类
- 类别数量：43
- 训练集：39,209 张
- 测试集：12,630 张
- 总计：51,839 张
- 图片格式：PPM
- 官方页面：<https://benchmark.ini.rub.de/gtsrb_dataset.html>
- 官方公共归档：<https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/>
- Kaggle 备用镜像：<https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign>

## 官方下载文件

| 文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| `GTSRB_Final_Training_Images.zip` | 276,294,756 B | `D32AC4B5FA9A1CBD1994768413902E8193599D9434CF0A8EB9CFD00A6D3A290C` |
| `GTSRB_Final_Test_Images.zip` | 88,978,620 B | `48BA6FAB7E877EB64EAF8DE99035B0AAECFBC279BEE23E35DECA4AC1D0A837FA` |
| `GTSRB_Final_Test_GT.zip` | 99,620 B | `F94E5A7614D75845C74C04DDB26B8796B9E483F43541DD95DD5B726504E16D6D` |

## 本地目录

```text
datasets/
|-- README.md
|-- GTSRB/
|   |-- downloads/
|   |   |-- GTSRB_Final_Training_Images.zip
|   |   |-- GTSRB_Final_Test_Images.zip
|   |   `-- GTSRB_Final_Test_GT.zip
|   |-- GTSRB/
|   |   |-- Final_Training/Images/00000 ... 00042/
|   |   `-- Final_Test/Images/
|   `-- GT-final_test.csv
`-- GTSRB_subset/
    |-- train/
    |-- test/
    |-- labels.txt
    `-- subset_manifest.txt
```

当前开发子集由 `cppcnn_create_subset` 生成：

- 10 类
- 每类 500 张训练图
- 5,000 张训练图
- 5,670 张测试图
- 原始类别 ID：`1,2,3,4,5,7,8,9,10,11`

第 0 和第 6 类不足 500 张训练图，因此默认工具会选择样本数满足要求的类别。详细下载、解压、生成和运行说明见 `../docs/dataset_guide.md`。
