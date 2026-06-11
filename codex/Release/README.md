# cppCNN 交通标志识别演示包

本目录是提供给教师的 Windows Release 演示包，不需要安装编译器或重新编译。

## 当前状态

可执行程序、类别标签、演示图片和训练模型均已就位。本地 Release 包可以直接运行，不需要重新训练。

模型位置：

```text
models/gtsrb_subset10.bin
```

该模型使用 10 类、每类 1,000 张图片训练 5 个 epoch：

```text
训练集：10,000 张
测试集：5,670 张
训练准确率：94.61%
测试准确率：89.63%
```

教师可直接双击 `run_demo.bat`。

## 目录内容

```text
Release/
|-- cppcnn_app.exe          # 静态 MSVC Release 可执行程序
|-- labels.txt              # 与 10 类开发模型对应的标签
|-- demo_images/            # 5 张 GTSRB 演示图片
|-- models/
|   `-- README.md           # 模型位置和格式说明
|-- run_demo.bat            # 双击演示
`-- README.md
```

## 演示方法

正式模型就位后：

1. 双击 `run_demo.bat`。
2. 程序使用 `demo_images/01_speed_limit_30.ppm`。
3. 控制台显示预测类别和置信度。
4. 当前构建未附带 OpenCV，因此图片窗口不可用；图片可直接用支持 PPM 的查看器打开。

也可以在命令行选择其他图片：

```powershell
.\cppcnn_app.exe predict `
  .\demo_images\02_speed_limit_50.ppm `
  .\models\gtsrb_subset10.bin `
  .\labels.txt
```

## 常见提示

- `Model file is missing`：发布包复制不完整，请确认 `models/gtsrb_subset10.bin` 存在。
- Windows SmartScreen 提示：选择“更多信息”，确认程序来源后运行。
- 无法打开 PNG/JPEG：此发布构建没有 OpenCV，请使用随包提供的 PPM 图片。
