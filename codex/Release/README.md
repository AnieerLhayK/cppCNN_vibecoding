# cppCNN 交通标志识别演示包

本目录是提供给教师的 Windows Release 演示包，不需要安装编译器或重新编译。

## 当前阶段状态

可执行程序、类别标签和演示图片已经就位。项目目前仍处于搭建阶段，按照要求尚未训练并放入正式模型权重，因此当前不能产生可信的交通标志识别结果。

不要使用随机参数或空文件冒充训练模型。正式交付前只需将训练好的文件放到：

```text
models/gtsrb_subset10.bin
```

放入后，教师可直接双击 `run_demo.bat`，无需重新训练。

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

- `Model file is missing`：模型仍未放入 `models/`，这是当前搭建阶段的预期状态。
- Windows SmartScreen 提示：选择“更多信息”，确认程序来源后运行。
- 无法打开 PNG/JPEG：此发布构建没有 OpenCV，请使用随包提供的 PPM 图片。
