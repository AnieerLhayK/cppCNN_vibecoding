# GUI 使用说明

## 运行条件

- Windows 10/11 x64。
- 无需安装 Qt、Visual Studio、CUDA、LibTorch、Python 或 OpenCV。
- 必须完整解压 ZIP，不能只从压缩软件中直接运行单个 EXE。

## 启动

进入提交包的 `application/` 目录，双击：

```text
run_demo.bat
```

也可以直接运行 `cppcnn_gui.exe`。

## 识别图片

1. 点击界面的“打开图片”，或按 `Ctrl+O`。
2. 选择 `demo_images/` 中任意图片，也可选择自己的 PNG、JPEG、BMP 或 PPM 图片。
3. 点击“开始识别”，或按 `Ctrl+Enter`。
4. 界面显示 Top-1 类别、置信度、Top-3 概率和推理耗时。
5. 按 `Esc` 清除当前图片。

## 切换模型

默认模型是：

```text
models/gtsrb_v2_subset10.bin
```

它识别 10 个开发子集类别。需要展示完整 GTSRB 43 类时，按 `Ctrl+M` 选择：

```text
models/gtsrb_v5_full43.bin
```

程序会自动读取同目录下的 `gtsrb_v5_full43.labels.txt`。

模型和标签必须匹配。不要手工修改 `.bin` 或 `.labels.txt` 内容。

## 命令行演示

双击 `run_cli_demo.bat` 可运行预设命令行预测。开发者也可以执行：

```powershell
.\cppcnn_app.exe predict `
  .\demo_images\01_class_00000_00000_00000.ppm `
  .\models\gtsrb_v5_full43.bin `
  .\models\gtsrb_v5_full43.labels.txt
```

## 常见问题

- 提示模型缺失：确认 `application/models/` 未被移动或删除。
- GUI 无法启动：重新完整解压，确认 DLL、`plugins/` 和 `qml/` 与 EXE 位于同一目录。
- 预测类别不合理：确认当前模型类别范围，并使用与模型匹配的标签。
- 自有图片打不开：可先转换为常见 PNG/JPEG，或使用包内 PPM 演示图片。
