# cppCNN 交通标志识别演示包

本目录是 `v1.1.0` 面向教师演示的 Windows x64 便携包。完成本地打包后，无需安装 Qt、Visual Studio、Python、OpenCV，也无需重新训练。

## 快速开始

双击：

```text
run_demo.bat
```

或直接运行：

```text
cppcnn_gui.exe
```

在界面中点击任意演示图片即可显示 Top-1、置信度、推理时间和 Top-3。五张图片分别展示限速、禁止超车、重型车辆限制和路口优先规则，不再是连续限速牌。

快捷键：

- `Ctrl+O`：打开图片。
- `Ctrl+Enter`：运行识别。
- `Esc`：清除当前图片。

## 目录内容

```text
Release/
├── cppcnn_gui.exe          # Qt 桌面界面
├── cppcnn_app.exe          # CLI 训练与推理程序
├── cppcnn_app_gpu.exe      # CLI GPU 加速版 (附带 LibTorch/CUDA DLL)
├── Qt6*.dll                # Qt 运行库，本地生成且不入 Git
├── plugins/                # Qt 平台与图片插件
├── qml/                    # Qt Quick 运行模块
├── models/
│   ├── gtsrb_v2_subset10.bin  # 默认 10 类模型
│   └── gtsrb_v5_full43.bin    # 完整 43 类模型 (Ctrl+M 加载)
├── demo_images/            # 五张 GTSRB 演示图片
├── labels.txt              # 10 类标签
├── qt.conf                 # 便携资源路径
├── VERSION.txt             # 发布版本
├── run_demo.bat            # GUI 启动脚本
└── run_cli_demo.bat        # CLI 演示脚本
```

## 模型结果

```text
训练样本：10,000
测试样本：5,670
Epoch：5
训练准确率：94.61%
测试准确率：89.63%
```

## 重新生成

在仓库根目录执行：

```powershell
.\codex\scripts\package_release.ps1 `
  -ModelPath .\codex\models\gtsrb_v2_subset10.bin `
  -QtRoot C:\Qt\6.11.1\msvc2022_64 `
  -Version 1.1.0
```

如果模型或 Qt 组件缺失，脚本会明确报错并停止，不会生成残缺包。正式下载请使用仓库的 [GitHub Releases](https://github.com/f32797653-beep/cppCNN_vibecoding/releases/latest)。
