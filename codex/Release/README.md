# cppCNN 交通标志识别演示包

本目录是 `v2.0.0` 面向教师演示的 Windows x64 便携包。完成本地打包后，无需安装 Qt、Visual Studio、Python、OpenCV，也无需重新训练。

## 快速开始

双击：

```text
run_demo.bat
```

或直接运行：

```text
cppcnn_gui.exe
```

在界面中点击任意演示图片即可显示 Top-1、置信度、推理时间和 Top-3。演示图覆盖多个不同交通标志类别，不再只是连续限速牌。

快捷键：

- `Ctrl+O`：打开图片。
- `Ctrl+Enter`：运行识别。
- `Esc`：清除当前图片。

## 目录内容

```text
Release/
├── cppcnn_gui.exe          # Qt 桌面界面
├── cppcnn_app.exe          # CLI 训练与推理程序
├── Qt6*.dll                # Qt 运行库，本地生成且不入 Git
├── plugins/                # Qt 平台与图片插件
├── qml/                    # Qt Quick 运行模块
├── models/
│   ├── gtsrb_v2_subset10.bin  # 默认 10 类模型
│   └── gtsrb_v5_full43.bin    # 完整 43 类模型 (Ctrl+M 加载)
├── demo_images/            # 50 张 GTSRB 演示图片
├── ai_records/
│   └── codex_session_export.html # AI 使用记录
├── labels.txt              # 10 类标签
├── qt.conf                 # 便携资源路径
├── VERSION.txt             # 发布版本
├── run_demo.bat            # GUI 启动脚本
└── run_cli_demo.bat        # CLI 演示脚本
```

## 模型结果

```text
默认模型：gtsrb_v2_subset10.bin，10 类，适合快速演示
完整模型：gtsrb_v5_full43.bin，43 类 Enhanced，可在 GUI 中 Ctrl+M 加载
43 类模型官方测试集 Top-1：95.26%
```

## 重新生成

在仓库根目录执行：

```powershell
.\codex\scripts\package_release.ps1 `
  -ModelPath .\codex\models\gtsrb_v2_subset10.bin `
  -QtRoot C:\Qt\6.11.1\msvc2022_64 `
  -Version 2.0.0
```

如果模型或 Qt 组件缺失，脚本会明确报错并停止，不会生成残缺包。正式下载请使用仓库的 [GitHub Releases](https://github.com/f32797653-beep/cppCNN_vibecoding/releases/latest)。
