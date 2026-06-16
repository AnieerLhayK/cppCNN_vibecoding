# 发布指南

## 发布物与 Git 边界

- Git 跟踪源码、测试、文档、启动脚本、小型演示图片和说明文件。
- `datasets/` 中的真实数据集不进入 Git。
- `models/*.bin` 与 `models/*.pt` 训练权重不进入 Git。
- Qt DLL、QML 运行时和插件不进入 Git。
- 模型、Qt 运行时、AI 使用记录和报告资料包只进入 GitHub Release 的 ZIP 资产。

这种方式符合机器学习项目管理规范，也允许教师直接下载并演示。

## 生成 Windows 便携包

在仓库根目录执行：

```powershell
.\codex\scripts\package_release.ps1 `
  -BuildDirectory D:\AI\data\codex\cache\staging\cppcnn-release-build `
  -ArtifactsDirectory D:\AI\data\codex\cache\staging\cppcnn-release-artifacts `
  -ModelPath .\codex\models\gtsrb_v2_subset10.bin `
  -QtRoot C:\Qt\6.11.1\msvc2022_64 `
  -Version 2.0.0
```

脚本会完成：

1. 使用 Visual Studio 2022 x64 构建 GUI 与 CPU CLI。
2. 使用 Qt 官方部署工具收集运行库。
3. 加入默认 10 类模型、完整 43 类模型、标签、演示图、AI 使用记录、启动脚本和使用说明。
4. 更新本地 `codex/Release/` 演示目录。
5. 在临时目录生成版本化 ZIP 和 `.sha256` 文件。

教师便携包默认不包含 `cppcnn_app_gpu.exe` 和 LibTorch/CUDA 运行库。GPU 训练工具仅用于开发者继续实验，普通演示不需要配置 CUDA 或 LibTorch。

## 发布前检查

```powershell
ctest --test-dir D:\AI\data\codex\cache\staging\cppcnn-release-build `
  -C Release --output-on-failure

Get-FileHash `
  D:\AI\data\codex\cache\staging\cppcnn-release-artifacts\cppCNN-Traffic-Sign-Studio-v2.0.0-windows-x64.zip `
  -Algorithm SHA256
```

还应在未配置 Qt `PATH` 的环境中启动压缩包内的 `run_demo.bat`，点击至少三张演示图，并确认 Top-1、置信度和 Top-3 正常显示。需要展示完整 43 类模型时，在 GUI 中按 `Ctrl+M` 选择 `models/gtsrb_v5_full43.bin`。

## 生成课程报告资料包

在完整便携应用已经生成后执行：

```powershell
.\codex\scripts\package_report_kit.ps1 `
  -Version 2.0.0 `
  -ApplicationDirectory .\codex\Release `
  -OutputDirectory D:\AI\data\codex\cache\staging\cppcnn-report-kit
```

生成的 `cppCNN-Codex-Report-Kit-v2.0.0.zip` 作为同一 Release 的补充资产上传。它只包含 Codex 实现，不包含 `claude/`、完整数据集、构建缓存或 Git 历史，但包含应用程序、源码、报告资料、宝宝级指南和 `codex_session_export.html` AI 使用记录。

## 创建 GitHub Release

正式发布前必须保证 `main` 已包含发布提交并推送。创建 Release：

```powershell
gh release create v2.0.0 `
  D:\AI\data\codex\cache\staging\cppcnn-release-artifacts\cppCNN-Traffic-Sign-Studio-v2.0.0-windows-x64.zip `
  D:\AI\data\codex\cache\staging\cppcnn-release-artifacts\cppCNN-Traffic-Sign-Studio-v2.0.0-windows-x64.zip.sha256 `
  D:\AI\data\codex\cache\staging\cppcnn-report-kit\cppCNN-Codex-Report-Kit-v2.0.0.zip `
  D:\AI\data\codex\cache\staging\cppcnn-report-kit\cppCNN-Codex-Report-Kit-v2.0.0.zip.sha256 `
  --repo f32797653-beep/cppCNN_vibecoding `
  --target main `
  --title "cppCNN Traffic Sign Studio v2.0.0" `
  --notes-file .\codex\docs\release_notes_v2.0.0.md `
  --latest
```

`v2.0.0-gpu` 是早期 GPU 后端标签；正式发布使用 `v2.0.0`，二者不会冲突。
