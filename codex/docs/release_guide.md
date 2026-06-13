# 发布指南

## 发布物与 Git 的边界

- Git 跟踪源码、测试、文档、启动脚本和小型演示图片。
- `datasets/` 中的真实数据集不进入 Git。
- `models/*.bin` 中的训练权重不进入 Git。
- Qt DLL、QML 运行时和插件不进入 Git。
- 模型与 Qt 运行时只进入 GitHub Release 的 ZIP 资产。

这种方式既符合机器学习项目管理规范，也允许教师直接下载并演示。

## 生成 Windows 便携包

在仓库根目录执行：

```powershell
.\codex\scripts\package_release.ps1 `
  -BuildDirectory D:\AI\data\codex\cache\staging\cppcnn-release-build `
  -ArtifactsDirectory D:\AI\data\codex\cache\staging\cppcnn-release-artifacts `
  -ModelPath .\codex\models\gtsrb_subset10.bin `
  -QtRoot C:\Qt\6.11.1\msvc2022_64 `
  -Version 1.0.0
```

脚本会完成：

1. 使用 Visual Studio 2022 x64 构建 GUI 与 CLI。
2. 使用 Qt 官方部署工具收集运行库。
3. 加入模型、标签、演示图、启动脚本和使用说明。
4. 更新本地 `codex/Release/` 演示目录。
5. 在临时目录生成版本化 ZIP 和 `.sha256` 文件。

## 发布前检查

```powershell
ctest --test-dir D:\AI\data\codex\cache\staging\cppcnn-release-build `
  -C Release --output-on-failure

Get-FileHash `
  D:\AI\data\codex\cache\staging\cppcnn-release-artifacts\cppCNN-Traffic-Sign-Studio-v1.0.0-windows-x64.zip `
  -Algorithm SHA256
```

还应在未配置 Qt `PATH` 的环境中启动压缩包内的 `run_demo.bat`，点击至少一张演示图并确认 Top-1、置信度和 Top-3 正常显示。

## 创建 GitHub Release

```powershell
gh release create v1.0.0 `
  D:\AI\data\codex\cache\staging\cppcnn-release-artifacts\cppCNN-Traffic-Sign-Studio-v1.0.0-windows-x64.zip `
  D:\AI\data\codex\cache\staging\cppcnn-release-artifacts\cppCNN-Traffic-Sign-Studio-v1.0.0-windows-x64.zip.sha256 `
  --repo f32797653-beep/cppCNN_vibecoding `
  --target main `
  --title "cppCNN Traffic Sign Studio v1.0.0" `
  --notes-file .\codex\docs\release_notes_v1.0.0.md `
  --latest
```

正式发布前必须保证 `main` 已推送、CI 通过，并且版本号、标签和发布说明一致。

