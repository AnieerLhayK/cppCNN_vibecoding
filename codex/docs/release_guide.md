# Release Guide

This guide explains how to recreate the `v2.1.0` GitHub Release assets for the Codex implementation.

## Git boundaries

- Source code, tests, scripts and documentation are tracked in Git.
- `datasets/` never enters Git.
- `models/*.bin` and `models/*.pt` never enter Git.
- Qt DLLs, QML runtime files and plugins never enter Git.
- Model weights, Qt runtime files, AI records and submission materials are distributed through GitHub Release ZIP assets only.

This keeps the repository clean while still letting teachers and classmates download a ready-to-run package.

## Build the portable runtime package

Run from the repository root:

```powershell
.\codex\scripts\package_release.ps1 `
  -BuildDirectory D:\AI\data\codex\cache\staging\cppcnn-release-build `
  -ArtifactsDirectory D:\AI\data\codex\cache\staging\cppcnn-release-artifacts `
  -ModelPath .\codex\models\gtsrb_v2_subset10.bin `
  -QtRoot C:\Qt\6.11.1\msvc2022_64 `
  -Version 2.1.0
```

This produces:

- `cppCNN-Traffic-Sign-Studio-v2.1.0-windows-x64.zip`
- `cppCNN-Traffic-Sign-Studio-v2.1.0-windows-x64.zip.sha256`

The runtime package includes the GUI, CLI, Qt runtime, models, labels, demo images and AI record.

## Build the user software package

Run from the repository root:

```powershell
.\codex\scripts\package_user_release.ps1 `
  -OutputDirectory D:\AI\data\codex\cache\staging\cppcnn-user-release `
  -Version 2.1.0
```

This produces:

- `cppCNN-Codex-User-Software-v2.1.0.zip`
- `cppCNN-Codex-User-Software-v2.1.0.zip.sha256`

The user package includes the runnable application plus a Windows shortcut file that opens the official GTSRB test-set website.

## Build the teacher submission package

Run from the repository root:

```powershell
.\codex\scripts\package_teacher_submission.ps1 `
  -Version 2.1.0 `
  -OutputDirectory D:\AI\data\codex\cache\staging\cppcnn-teacher-submission
```

This produces:

- `cppCNN-Codex-Teacher-Submission-v2.1.0.zip`
- `cppCNN-Codex-Teacher-Submission-v2.1.0.zip.sha256`

## Create the GitHub Release

After the assets are generated and `main` contains the release commit, create the GitHub Release:

```powershell
gh release create v2.1.0 `
  D:\AI\data\codex\cache\staging\cppcnn-release-artifacts\cppCNN-Traffic-Sign-Studio-v2.1.0-windows-x64.zip `
  D:\AI\data\codex\cache\staging\cppcnn-release-artifacts\cppCNN-Traffic-Sign-Studio-v2.1.0-windows-x64.zip.sha256 `
  D:\AI\data\codex\cache\staging\cppcnn-user-release\cppCNN-Codex-User-Software-v2.1.0.zip `
  D:\AI\data\codex\cache\staging\cppcnn-user-release\cppCNN-Codex-User-Software-v2.1.0.zip.sha256 `
  D:\AI\data\codex\cache\staging\cppcnn-teacher-submission\cppCNN-Codex-Teacher-Submission-v2.1.0.zip `
  D:\AI\data\codex\cache\staging\cppcnn-teacher-submission\cppCNN-Codex-Teacher-Submission-v2.1.0.zip.sha256 `
  --repo AnieerLhayK/cppCNN_vibecoding `
  --target main `
  --title "cppCNN Traffic Sign Studio v2.1.0" `
  --notes-file .\codex\docs\release_notes_v2.1.0.md `
  --latest
```

The `v2.1.0` release now carries all three ZIP assets:

1. Portable runtime package.
2. Teacher submission bundle.
3. End-user package with the test-set website shortcut.
