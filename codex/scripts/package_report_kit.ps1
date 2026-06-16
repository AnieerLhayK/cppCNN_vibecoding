param(
    [string]$Version = "2.0.0",
    [string]$ApplicationDirectory = "",
    [string]$OutputDirectory = (Join-Path ([System.IO.Path]::GetTempPath()) "cppcnn-report-kit")
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$guideRoot = Join-Path $projectRoot "report_kit"

if ([string]::IsNullOrWhiteSpace($ApplicationDirectory)) {
    $ApplicationDirectory = Join-Path $projectRoot "Release"
}

if ($Version -notmatch '^\d+\.\d+\.\d+([-.][0-9A-Za-z.-]+)?$') {
    throw "Version must use a release form such as 2.0.0: $Version"
}

$packageName = "cppCNN-Codex-Report-Kit-v$Version"
$packageRoot = Join-Path $OutputDirectory $packageName
$archivePath = Join-Path $OutputDirectory "$packageName.zip"
$checksumPath = "$archivePath.sha256"

$outputFull = [System.IO.Path]::GetFullPath($OutputDirectory)
$packageFull = [System.IO.Path]::GetFullPath($packageRoot)
if (-not $packageFull.StartsWith(
        $outputFull + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to clean a package directory outside the output directory: $packageFull"
}

$requiredApplicationFiles = @(
    "cppcnn_gui.exe",
    "cppcnn_app.exe",
    "run_demo.bat",
    "labels.txt",
    "models\gtsrb_v2_subset10.bin",
    "models\gtsrb_v5_full43.bin",
    "models\gtsrb_v5_full43.labels.txt",
    "plugins\platforms\qwindows.dll",
    "qml\QtQuick\qtquick2plugin.dll"
)
foreach ($relativePath in $requiredApplicationFiles) {
    $path = Join-Path $ApplicationDirectory $relativePath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Portable application is incomplete; missing: $path"
    }
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
if (Test-Path -LiteralPath $packageRoot) {
    Remove-Item -LiteralPath $packageRoot -Recurse -Force
}
if (Test-Path -LiteralPath $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}
if (Test-Path -LiteralPath $checksumPath) {
    Remove-Item -LiteralPath $checksumPath -Force
}

$applicationTarget = Join-Path $packageRoot "application"
$sourceTarget = Join-Path $packageRoot "source"
$materialsTarget = Join-Path $packageRoot "report_materials"
New-Item -ItemType Directory -Force -Path $applicationTarget | Out-Null
New-Item -ItemType Directory -Force -Path $sourceTarget | Out-Null
New-Item -ItemType Directory -Force -Path $materialsTarget | Out-Null

Get-ChildItem -LiteralPath $ApplicationDirectory -Force |
    Where-Object { $_.Name -ne "log" } |
    Copy-Item -Destination $applicationTarget -Recurse -Force

$sourceEntries = @(
    ".gitignore",
    "CHANGELOG.md",
    "CMakeLists.txt",
    "README.md",
    "assets",
    "datasets\.gitignore",
    "datasets\README.md",
    "docs",
    "models\.gitignore",
    "models\README.md",
    "models\gtsrb_v2_subset10.labels.txt",
    "models\gtsrb_v4_semantic10.labels.txt",
    "models\gtsrb_v5_full43.labels.txt",
    "models\gtsrb_semantic10_model_card.md",
    "qml",
    "report_kit",
    "scripts",
    "src",
    "tests",
    "tools"
)
foreach ($entry in $sourceEntries) {
    $source = Join-Path $projectRoot $entry
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Required source entry is missing: $source"
    }
    $destination = Join-Path $sourceTarget $entry
    $parent = Split-Path -Parent $destination
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    Copy-Item -LiteralPath $source -Destination $destination -Recurse -Force
}

$materialEntries = @(
    "docs\project_report.md",
    "docs\design_notes.md",
    "docs\dataset_guide.md",
    "docs\developer_training.md",
    "docs\gui_preview.png",
    "docs\model_report.md",
    "docs\phase_b_baseline_results.md",
    "docs\phase_c_results.md",
    "docs\release_guide.md",
    "models\gtsrb_semantic10_model_card.md"
)
foreach ($entry in $materialEntries) {
    $source = Join-Path $projectRoot $entry
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required report material is missing: $source"
    }
    Copy-Item -LiteralPath $source -Destination $materialsTarget -Force
}

$aiRecordSource = Join-Path (Split-Path -Parent $projectRoot) "docs\exports\codex_session_export.html"
if (-not (Test-Path -LiteralPath $aiRecordSource -PathType Leaf)) {
    throw "AI usage record is missing: $aiRecordSource"
}
Copy-Item `
    -LiteralPath $aiRecordSource `
    -Destination (Join-Path $materialsTarget "codex_session_export.html") `
    -Force

Get-ChildItem -LiteralPath $guideRoot -File |
    Copy-Item -Destination $packageRoot -Force

Set-Content `
    -LiteralPath (Join-Path $packageRoot "VERSION.txt") `
    -Encoding Ascii `
    -Value "cppCNN Codex Report Kit $Version`r`nCodex-only course report and demo package"

$manifest = @(
    "cppCNN Codex Report Kit v$Version",
    "",
    "Scope: codex/ pure C++ implementation only",
    "Default application model: GTSRB subset, 10 classes",
    "Optional full model: GTSRB Enhanced, 43 classes",
    "Dataset included: no",
    "Pretrained application models included: yes",
    "AI usage record included: report_materials/codex_session_export.html",
    "",
    "Top-level contents:",
    "- application/: portable Windows GUI, runtime, models, labels and demos",
    "- source/: buildable Codex C++ source without datasets or model binaries",
    "- report_materials/: report, design, dataset, training, screenshot and AI usage materials",
    "- 01_宝宝级资料包指南.md: run, report and defense guide",
    "- 02_代码阅读指南.md: code reading guide"
)
Set-Content `
    -LiteralPath (Join-Path $packageRoot "PACKAGE_MANIFEST.txt") `
    -Encoding UTF8 `
    -Value $manifest

Compress-Archive `
    -LiteralPath $packageRoot `
    -DestinationPath $archivePath `
    -CompressionLevel Optimal

$archiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content `
    -LiteralPath $checksumPath `
    -Encoding Ascii `
    -Value "$archiveHash  $([System.IO.Path]::GetFileName($archivePath))"

Write-Host "Report kit directory: $packageRoot"
Write-Host "Report kit archive: $archivePath"
Write-Host "SHA-256: $archiveHash"
