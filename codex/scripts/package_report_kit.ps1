param(
    [string]$Version = "1.1.0",
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
    throw "Version must use a release form such as 1.1.0: $Version"
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
    "models\gtsrb_subset10.bin",
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
    "include",
    "models\.gitignore",
    "models\README.md",
    "models\gtsrb_semantic10.labels.txt",
    "models\gtsrb_semantic10_model_card.md",
    "models\gtsrb_subset10.labels.txt",
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
    "Application model: GTSRB 10 classes",
    "Application model test Top-1: 89.63%",
    "Dataset included: no",
    "Pretrained application model included: yes",
    "",
    "Top-level contents:",
    "- application/: portable Windows GUI, runtime, model, labels and demos",
    "- source/: buildable Codex C++ source without datasets or model binaries",
    "- report_materials/: report, design, dataset, training and screenshot materials",
    "- numbered Chinese guides: run, read code, write report and prepare defense"
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
