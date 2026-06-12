param(
    [string]$BuildDirectory = (Join-Path ([System.IO.Path]::GetTempPath()) "cppcnn-release-build"),
    [string]$ModelPath = "",
    [string]$QtRoot = "C:\Qt\6.11.1\msvc2022_64"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$releaseRoot = Join-Path $projectRoot "Release"
$sourceDataset = Join-Path $projectRoot "datasets\GTSRB_subset"
$packageRoot = Join-Path $BuildDirectory "portable"

$demoImages = @(
    @{ Source = "test\00001\00001.ppm"; Destination = "01_speed_limit_30.ppm" },
    @{ Source = "test\00002\00034.ppm"; Destination = "02_speed_limit_50.ppm" },
    @{ Source = "test\00003\00036.ppm"; Destination = "03_speed_limit_60.ppm" },
    @{ Source = "test\00004\00014.ppm"; Destination = "04_speed_limit_70.ppm" },
    @{ Source = "test\00005\00030.ppm"; Destination = "05_speed_limit_80.ppm" }
)

if (-not (Test-Path -LiteralPath $sourceDataset -PathType Container)) {
    throw "Development dataset is missing: $sourceDataset"
}
if (-not (Test-Path -LiteralPath (Join-Path $QtRoot "bin\windeployqt.exe") -PathType Leaf)) {
    throw "Qt 6 MSVC deployment tools were not found under: $QtRoot"
}

if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = Join-Path $projectRoot "models\gtsrb_subset10.bin"
}
if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
    throw "Trained model is missing: $ModelPath"
}

$buildFull = [System.IO.Path]::GetFullPath($BuildDirectory)
$packageFull = [System.IO.Path]::GetFullPath($packageRoot)
if (-not $packageFull.StartsWith($buildFull, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to clean a package directory outside the build directory: $packageFull"
}

New-Item -ItemType Directory -Force -Path $BuildDirectory | Out-Null
if (Test-Path -LiteralPath $packageRoot) {
    Remove-Item -LiteralPath $packageRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $packageRoot | Out-Null

cmake `
    -S $projectRoot `
    -B $BuildDirectory `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -DCPPCNN_BUILD_TESTS=OFF `
    -DCPPCNN_BUILD_DEMO_GENERATOR=OFF `
    -DCPPCNN_BUILD_SUBSET_TOOL=OFF `
    -DCPPCNN_BUILD_GUI=ON `
    "-DCMAKE_PREFIX_PATH=$QtRoot"
if ($LASTEXITCODE -ne 0) {
    throw "CMake configuration failed."
}

cmake --build $BuildDirectory --config Release --target cppcnn_app cppcnn_gui --parallel
if ($LASTEXITCODE -ne 0) {
    throw "Release build failed."
}

cmake --install $BuildDirectory --config Release --prefix $packageRoot
if ($LASTEXITCODE -ne 0) {
    throw "Qt deployment failed."
}

$deployedBin = Join-Path $packageRoot "bin"
if (Test-Path -LiteralPath $deployedBin -PathType Container) {
    Copy-Item -Path (Join-Path $deployedBin "*") -Destination $packageRoot -Recurse -Force
    Remove-Item -LiteralPath $deployedBin -Recurse -Force
}
Set-Content `
    -LiteralPath (Join-Path $packageRoot "qt.conf") `
    -Encoding Ascii `
    -Value "[Paths]`r`nPrefix = ."

New-Item -ItemType Directory -Force -Path (Join-Path $packageRoot "demo_images") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $packageRoot "models") | Out-Null

Copy-Item `
    -LiteralPath (Join-Path $sourceDataset "labels.txt") `
    -Destination (Join-Path $packageRoot "labels.txt") `
    -Force

foreach ($image in $demoImages) {
    $source = Join-Path $sourceDataset $image.Source
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Demo image is missing: $source"
    }
    Copy-Item `
        -LiteralPath $source `
        -Destination (Join-Path $packageRoot "demo_images\$($image.Destination)") `
        -Force
}

Copy-Item `
    -LiteralPath $ModelPath `
    -Destination (Join-Path $packageRoot "models\gtsrb_subset10.bin") `
    -Force

New-Item -ItemType Directory -Force -Path $releaseRoot | Out-Null
$generatedDirectories = @("bin", "generic", "iconengines", "imageformats", "networkinformation", "platforms", "plugins", "qml", "styles", "tls", "translations")
foreach ($directory in $generatedDirectories) {
    $generatedPath = Join-Path $releaseRoot $directory
    if (Test-Path -LiteralPath $generatedPath) {
        Remove-Item -LiteralPath $generatedPath -Recurse -Force
    }
}
Copy-Item -Path (Join-Path $packageRoot "*") -Destination $releaseRoot -Recurse -Force

Write-Host "Portable Qt Release package prepared at: $releaseRoot"
Write-Host "Run cppcnn_gui.exe for the desktop interface."
