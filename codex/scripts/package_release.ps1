param(
    [string]$BuildDirectory = (Join-Path ([System.IO.Path]::GetTempPath()) "cppcnn-release-build"),
    [string]$ModelPath = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$releaseRoot = Join-Path $projectRoot "Release"
$sourceDataset = Join-Path $projectRoot "datasets\GTSRB_subset"

$demoImages = @(
    @{ Source = "test\00001\00001.ppm"; Destination = "01_speed_limit_30.ppm" },
    @{ Source = "test\00002\00034.ppm"; Destination = "02_speed_limit_50.ppm" },
    @{ Source = "test\00003\00023.ppm"; Destination = "03_speed_limit_60.ppm" },
    @{ Source = "test\00004\00014.ppm"; Destination = "04_speed_limit_70.ppm" },
    @{ Source = "test\00005\00030.ppm"; Destination = "05_speed_limit_80.ppm" }
)

if (-not (Test-Path -LiteralPath $sourceDataset -PathType Container)) {
    throw "Development dataset is missing: $sourceDataset"
}

New-Item -ItemType Directory -Force -Path $BuildDirectory | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $releaseRoot "demo_images") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $releaseRoot "models") | Out-Null

cmake `
    -S $projectRoot `
    -B $BuildDirectory `
    -DCPPCNN_BUILD_TESTS=OFF `
    -DCPPCNN_BUILD_DEMO_GENERATOR=OFF `
    -DCPPCNN_BUILD_SUBSET_TOOL=OFF `
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
if ($LASTEXITCODE -ne 0) {
    throw "CMake configuration failed."
}

cmake --build $BuildDirectory --config Release --target cppcnn_app
if ($LASTEXITCODE -ne 0) {
    throw "Release build failed."
}

$executable = Join-Path $BuildDirectory "Release\cppcnn_app.exe"
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Built executable was not found: $executable"
}
Copy-Item -LiteralPath $executable -Destination (Join-Path $releaseRoot "cppcnn_app.exe") -Force

Copy-Item `
    -LiteralPath (Join-Path $sourceDataset "labels.txt") `
    -Destination (Join-Path $releaseRoot "labels.txt") `
    -Force

foreach ($image in $demoImages) {
    $source = Join-Path $sourceDataset $image.Source
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Demo image is missing: $source"
    }
    Copy-Item `
        -LiteralPath $source `
        -Destination (Join-Path $releaseRoot "demo_images\$($image.Destination)") `
        -Force
}

if (-not [string]::IsNullOrWhiteSpace($ModelPath)) {
    if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
        throw "Model file is missing: $ModelPath"
    }
    Copy-Item `
        -LiteralPath $ModelPath `
        -Destination (Join-Path $releaseRoot "models\gtsrb_subset10.bin") `
        -Force
    Write-Host "Model copied into the Release package."
} else {
    Write-Host "The trained model remains intentionally absent during the setup stage."
}

Write-Host "Release package prepared at: $releaseRoot"
