param(
    [string]$BuildDirectory = (Join-Path ([System.IO.Path]::GetTempPath()) "cppcnn-release-build"),
    [string]$ModelPath = "",
    [string]$QtRoot = "C:\Qt\6.11.1\msvc2022_64",
    [string]$LibTorchRoot = "D:\SDK\libtorch-2.12.0-cu130",
    [string]$Version = "1.1.0",
    [string]$ArtifactsDirectory = (Join-Path ([System.IO.Path]::GetTempPath()) "cppcnn-release-artifacts")
)

$ErrorActionPreference = "Stop"

function Invoke-NativeStep {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,
        [Parameter(Mandatory = $true)]
        [string]$FailureMessage
    )

    $previousPreference = $ErrorActionPreference
    try {
        # Windows PowerShell surfaces native stderr as an ErrorRecord. Qt's
        # deployment tools use stderr for non-fatal warnings, so exit codes
        # are the reliable success signal for these commands.
        $ErrorActionPreference = "Continue"
        & $Command
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    if ($exitCode -ne 0) {
        throw "$FailureMessage (exit code $exitCode)"
    }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$releaseRoot = Join-Path $projectRoot "Release"
$sourceDataset = Join-Path $projectRoot "datasets\GTSRB_subset"
$packageRoot = Join-Path $BuildDirectory "portable"
$artifactName = "cppCNN-Traffic-Sign-Studio-v$Version-windows-x64"
$artifactPackageRoot = Join-Path $ArtifactsDirectory $artifactName
$archivePath = Join-Path $ArtifactsDirectory "$artifactName.zip"
$checksumPath = "$archivePath.sha256"

if ($Version -notmatch '^\d+\.\d+\.\d+([-.][0-9A-Za-z.-]+)?$') {
    throw "Version must use a release form such as 1.0.0 or 1.0.0-rc1: $Version"
}

$demoImages = @(
    @{ Source = "test\00001\00284.ppm"; Destination = "01_speed_limit_30.ppm" },
    @{ Source = "test\00007\00366.ppm"; Destination = "02_speed_limit_100.ppm" },
    @{ Source = "test\00009\00497.ppm"; Destination = "03_no_passing.ppm" },
    @{ Source = "test\00010\00084.ppm"; Destination = "04_heavy_vehicle_no_passing.ppm" },
    @{ Source = "test\00011\00601.ppm"; Destination = "05_intersection_priority.ppm" }
)

if (-not (Test-Path -LiteralPath $sourceDataset -PathType Container)) {
    throw "Development dataset is missing: $sourceDataset"
}
if (-not (Test-Path -LiteralPath (Join-Path $QtRoot "bin\windeployqt.exe") -PathType Leaf)) {
    throw "Qt 6 MSVC deployment tools were not found under: $QtRoot"
}

if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = Join-Path $projectRoot "models\gtsrb_v2_subset10.bin"
}
if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
    throw "Trained model is missing: $ModelPath"
}

$buildFull = [System.IO.Path]::GetFullPath($BuildDirectory)
$packageFull = [System.IO.Path]::GetFullPath($packageRoot)
if (-not $packageFull.StartsWith($buildFull, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to clean a package directory outside the build directory: $packageFull"
}

$artifactsFull = [System.IO.Path]::GetFullPath($ArtifactsDirectory)
$artifactPackageFull = [System.IO.Path]::GetFullPath($artifactPackageRoot)
if (-not $artifactPackageFull.StartsWith($artifactsFull, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to clean an artifact directory outside the configured artifacts directory: $artifactPackageFull"
}

New-Item -ItemType Directory -Force -Path $BuildDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $ArtifactsDirectory | Out-Null
if (Test-Path -LiteralPath $packageRoot) {
    Remove-Item -LiteralPath $packageRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $packageRoot | Out-Null

Invoke-NativeStep -FailureMessage "CMake configuration failed." -Command {
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
}

Invoke-NativeStep -FailureMessage "Release build failed." -Command {
    cmake --build $BuildDirectory --config Release --target cppcnn_app cppcnn_gui --parallel
}

Invoke-NativeStep -FailureMessage "Qt deployment failed." -Command {
    cmake --install $BuildDirectory --config Release --prefix $packageRoot
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

# QML debugging plugins are not required by the teacher-facing release build.
$qmlToolingPath = Join-Path $packageRoot "plugins\qmltooling"
if (Test-Path -LiteralPath $qmlToolingPath -PathType Container) {
    Remove-Item -LiteralPath $qmlToolingPath -Recurse -Force
}

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
    -Destination (Join-Path $packageRoot "models\gtsrb_v2_subset10.bin") `
    -Force

# Also include the full 43-class model for manual Ctrl+M loading.
$full43Model = Join-Path $projectRoot "models\gtsrb_v5_full43.bin"
$full43Labels = Join-Path $projectRoot "models\gtsrb_v5_full43.labels.txt"
if (Test-Path -LiteralPath $full43Model -PathType Leaf) {
    Copy-Item -LiteralPath $full43Model -Destination (Join-Path $packageRoot "models\gtsrb_v5_full43.bin") -Force
}
if (Test-Path -LiteralPath $full43Labels -PathType Leaf) {
    Copy-Item -LiteralPath $full43Labels -Destination (Join-Path $packageRoot "models\gtsrb_v5_full43.labels.txt") -Force
}

# Include the GPU-accelerated CLI if pre-built.
$gpuAppSource = Join-Path $projectRoot "build_libtorch\cppcnn_app_gpu.exe"
if (Test-Path -LiteralPath $gpuAppSource -PathType Leaf) {
    Write-Host "GPU executable found; including in package."
    Copy-Item -LiteralPath $gpuAppSource -Destination (Join-Path $packageRoot "cppcnn_app_gpu.exe") -Force

    # Deploy LibTorch / CUDA runtime DLLs alongside the GPU executable.
    $libTorchBin = Join-Path $LibTorchRoot "bin"
    if (Test-Path -LiteralPath $libTorchBin -PathType Container) {
        Write-Host "Deploying LibTorch/CUDA runtime DLLs from $libTorchBin"
        Copy-Item -Path (Join-Path $libTorchBin "*.dll") -Destination $packageRoot -Force
    }
    else {
        Write-Warning "LibTorch root not found at '$LibTorchRoot' — GPU executable will not run without its runtime DLLs."
    }
}

$deliveryFiles = @(
    "README.md",
    "run_demo.bat",
    "run_cli_demo.bat"
)
foreach ($deliveryFile in $deliveryFiles) {
    Copy-Item `
        -LiteralPath (Join-Path $releaseRoot $deliveryFile) `
        -Destination (Join-Path $packageRoot $deliveryFile) `
        -Force
}
Copy-Item `
    -LiteralPath (Join-Path $releaseRoot "models\README.md") `
    -Destination (Join-Path $packageRoot "models\README.md") `
    -Force
Copy-Item `
    -LiteralPath (Join-Path $releaseRoot "demo_images\README.md") `
    -Destination (Join-Path $packageRoot "demo_images\README.md") `
    -Force
Set-Content `
    -LiteralPath (Join-Path $packageRoot "VERSION.txt") `
    -Encoding Ascii `
    -Value "cppCNN Traffic Sign Studio $Version`r`nWindows x64 portable release"

New-Item -ItemType Directory -Force -Path $releaseRoot | Out-Null
$generatedDirectories = @("bin", "generic", "iconengines", "imageformats", "networkinformation", "platforms", "plugins", "qml", "styles", "tls", "translations")
foreach ($directory in $generatedDirectories) {
    $generatedPath = Join-Path $releaseRoot $directory
    if (Test-Path -LiteralPath $generatedPath) {
        Remove-Item -LiteralPath $generatedPath -Recurse -Force
    }
}
Copy-Item -Path (Join-Path $packageRoot "*") -Destination $releaseRoot -Recurse -Force

if (Test-Path -LiteralPath $artifactPackageRoot) {
    Remove-Item -LiteralPath $artifactPackageRoot -Recurse -Force
}
if (Test-Path -LiteralPath $archivePath -PathType Leaf) {
    Remove-Item -LiteralPath $archivePath -Force
}
if (Test-Path -LiteralPath $checksumPath -PathType Leaf) {
    Remove-Item -LiteralPath $checksumPath -Force
}

Copy-Item -LiteralPath $packageRoot -Destination $artifactPackageRoot -Recurse
Compress-Archive -LiteralPath $artifactPackageRoot -DestinationPath $archivePath -CompressionLevel Optimal
$archiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content `
    -LiteralPath $checksumPath `
    -Encoding Ascii `
    -Value "$archiveHash  $([System.IO.Path]::GetFileName($archivePath))"

Write-Host "Portable Qt Release package prepared at: $releaseRoot"
Write-Host "Versioned archive: $archivePath"
Write-Host "SHA-256: $archiveHash"
Write-Host "Run cppcnn_gui.exe for the desktop interface."
