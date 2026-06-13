param(
    [string]$DatasetSource = "",
    [string]$SubsetDirectory = "",
    [string]$ModelPath = "",
    [int[]]$ClassIds = @(2, 9, 11, 13, 14, 17, 18, 25, 28, 38),
    [ValidateRange(500, 1000)]
    [int]$ImagesPerClass = 500,
    [ValidateRange(1, 1000)]
    [int]$Epochs = 5,
    [ValidateRange(1, 4096)]
    [int]$BatchSize = 16,
    [ValidateRange(0.000001, 10.0)]
    [double]$LearningRate = 0.01,
    [ValidateRange(0.0, 10.0)]
    [double]$WeightDecay = 0.0001,
    [ValidateRange(0, 2147483647)]
    [int]$Seed = 42,
    [string]$BuildDirectory = "",
    [switch]$SkipBuild,
    [switch]$SkipSubset
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

function Resolve-ProjectPath {
    param(
        [string]$Value,
        [string]$DefaultRelativePath
    )
    $path = if ([string]::IsNullOrWhiteSpace($Value)) {
        Join-Path $projectRoot $DefaultRelativePath
    } elseif ([System.IO.Path]::IsPathRooted($Value)) {
        $Value
    } else {
        Join-Path $projectRoot $Value
    }
    return [System.IO.Path]::GetFullPath($path)
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$DatasetSource = Resolve-ProjectPath $DatasetSource "datasets\GTSRB"
$SubsetDirectory = Resolve-ProjectPath $SubsetDirectory "datasets\GTSRB_semantic10"
$ModelPath = Resolve-ProjectPath $ModelPath "models\gtsrb_semantic10.bin"
if ([string]::IsNullOrWhiteSpace($BuildDirectory)) {
    $BuildDirectory = "D:\AI\data\codex\cache\staging\cppcnn-training-build"
}
$BuildDirectory = [System.IO.Path]::GetFullPath($BuildDirectory)

if ($ClassIds.Count -lt 2) {
    throw "ClassIds must contain at least two GTSRB class IDs."
}
$uniqueClassIds = @($ClassIds | Sort-Object -Unique)
if ($uniqueClassIds.Count -ne $ClassIds.Count) {
    throw "ClassIds must not contain duplicates."
}
foreach ($classId in $uniqueClassIds) {
    if ($classId -lt 0 -or $classId -gt 42) {
        throw "ClassIds must contain GTSRB IDs from 0 to 42."
    }
}

$subsetTool = Join-Path $BuildDirectory "Release\cppcnn_create_subset.exe"
$trainingApp = Join-Path $BuildDirectory "Release\cppcnn_app.exe"
if (-not $SkipBuild) {
    New-Item -ItemType Directory -Force -Path $BuildDirectory | Out-Null
    Invoke-NativeStep -FailureMessage "CMake configuration failed." -Command {
        cmake `
            -S $projectRoot `
            -B $BuildDirectory `
            -G "Visual Studio 17 2022" `
            -A x64 `
            -DCPPCNN_BUILD_TESTS=ON `
            -DCPPCNN_BUILD_DEMO_GENERATOR=OFF `
            -DCPPCNN_BUILD_SUBSET_TOOL=ON `
            -DCPPCNN_BUILD_GUI=OFF
    }
    Invoke-NativeStep -FailureMessage "Training tools build failed." -Command {
        cmake --build $BuildDirectory --config Release `
            --target cppcnn_app cppcnn_create_subset cppcnn_basic_tests --parallel
    }
    Invoke-NativeStep -FailureMessage "Core tests failed." -Command {
        ctest --test-dir $BuildDirectory -C Release --output-on-failure
    }
}

if (-not (Test-Path -LiteralPath $trainingApp -PathType Leaf)) {
    throw "Training executable is missing: $trainingApp"
}
if (-not $SkipSubset) {
    if (-not (Test-Path -LiteralPath $DatasetSource -PathType Container)) {
        throw "Full GTSRB dataset is missing: $DatasetSource"
    }
    if (-not (Test-Path -LiteralPath $subsetTool -PathType Leaf)) {
        throw "Subset tool is missing: $subsetTool"
    }
    $classIdText = ($uniqueClassIds -join ",")
    Invoke-NativeStep -FailureMessage "Semantic subset creation failed." -Command {
        & $subsetTool `
            $DatasetSource `
            $SubsetDirectory `
            $uniqueClassIds.Count `
            $ImagesPerClass `
            $classIdText
    }
}

if (-not (Test-Path -LiteralPath $SubsetDirectory -PathType Container)) {
    throw "Training subset is missing: $SubsetDirectory"
}

$modelDirectory = Split-Path -Parent $ModelPath
New-Item -ItemType Directory -Force -Path $modelDirectory | Out-Null
if (Test-Path -LiteralPath $ModelPath -PathType Leaf) {
    $archiveDirectory = Join-Path $modelDirectory "archive"
    New-Item -ItemType Directory -Force -Path $archiveDirectory | Out-Null
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $modelBaseName = [System.IO.Path]::GetFileNameWithoutExtension($ModelPath)
    $runArchive = Join-Path $archiveDirectory "$modelBaseName.$timestamp"
    New-Item -ItemType Directory -Force -Path $runArchive | Out-Null
    $relatedFiles = @(
        $ModelPath,
        [System.IO.Path]::ChangeExtension($ModelPath, ".training.log"),
        [System.IO.Path]::ChangeExtension($ModelPath, ".metadata.json"),
        [System.IO.Path]::ChangeExtension($ModelPath, ".labels.txt")
    )
    foreach ($relatedFile in $relatedFiles) {
        if (Test-Path -LiteralPath $relatedFile -PathType Leaf) {
            Move-Item -LiteralPath $relatedFile -Destination $runArchive
        }
    }
}

$runStarted = Get-Date
$logPath = [System.IO.Path]::ChangeExtension($ModelPath, ".training.log")
$metadataPath = [System.IO.Path]::ChangeExtension($ModelPath, ".metadata.json")
$classCount = $uniqueClassIds.Count

& $trainingApp train `
    $SubsetDirectory `
    $ModelPath `
    $classCount `
    $Epochs `
    0 `
    $BatchSize `
    $LearningRate.ToString([Globalization.CultureInfo]::InvariantCulture) `
    $WeightDecay.ToString([Globalization.CultureInfo]::InvariantCulture) `
    $Seed 2>&1 | Tee-Object -FilePath $logPath
if ($LASTEXITCODE -ne 0) {
    throw "Model training failed (exit code $LASTEXITCODE). See: $logPath"
}

$evaluationOutput = & $trainingApp evaluate $SubsetDirectory $ModelPath 0 2>&1
if ($LASTEXITCODE -ne 0) {
    $evaluationOutput | Tee-Object -FilePath $logPath -Append
    throw "Model evaluation failed (exit code $LASTEXITCODE). See: $logPath"
}
$evaluationOutput | Tee-Object -FilePath $logPath -Append

$labelsPath = Join-Path $SubsetDirectory "labels.txt"
$labels = @(
    Get-Content -LiteralPath $labelsPath -Encoding UTF8 |
        ForEach-Object { [string]$_ })
$companionLabelsPath = [System.IO.Path]::ChangeExtension($ModelPath, ".labels.txt")
Copy-Item -LiteralPath $labelsPath -Destination $companionLabelsPath -Force
$accuracyLine = [string](
    $evaluationOutput | Where-Object { $_ -like "Accuracy:*" } | Select-Object -Last 1)
$lossLine = [string](
    $evaluationOutput | Where-Object { $_ -like "Average loss:*" } | Select-Object -Last 1)
$samplesLine = [string](
    $evaluationOutput | Where-Object { $_ -like "Evaluation samples:*" } | Select-Object -Last 1)
$accuracyValue = ($accuracyLine -replace "^Accuracy:\s*", "") -replace "%", ""
$lossValue = $lossLine -replace "^Average loss:\s*", ""
$samplesValue = $samplesLine -replace "^Evaluation samples:\s*", ""

$metadata = [ordered]@{
    schema_version = 1
    model = [System.IO.Path]::GetFileName($ModelPath)
    model_sha256 = (Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    created_at = (Get-Date).ToString("o")
    duration_seconds = [math]::Round(((Get-Date) - $runStarted).TotalSeconds, 2)
    dataset = $SubsetDirectory
    source_dataset = $DatasetSource
    class_ids = @($uniqueClassIds)
    labels = @($labels)
    training_images_per_class = $ImagesPerClass
    epochs = $Epochs
    batch_size = $BatchSize
    learning_rate = $LearningRate
    weight_decay = $WeightDecay
    seed = $Seed
    evaluation_samples = [int]$samplesValue.Trim()
    evaluation_loss = [double]$lossValue.Trim()
    evaluation_accuracy_percent = [double]$accuracyValue.Trim()
    training_log = $logPath
    labels_file = $companionLabelsPath
}
$metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $metadataPath -Encoding UTF8

Write-Host ""
Write-Host "Training workflow complete."
Write-Host "Model: $ModelPath"
Write-Host "Metadata: $metadataPath"
Write-Host "Labels: $companionLabelsPath"
Write-Host "Log: $logPath"
