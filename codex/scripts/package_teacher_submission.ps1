param(
    [string]$Version = "2.0.0",
    [string]$OutputDirectory = "D:\AI\data\codex\cache\staging\cppcnn-teacher-submission",
    [int]$TrainImagesPerClass = 100,
    [int]$TestImagesPerClass = 20
)

$ErrorActionPreference = "Stop"

function Copy-DirectoryContent {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "Required directory is missing: $Source"
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Path (Join-Path $Source "*") -Destination $Destination -Recurse -Force
}

function Copy-RequiredFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Required file is missing: $Source"
    }
    $destinationParent = Split-Path -Parent $Destination
    if ($destinationParent) {
        New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

if ($Version -notmatch '^\d+\.\d+\.\d+([-.][0-9A-Za-z.-]+)?$') {
    throw "Version must use a form such as 2.0.0 or 2.0.0-final."
}
if ($TrainImagesPerClass -lt 1 -or $TestImagesPerClass -lt 1) {
    throw "Sample counts must be positive."
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$repositoryRoot = Split-Path -Parent $projectRoot
$packageName = "cppCNN-Codex-Teacher-Submission-v$Version"
$packageRoot = Join-Path $OutputDirectory $packageName
$archivePath = Join-Path $OutputDirectory "$packageName.zip"
$checksumPath = "$archivePath.sha256"

$outputFull = [System.IO.Path]::GetFullPath($OutputDirectory)
$packageFull = [System.IO.Path]::GetFullPath($packageRoot)
if (-not $packageFull.StartsWith(
        $outputFull + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to clean a package directory outside the output directory."
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
if (Test-Path -LiteralPath $packageRoot) {
    Remove-Item -LiteralPath $packageRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $packageRoot | Out-Null

$submissionDocs = Join-Path $projectRoot "docs\teacher_submission"
Copy-RequiredFile `
    -Source (Join-Path $submissionDocs "README.md") `
    -Destination (Join-Path $packageRoot "README.md")

foreach ($docName in @(
    "application_usage.md",
    "technical_details.md",
    "model_versions.md",
    "build_and_development.md"
)) {
    Copy-RequiredFile `
        -Source (Join-Path $submissionDocs $docName) `
        -Destination (Join-Path $packageRoot "docs\$docName")
}

# Teacher-facing portable application. Training and reporting handoff material
# is deliberately excluded from this runtime directory.
$applicationRoot = Join-Path $packageRoot "application"
Copy-DirectoryContent -Source (Join-Path $projectRoot "Release") -Destination $applicationRoot
$applicationAiRecords = Join-Path $applicationRoot "ai_records"
if (Test-Path -LiteralPath $applicationAiRecords) {
    Remove-Item -LiteralPath $applicationAiRecords -Recurse -Force
}
Copy-RequiredFile `
    -Source (Join-Path $submissionDocs "application_usage.md") `
    -Destination (Join-Path $applicationRoot "README_TEACHER.md")

# Complete Codex implementation source, including optional GPU code and CMake
# linkage, but excluding datasets, model weights, builds and old report kits.
$sourceRoot = Join-Path $packageRoot "source"
foreach ($directory in @("src", "qml", "assets", "tests", "tools")) {
    Copy-DirectoryContent `
        -Source (Join-Path $projectRoot $directory) `
        -Destination (Join-Path $sourceRoot $directory)
}
New-Item -ItemType Directory -Force -Path (Join-Path $sourceRoot "scripts") | Out-Null
foreach ($scriptName in @("create_subset.cpp", "train_model.ps1")) {
    Copy-RequiredFile `
        -Source (Join-Path $projectRoot "scripts\$scriptName") `
        -Destination (Join-Path $sourceRoot "scripts\$scriptName")
}
foreach ($fileName in @("CMakeLists.txt", "README.md", ".gitignore")) {
    Copy-RequiredFile `
        -Source (Join-Path $projectRoot $fileName) `
        -Destination (Join-Path $sourceRoot $fileName)
}
foreach ($docName in @(
    "dataset_guide.md",
    "design_notes.md",
    "developer_training.md",
    "model_report.md",
    "phase_b_baseline_results.md",
    "phase_c_results.md",
    "project_report.md"
)) {
    Copy-RequiredFile `
        -Source (Join-Path $projectRoot "docs\$docName") `
        -Destination (Join-Path $sourceRoot "docs\$docName")
}

# A compact but directly loadable 10-class sample dataset.
$sampleSource = Join-Path $projectRoot "datasets\GTSRB_subset"
$sampleRoot = Join-Path $packageRoot "sample_dataset"
New-Item -ItemType Directory -Force -Path $sampleRoot | Out-Null
Copy-RequiredFile `
    -Source (Join-Path $sampleSource "labels.txt") `
    -Destination (Join-Path $sampleRoot "labels.txt")

$classDirectories = Get-ChildItem -LiteralPath (Join-Path $sampleSource "train") -Directory |
    Sort-Object Name
if ($classDirectories.Count -ne 10) {
    throw "Expected exactly 10 classes in GTSRB_subset, found $($classDirectories.Count)."
}

foreach ($classDirectory in $classDirectories) {
    foreach ($split in @(
        @{ Name = "train"; Limit = $TrainImagesPerClass },
        @{ Name = "test"; Limit = $TestImagesPerClass }
    )) {
        $sourceClass = Join-Path (Join-Path $sampleSource $split.Name) $classDirectory.Name
        $destinationClass = Join-Path (Join-Path $sampleRoot $split.Name) $classDirectory.Name
        $images = Get-ChildItem -LiteralPath $sourceClass -File -Filter *.ppm |
            Sort-Object Name |
            Select-Object -First $split.Limit
        if ($images.Count -lt $split.Limit) {
            throw "Class $($classDirectory.Name) has only $($images.Count) $($split.Name) images."
        }
        New-Item -ItemType Directory -Force -Path $destinationClass | Out-Null
        foreach ($image in $images) {
            Copy-Item -LiteralPath $image.FullName -Destination $destinationClass -Force
        }
    }
}

Set-Content `
    -LiteralPath (Join-Path $sampleRoot "README.md") `
    -Value @(
        "# GTSRB Compact Sample Dataset",
        "",
        "- Classes: 10",
        "- Training images: $TrainImagesPerClass per class, $($TrainImagesPerClass * 10) total",
        "- Test images: $TestImagesPerClass per class, $($TestImagesPerClass * 10) total",
        "- Format: PPM",
        "- Purpose: DataLoader, training, evaluation and prediction verification",
        "",
        "This compact sample is not the full dataset used for the reported results."
    ) `
    -Encoding UTF8

# All locally available model versions and their metadata.
$modelDestination = Join-Path $packageRoot "models"
New-Item -ItemType Directory -Force -Path $modelDestination | Out-Null
$modelPatterns = @(
    "*.bin",
    "*.pt",
    "*.labels.txt",
    "*.metadata.json",
    "*.training.log",
    "*model_card.md"
)
$copiedModelFiles = New-Object System.Collections.Generic.HashSet[string]
foreach ($pattern in $modelPatterns) {
    $matchingModelFiles = Get-ChildItem `
        -LiteralPath (Join-Path $projectRoot "models") `
        -File `
        -Filter $pattern
    foreach ($modelFile in $matchingModelFiles) {
        Copy-Item -LiteralPath $modelFile.FullName -Destination $modelDestination -Force
        $null = $copiedModelFiles.Add($modelFile.Name)
    }
}
Copy-RequiredFile `
    -Source (Join-Path $submissionDocs "model_versions.md") `
    -Destination (Join-Path $modelDestination "MODEL_VERSIONS.md")
if ($copiedModelFiles.Count -lt 10) {
    throw "Too few model artifacts were copied: $($copiedModelFiles.Count)."
}

# AI process evidence. No slide decks or onboarding/report-kit files are copied.
$aiDestination = Join-Path $packageRoot "ai_records"
New-Item -ItemType Directory -Force -Path $aiDestination | Out-Null
Copy-RequiredFile `
    -Source (Join-Path $repositoryRoot "docs\exports\codex_session_export.html") `
    -Destination (Join-Path $aiDestination "codex_session_export.html")
$pdfRecords = Get-ChildItem `
    -LiteralPath (Join-Path $repositoryRoot "docs\exports") `
    -File `
    -Filter *.pdf
if ($pdfRecords.Count -ne 1) {
    throw "Expected exactly one PDF AI record, found $($pdfRecords.Count)."
}
Copy-RequiredFile `
    -Source $pdfRecords[0].FullName `
    -Destination (Join-Path $aiDestination "gpt_cpp_code_discussion.pdf")
Copy-RequiredFile `
    -Source (Join-Path $submissionDocs "ai_records.md") `
    -Destination (Join-Path $aiDestination "README.md")
if ((Get-ChildItem -LiteralPath $aiDestination -File).Count -ne 3) {
    throw "AI records package is incomplete."
}

Set-Content `
    -LiteralPath (Join-Path $packageRoot "VERSION.txt") `
    -Value @(
        "cppCNN Codex Teacher Submission",
        "Package version: $Version",
        "Project: C++ CNN Traffic Sign Recognition",
        "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
    ) `
    -Encoding UTF8

$manifestPath = Join-Path $packageRoot "PACKAGE_MANIFEST.txt"
Get-ChildItem -LiteralPath $packageRoot -Recurse -File |
    Sort-Object FullName |
    ForEach-Object {
        $_.FullName.Substring($packageRoot.Length + 1)
    } |
    Set-Content -LiteralPath $manifestPath -Encoding UTF8

$hashPath = Join-Path $packageRoot "SHA256SUMS.txt"
Get-ChildItem -LiteralPath $packageRoot -Recurse -File |
    Where-Object { $_.FullName -ne $hashPath } |
    Sort-Object FullName |
    ForEach-Object {
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $relative = $_.FullName.Substring($packageRoot.Length + 1).Replace("\", "/")
        "$hash  $relative"
    } |
    Set-Content -LiteralPath $hashPath -Encoding Ascii

if (Test-Path -LiteralPath $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}
if (Test-Path -LiteralPath $checksumPath) {
    Remove-Item -LiteralPath $checksumPath -Force
}

Compress-Archive -LiteralPath $packageRoot -DestinationPath $archivePath -CompressionLevel Optimal
$archiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content `
    -LiteralPath $checksumPath `
    -Value "$archiveHash  $([System.IO.Path]::GetFileName($archivePath))" `
    -Encoding Ascii

Write-Host "Teacher submission package: $archivePath"
Write-Host "SHA-256: $archiveHash"
Write-Host "Expanded package: $packageRoot"
