param(
    [string]$Version = "2.1.0",
    [string]$OutputDirectory = "D:\AI\data\codex\cache\staging\cppcnn-user-release",
    [string]$ApplicationDirectory = (Join-Path (Split-Path -Parent $PSScriptRoot) "Release"),
    [string]$TestSetWebsite = "https://benchmark.ini.rub.de/gtsrb_dataset.html"
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

if ($Version -notmatch '^\d+\.\d+\.\d+([-.][0-9A-Za-z.-]+)?$') {
    throw "Version must use a form such as 2.1.0 or 2.1.0-final."
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$packageName = "cppCNN-Codex-User-Software-v$Version"
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

Copy-DirectoryContent -Source $ApplicationDirectory -Destination $packageRoot

Set-Content `
    -LiteralPath (Join-Path $packageRoot "TEST_SET_WEBSITE.url") `
    -Encoding Ascii `
    -Value @"
[InternetShortcut]
URL=$TestSetWebsite
"@

Set-Content `
    -LiteralPath (Join-Path $packageRoot "USER_PACKAGE_README.md") `
    -Encoding UTF8 `
    -Value @(
        "# cppCNN Codex User Software",
        "",
        "This package is the end-user release for the Codex implementation.",
        "",
        '- Double-click `run_demo.bat` to launch the Qt GUI.',
        '- `TEST_SET_WEBSITE.url` opens the official GTSRB test-set website.',
        "- The shipped models are ready to use and no extra training step is required.",
        "",
        "Test-set website:",
        $TestSetWebsite
    )

Set-Content `
    -LiteralPath (Join-Path $packageRoot "VERSION.txt") `
    -Encoding UTF8 `
    -Value @(
        "cppCNN Codex User Software",
        "Package version: $Version",
        "Project: C++ CNN Traffic Sign Recognition",
        "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
    )

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
    -Encoding Ascii `
    -Value "$archiveHash  $([System.IO.Path]::GetFileName($archivePath))"

Write-Host "User software package: $archivePath"
Write-Host "SHA-256: $archiveHash"
Write-Host "Expanded package: $packageRoot"
