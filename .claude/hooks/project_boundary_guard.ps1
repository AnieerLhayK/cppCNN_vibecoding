$ErrorActionPreference = "Stop"

function Deny([string]$Reason) {
    [Console]::Error.WriteLine($Reason)
    exit 2
}

function Resolve-GuardPath([string]$RawPath, [string]$Cwd) {
    if ([System.IO.Path]::IsPathRooted($RawPath)) {
        return [System.IO.Path]::GetFullPath($RawPath)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $Cwd $RawPath))
}

function Test-Within([string]$Path, [string]$Root) {
    $fullPath = [System.IO.Path]::GetFullPath($Path).TrimEnd("\", "/")
    $fullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd("\", "/")
    return $fullPath.Equals($fullRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
        $fullPath.StartsWith($fullRoot + "\", [System.StringComparison]::OrdinalIgnoreCase)
}

$payload = [Console]::In.ReadToEnd() | ConvertFrom-Json
$projectRoot = [System.IO.Path]::GetFullPath($env:CLAUDE_PROJECT_DIR)
$cwd = [System.IO.Path]::GetFullPath([string]$payload.cwd)
$toolName = [string]$payload.tool_name
$toolInput = $payload.tool_input

if (-not (Test-Within $cwd $projectRoot)) {
    Deny "Blocked: Claude cwd is outside the CNN project root: $cwd"
}

if ($toolName -in @("Write", "Edit", "MultiEdit", "NotebookEdit")) {
    $rawPath = if ($toolInput.file_path) { $toolInput.file_path } else { $toolInput.notebook_path }
    if ($rawPath) {
        $target = Resolve-GuardPath ([string]$rawPath) $cwd
        if (-not (Test-Within $target $projectRoot)) {
            Deny "Blocked: file mutation is outside the CNN Git root."
        }
    }
}

if ($toolName -eq "Bash") {
    $command = [string]$toolInput.command
    $mutating = "(?i)(?:^|[;&|]\s*)(?:mkdir|md|rmdir|rm|del|erase|copy|cp|move|mv|new-item|remove-item|move-item|copy-item|set-content|add-content|out-file|git\s+(?:add|commit|checkout|switch|reset|clean|merge|rebase))\b"
    if ($command -match $mutating) {
        foreach ($match in [regex]::Matches($command, "(?i)([A-Z]:[\\/][^`"'`r`n;&|]*)")) {
            $target = Resolve-GuardPath $match.Groups[1].Value.Trim() $cwd
            if (-not (Test-Within $target $projectRoot)) {
                Deny "Blocked: mutating command targets a path outside the CNN Git root."
            }
        }
    }
}

exit 0
