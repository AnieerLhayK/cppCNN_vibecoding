@echo off
setlocal
cd /d "%~dp0"

set "APP=cppcnn_gui.exe"
set "MODEL=models\gtsrb_subset10.bin"

if not exist "%APP%" (
    echo [ERROR] Missing executable: %APP%
    echo Rebuild the Release package with scripts\package_release.ps1.
    pause
    exit /b 1
)

if not exist "%MODEL%" (
    echo [ERROR] The trained model is missing.
    echo Expected model: %MODEL%
    echo Rebuild the Release package with scripts\package_release.ps1.
    pause
    exit /b 2
)

start "" "%APP%"
exit /b 0
