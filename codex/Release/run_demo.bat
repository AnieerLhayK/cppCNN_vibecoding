@echo off
setlocal
cd /d "%~dp0"

set "APP=cppcnn_app.exe"
set "MODEL=models\gtsrb_subset10.bin"
set "IMAGE=demo_images\01_speed_limit_30.ppm"
set "LABELS=labels.txt"

if not exist "%APP%" (
    echo [ERROR] Missing executable: %APP%
    echo Rebuild the Release package with scripts\package_release.ps1.
    pause
    exit /b 1
)

if not exist "%MODEL%" (
    echo [INFO] The Release package structure is ready, but the trained model is not included yet.
    echo Expected model: %MODEL%
    echo This is intentional during the current project setup stage.
    echo After training, place the model at the path above and run this file again.
    pause
    exit /b 2
)

if not exist "%IMAGE%" (
    echo [ERROR] Missing demo image: %IMAGE%
    pause
    exit /b 3
)

"%APP%" predict "%IMAGE%" "%MODEL%" "%LABELS%"
set "RESULT=%ERRORLEVEL%"
echo.
pause
exit /b %RESULT%
