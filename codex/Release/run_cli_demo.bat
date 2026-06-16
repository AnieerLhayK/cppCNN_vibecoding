@echo off
setlocal
cd /d "%~dp0"

set "APP=cppcnn_app.exe"
set "MODEL=models\gtsrb_v2_subset10.bin"
set "IMAGE=demo_images\01_speed_limit_30.ppm"
set "LABELS=labels.txt"

if not exist "%APP%" (
    echo [ERROR] Missing executable: %APP%
    pause
    exit /b 1
)
if not exist "%MODEL%" (
    echo [ERROR] Missing model: %MODEL%
    pause
    exit /b 2
)

"%APP%" predict "%IMAGE%" "%MODEL%" "%LABELS%"
set "RESULT=%ERRORLEVEL%"
echo.
pause
exit /b %RESULT%
