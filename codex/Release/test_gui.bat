@echo off
echo Starting cppcnn_gui.exe...
cd /d "%~dp0"
cppcnn_gui.exe
echo Exit code: %ERRORLEVEL%
echo.
echo If you see no window, the app failed to start.
pause
