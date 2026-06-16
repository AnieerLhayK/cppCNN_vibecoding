@echo off
cd /d "D:\AAAfromC\source\repos\CNN\codex\Release"
set QT_DEBUG_PLUGINS=1
set QML_IMPORT_TRACE=1
set QT_QPA_DEBUG=1
set QT_LOGGING_RULES=qt.qpa.*=true
cppcnn_gui.exe
echo EXIT_CODE=%ERRORLEVEL%

