@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-case.ps1" %*
exit /b %ERRORLEVEL%
