@echo off
REM Double-click wrapper for setup.ps1 - runs it with the execution-policy
REM restriction bypassed for just this one script, without changing your
REM system's PowerShell policy permanently.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
pause
