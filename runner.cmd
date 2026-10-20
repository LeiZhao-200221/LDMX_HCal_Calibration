@echo off
rem Launcher of src\runner.py for cmd.exe and PowerShell (works whatever the PowerShell execution policy)
rem   .\runner.cmd ci --electron 0.3 0.5 -s -test
rem Uses %LDMX_PYTHON% if set, otherwise "python" on PATH.
setlocal
if defined LDMX_PYTHON (set "PY=%LDMX_PYTHON%") else (set "PY=python")
"%PY%" -X utf8 "%~dp0src\runner.py" %*
exit /b %ERRORLEVEL%
