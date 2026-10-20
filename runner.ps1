# PowerShell launcher of src/runner.py (works from any folder)
#   .\runner.ps1 ci --electron 0.3 0.5 -s -test
#   .\runner.ps1 check
# Uses $env:LDMX_PYTHON if set, otherwise "python" on PATH.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = if ($env:LDMX_PYTHON) { $env:LDMX_PYTHON } else { "python" }
# "Continue" for the native call: with "Stop", Windows PowerShell 5.1 aborts the run at the first stderr line when
# the caller redirects or pipes stderr (2>&1, 2> file); the exit code of runner.py is passed on below
$ErrorActionPreference = "Continue"
& $py -X utf8 (Join-Path $here "src\runner.py") @args
exit $LASTEXITCODE
