@echo off
setlocal EnableDelayedExpansion
chcp 65001 >NUL
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
  set "BASE_PY="
  py -3.13 -c "import sys" >NUL 2>NUL && set "BASE_PY=py -3.13"
  if not defined BASE_PY py -3.12 -c "import sys" >NUL 2>NUL && set "BASE_PY=py -3.12"
  if not defined BASE_PY py -3.11 -c "import sys" >NUL 2>NUL && set "BASE_PY=py -3.11"
  if not defined BASE_PY py -3.10 -c "import sys" >NUL 2>NUL && set "BASE_PY=py -3.10"
  if not defined BASE_PY set "BASE_PY=python"
  !BASE_PY! -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >NUL 2>NUL
  if errorlevel 1 (
    echo [eis-ecm-drt-mcp] Python 3.10 or newer is required. Please install Python 3.10+ and add it to PATH. 1>&2
    exit /b 1
  )
  echo [eis-ecm-drt-mcp] First run: creating local Python environment... 1>&2
  !BASE_PY! -m venv "%~dp0.venv"
)

if not exist "%~dp0.venv\.eis_ecm_drt_installed" (
  echo [eis-ecm-drt-mcp] First run: installing dependencies... 1>&2
  "%PYTHON%" -m pip install -r requirements.txt >NUL || goto install_failed
  echo installed>"%~dp0.venv\.eis_ecm_drt_installed"
)

"%PYTHON%" -m eis_ecm_drt.server
exit /b %ERRORLEVEL%

:install_failed
echo [eis-ecm-drt-mcp] Dependency installation failed. Check your internet connection or install requirements.txt manually. 1>&2
exit /b 1
