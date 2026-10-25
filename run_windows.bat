@echo off
setlocal
cd /d "%~dp0"
title AI Vision and People Analytics
if not exist ".venv\Scripts\python.exe" (
  echo Project environment not found. Run setup_windows.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo The virtual environment [.venv] is invalid or was copied from another computer.
  echo Please run setup_windows.bat to set up the environment for this machine.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python main.py
if errorlevel 1 (
  echo.
  echo Application exited with an error. Read the traceback above.
  pause
)
