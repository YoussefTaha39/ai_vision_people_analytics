@echo off
setlocal
cd /d "%~dp0"
title AI Vision & People Analytics - Setup

echo ======================================================
echo   AI Vision ^& People Analytics - Windows Setup
echo ======================================================
echo.
set "PY_CMD="
where py >nul 2>nul
if not errorlevel 1 (
  set "PY_CMD=py -3.12"
) else (
  where python >nul 2>nul
  if not errorlevel 1 (
    set "PY_CMD=python"
  )
)

if "%PY_CMD%"=="" (
  echo Python was not found on this system.
  echo Please install Python 3.12 x64 from python.org and check "Add python.exe to PATH".
  pause
  exit /b 1
)

if exist ".venv" (
  .venv\Scripts\python.exe -c "import sys" >nul 2>nul
  if errorlevel 1 (
    echo Detected .venv copied from another computer or corrupted. Re-creating clean environment...
    rmdir /s /q .venv
  )
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating project virtual environment...
  %PY_CMD% -m venv .venv
  if errorlevel 1 (
    echo Failed to create venv with %PY_CMD%. Trying fallback 'python -m venv .venv'...
    python -m venv .venv
    if errorlevel 1 (
      echo Failed to create virtual environment. Ensure Python 3.12 x64 is installed.
      pause
      exit /b 1
    )
  )
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip

echo.
echo Choose PyTorch installation:
echo   1 - NVIDIA GPU (CUDA 12.8 wheel; recommended for RTX 3050 with a compatible recent driver)
echo   2 - CPU only
echo   3 - Skip PyTorch installation (if you already configured it)
set /p TORCH_CHOICE="Enter choice [1/2/3]: "
if "%TORCH_CHOICE%"=="1" (
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  if errorlevel 1 (
    echo.
    echo CUDA wheel install failed. Use https://pytorch.org/get-started/locally/ to select the current supported wheel.
    pause
    exit /b 1
  )
) else if "%TORCH_CHOICE%"=="2" (
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 (
    echo PyTorch CPU install failed.
    pause
    exit /b 1
  )
) else if not "%TORCH_CHOICE%"=="3" (
  echo Unknown choice.
  pause
  exit /b 1
)

echo.
echo Installing application libraries...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. Read the error above.
  pause
  exit /b 1
)

echo.
echo Checking CUDA / GPU...
python gpu_check.py
echo.
echo Setup finished. Run run_windows.bat to start the application.
pause
