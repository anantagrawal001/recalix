@echo off
setlocal
title RECALIX V10

:: ALWAYS run from script directory (critical)
cd /d %~dp0

set ROOT=%cd%
set BACKEND=%ROOT%\backend
set FRONTEND=%ROOT%\frontend
set VENV=%ROOT%\venv_gpu

echo =========================================
echo   RECALIX V10 - START
echo =========================================
echo ROOT: %ROOT%
echo VENV: %VENV%
echo.

:: ---- FORCE venv activation (no fragile detection) ----
if exist "%VENV%\Scripts\activate.bat" (
    echo [GPU] Activating venv_gpu...
    call "%VENV%\Scripts\activate.bat"
) else (
    echo [ERROR] venv_gpu not found.
    echo Run setup_gpu.bat first.
    pause
    exit /b 1
)

:: ---- VERIFY CUDA (non-negotiable) ----
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python/CUDA check failed
    pause
    exit /b 1
)

echo.

:: ---- Backend setup ----
cd /d "%BACKEND%"

if not exist "data" mkdir data
if not exist "data\thumbnails" mkdir data\thumbnails

echo [Start] Backend ^> http://localhost:8000
start "RECALIX-Backend" cmd /k "%VENV%\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

timeout /t 4 >nul

:: ---- Frontend ----
node --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    if not exist "%FRONTEND%\node_modules" (
        echo [Setup] Installing frontend deps...
        cd /d "%FRONTEND%"
        call npm install
        cd /d "%BACKEND%"
    )

    echo [Start] Frontend ^> http://localhost:5173
    start "RECALIX-Frontend" cmd /k "cd /d "%FRONTEND%" && npm run dev"

    timeout /t 2 >nul
    echo.
    echo Open: http://localhost:5173
) else (
    echo Open: http://localhost:8000
)

echo.
echo Press any key to stop all servers...
pause >nul

taskkill /FI "WindowTitle eq RECALIX-Backend*" /F >nul 2>&1
taskkill /FI "WindowTitle eq RECALIX-Frontend*" /F >nul 2>&1

echo Stopped.
