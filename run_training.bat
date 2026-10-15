@echo off
setlocal
title RECALIX Training

set ROOT=%~dp0
set VENV=%ROOT%venv_gpu
set TRAINING=%ROOT%training

if not exist "%VENV%\Scripts\python.exe" (
    echo [ERROR] venv_gpu not found. Run setup_gpu.bat first.
    pause & exit /b 1
)

set PYTHON="%VENV%\Scripts\python.exe"
cd /d "%TRAINING%"

echo.
echo  RECALIX Training
echo  ================
echo  Modes: basic, baseline, semi_hard, hard
echo.

set /p MODE="Mode (Enter = semi_hard): "
if "%MODE%"=="" set MODE=semi_hard

set /p EPOCHS="Epochs (Enter = 20): "
if "%EPOCHS%"=="" set EPOCHS=20

echo.
echo  Starting %MODE% for %EPOCHS% epochs...
echo.

%PYTHON% train.py --mode %MODE% --epochs %EPOCHS%

echo.
echo  Done. Restart start.bat to load the new model.
pause
