@echo off
setlocal
title RECALIX Project Agent

cd /d %~dp0

set PYTHON=%cd%\venv_gpu\Scripts\python.exe
if not exist "%PYTHON%" set PYTHON=python

echo =========================================
echo   RECALIX Project Agent
echo =========================================
echo Watches code, jobs, status files, and build health.
echo Logs: backend\data\agent_logs\project_agent.log
echo.

"%PYTHON%" tools\project_agent.py --interval 30 --build-interval 300

pause
