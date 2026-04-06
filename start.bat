@echo off
setlocal EnableDelayedExpansion

echo.
echo  ██████╗ ███████╗ ██████╗ █████╗ ██╗     ██╗██╗  ██╗
echo  ██╔══██╗██╔════╝██╔════╝██╔══██╗██║     ██║╚██╗██╔╝
echo  ██████╔╝█████╗  ██║     ███████║██║     ██║ ╚███╔╝
echo  ██╔══██╗██╔══╝  ██║     ██╔══██║██║     ██║ ██╔██╗
echo  ██║  ██║███████╗╚██████╗██║  ██║███████╗██║██╔╝ ██╗
echo  ╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═╝
echo                         V9  —  Local AI Photo Search
echo.

set ROOT=%~dp0
set BACKEND=%ROOT%backend
set FRONTEND=%ROOT%frontend
set PY=%ROOT%venv\Scripts\python.exe

:: ── Check Python ─────────────────────────────────────────────────────────────
"%PY%" --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Install Python 3.10+ from https://python.org
    pause & exit /b 1
)

:: ── Check Node ────────────────────────────────────────────────────────────────
node --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] Node.js not found. Frontend dev server will not start.
    echo        The backend will still serve a pre-built frontend if available.
    goto :start_backend
)

:: ── Install frontend deps if needed ──────────────────────────────────────────
if not exist "%FRONTEND%\node_modules" (
    echo [Setup] Installing frontend dependencies...
    cd /d "%FRONTEND%"
    call npm install
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] npm install failed.
        pause & exit /b 1
    )
)

:: ── Install backend deps if needed ───────────────────────────────────────────
:start_backend
cd /d "%BACKEND%"

if not exist "data" mkdir data
if not exist "data\thumbnails" mkdir data\thumbnails

echo [Setup] Checking Python dependencies...
"%PY%" -m pip install -r requirements.txt --quiet

echo.
echo taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1
echo [Start] Launching RECALIX backend on http://localhost:8000
echo         Press Ctrl+C to stop.
echo.

:: ── Launch backend ────────────────────────────────────────────────────────────
start "RECALIX Backend" /B "%PY%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
:: Wait for backend to start
timeout /t 3 /nobreak >nul

:: ── Launch frontend dev server (if Node available) ───────────────────────────
node --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    if exist "%FRONTEND%\node_modules" (
        echo [Start] Launching frontend dev server on http://localhost:5173
        start "RECALIX Frontend" /B cmd /c "cd /d "%FRONTEND%" && npm run dev"
        timeout /t 2 /nobreak >nul
        echo.
        echo  Open in browser: http://localhost:5173
    )
) else (
    echo.
    echo  Open in browser: http://localhost:8000
)

echo.
echo  Both servers running. Press any key to stop all...
pause >nul

:: Kill both servers
taskkill /FI "WindowTitle eq RECALIX Backend*" /F >nul 2>&1
taskkill /FI "WindowTitle eq RECALIX Frontend*" /F >nul 2>&1
echo Stopped.
