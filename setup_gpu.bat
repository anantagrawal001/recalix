@echo off
setlocal EnableDelayedExpansion
title RECALIX V10 — Universal Setup
color 0B

echo.
echo  =========================================================
echo   RECALIX V10 — Universal Setup
echo  =========================================================
echo.
echo  This script auto-detects your hardware and installs
echo  the correct version of PyTorch (GPU or CPU).
echo.

set ROOT=%~dp0
set BACKEND=%ROOT%backend
set VENV=%ROOT%venv_gpu

:: ── Step 1: Find Python ────────────────────────────────────────────────────
echo [Step 1] Finding Python 3.10, 3.11, or 3.12...
echo.

set PY=

:: Check py launcher (most reliable on Windows)
for %%V in (3.12 3.11 3.10) do (
    py -%%V --version >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set PY=py -%%V
        echo   Found: Python %%V via py launcher
        goto :found_py
    )
)

:: Check common paths
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
    "C:\Program Files\Python312\python.exe"
    "C:\Program Files\Python311\python.exe"
    "C:\Program Files\Python310\python.exe"
) do (
    if exist %%P (
        set PY=%%P
        echo   Found: %%P
        goto :found_py
    )
)

echo.
echo  *** Python 3.10+ not found ***
echo.
echo  Please install Python 3.11 from:
echo  https://www.python.org/downloads/
echo.
echo  IMPORTANT: Check "Add Python to PATH" during install!
echo.
pause & exit /b 1

:found_py
echo.

:: ── Step 2: Detect GPU ─────────────────────────────────────────────────────
echo [Step 2] Detecting GPU...
echo.

set HAS_GPU=0
set GPU_NAME=
nvidia-smi -L >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=2,* delims=:" %%A in ('nvidia-smi -L 2^>nul') do (
        if not defined GPU_NAME (
            for /f "tokens=* delims= " %%N in ("%%A") do (
                for /f "tokens=1 delims=(" %%T in ("%%N") do (
                    set GPU_NAME=%%T
                )
            )
        )
    )
    if defined GPU_NAME (
        set HAS_GPU=1
        echo   NVIDIA GPU found: !GPU_NAME!
    )
)

if !HAS_GPU! EQU 0 (
    echo   No NVIDIA GPU detected — will install CPU-only PyTorch
    echo   [This is fine! RECALIX works on CPU, just slower]
)
echo.

:: ── Step 3: Create venv ────────────────────────────────────────────────────
echo [Step 3] Preparing virtual environment...

if exist "%VENV%" (
    echo   Removing old environment...
    rmdir /s /q "%VENV%" 2>nul
)

echo   Creating fresh virtual environment...
%PY% -m venv "%VENV%"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create venv. Try running as Administrator.
    pause & exit /b 1
)
echo   venv_gpu created.
echo.

set VPYTHON="%VENV%\Scripts\python.exe"
set VPIP="%VENV%\Scripts\pip.exe"

:: Upgrade pip
%VPYTHON% -m pip install --upgrade pip -q

:: ── Step 4: Install PyTorch ────────────────────────────────────────────────
echo [Step 4] Installing PyTorch...
echo.

set TORCH_INSTALLED=0

if !HAS_GPU! EQU 1 (
    echo   GPU detected — trying CUDA versions...
    echo.
    
    :: Try cu128 first (newest, supports Blackwell/sm_120)
    echo   Trying cu128 (supports RTX 50xx series)...
    %VPIP% install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -q
    if !ERRORLEVEL! EQU 0 (
        :: Verify GPU kernel works
        %VPYTHON% -c "import torch; t = torch.zeros(1).cuda(); _ = t+1; print('GPU_OK')" 2>nul | findstr "GPU_OK" >nul
        if !ERRORLEVEL! EQU 0 (
            echo   ✓ PyTorch + cu128 installed — GPU acceleration active!
            set TORCH_INSTALLED=1
            goto :torch_done
        ) else (
            echo   cu128 kernel test failed, trying cu126...
            %VPIP% uninstall torch torchvision torchaudio -y -q 2>nul
        )
    )
    
    :: Try cu126
    echo   Trying cu126...
    %VPIP% install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 -q
    if !ERRORLEVEL! EQU 0 (
        %VPYTHON% -c "import torch; t = torch.zeros(1).cuda(); _ = t+1; print('GPU_OK')" 2>nul | findstr "GPU_OK" >nul
        if !ERRORLEVEL! EQU 0 (
            echo   ✓ PyTorch + cu126 installed — GPU acceleration active!
            set TORCH_INSTALLED=1
            goto :torch_done
        ) else (
            echo   cu126 kernel test failed, trying cu121...
            %VPIP% uninstall torch torchvision torchaudio -y -q 2>nul
        )
    )
    
    :: Try cu121
    echo   Trying cu121...
    %VPIP% install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
    if !ERRORLEVEL! EQU 0 (
        %VPYTHON% -c "import torch; t = torch.zeros(1).cuda(); _ = t+1; print('GPU_OK')" 2>nul | findstr "GPU_OK" >nul
        if !ERRORLEVEL! EQU 0 (
            echo   ✓ PyTorch + cu121 installed — GPU acceleration active!
            set TORCH_INSTALLED=1
            goto :torch_done
        ) else (
            echo   No CUDA version worked — falling back to CPU
            %VPIP% uninstall torch torchvision torchaudio -y -q 2>nul
        )
    )
)

:: CPU fallback
if !TORCH_INSTALLED! EQU 0 (
    echo   Installing CPU-only PyTorch...
    %VPIP% install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to install PyTorch.
        pause & exit /b 1
    )
    echo   ✓ PyTorch installed (CPU mode)
    set TORCH_INSTALLED=1
)

:torch_done
echo.

:: ── Step 5: Install dependencies ───────────────────────────────────────────
echo [Step 5] Installing RECALIX dependencies...
%VPIP% install fastapi "uvicorn[standard]" pydantic open-clip-torch numpy Pillow easyocr faiss-cpu python-multipart aiofiles -q
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies.
    pause & exit /b 1
)
echo   ✓ All dependencies installed
echo.

:: ── Step 6: Verify ─────────────────────────────────────────────────────────
echo [Step 6] Verifying installation...
echo.

echo import torch > "%VENV%\test_setup.py"
echo import open_clip >> "%VENV%\test_setup.py"
echo print('  PyTorch     :', torch.__version__) >> "%VENV%\test_setup.py"
echo print('  CUDA        :', torch.cuda.is_available()) >> "%VENV%\test_setup.py"
echo if torch.cuda.is_available(): >> "%VENV%\test_setup.py"
echo     print('  GPU         :', torch.cuda.get_device_name(0)) >> "%VENV%\test_setup.py"
echo     cap = torch.cuda.get_device_capability(0) >> "%VENV%\test_setup.py"
echo     print('  Compute cap : sm_%%d%%d' %% cap) >> "%VENV%\test_setup.py"
echo print() >> "%VENV%\test_setup.py"
echo print('Setup verification PASSED!') >> "%VENV%\test_setup.py"

%VPYTHON% "%VENV%\test_setup.py"
del "%VENV%\test_setup.py" 2>nul

echo.
echo  =========================================================
echo   Setup complete!
echo.
echo   To start RECALIX:
echo     - Double-click recalix.exe
echo     - Or run: start.bat
echo  =========================================================
echo.
pause
