# RECALIX V10 — Setup Changelog

> Date: 2026-06-04 | Session: Initial Setup & Bug Fixes

---

## Issues Encountered & Fixed

### 1. nvidia-smi Compatibility Fix
**File:** `setup_gpu.bat`

**Problem:** The GPU detection used `nvidia-smi --query-gpu=name --format=csv,noheader,nounits` which is not supported on all nvidia-smi driver versions. This caused the GPU name to be captured as an error string.

**Error Output:**
```
NVIDIA GPU found: ERROR: Option noheader is not recognized.
```

**Fix:** Replaced with `nvidia-smi -L` which is universally supported across all NVIDIA driver versions. The GPU name is parsed from output like `GPU 0: NVIDIA GeForce RTX 5060 Laptop GPU (UUID: ...)`.

**Before:**
```batch
nvidia-smi --query-gpu=name --format=csv,noheader,nounits >nul 2>&1
for /f "tokens=*" %%G in ('nvidia-smi --query-gpu=name --format=csv,noheader,nounits') do (
    set GPU_NAME=%%G
)
```

**After:**
```batch
nvidia-smi -L >nul 2>&1
for /f "tokens=2,* delims=:" %%A in ('nvidia-smi -L') do (
    if not defined GPU_NAME (
        for /f "tokens=* delims= " %%N in ("%%A") do (
            for /f "tokens=1 delims=(" %%T in ("%%N") do (
                set GPU_NAME=%%T
            )
        )
    )
)
```

---

### 2. Pip Self-Upgrade Corruption
**Problem:** Running `pip install --upgrade pip` inside the venv corrupted pip itself. The temp file path for the dist-info INSTALLER was causing a `FileNotFoundError`, and subsequent pip commands failed with `No module named pip`.

**Root Cause:** pip was trying to replace its own files while running, leading to a race condition on Windows with long paths.

**Fix:** Skipped the pip self-upgrade step. The pip bundled with Python 3.12's venv (v24.3.1) works fine for all RECALIX dependencies. The upgrade is cosmetic and not worth the risk of corruption.

---

### 3. Batch Script Syntax Error in PyTorch Installation
**Problem:** The `setup_gpu.bat` had inline Python code inside nested `if` blocks:
```batch
%VPYTHON% -c "import torch; t = torch.zeros(1).cuda(); _ = t+1; print('GPU_OK')"
```
The parentheses `()` in `torch.zeros(1).cuda()` conflicted with batch `if (...) (...)` block syntax, causing `"... was unexpected at this time"` errors.

Additionally, `!` characters at the end of echo statements (like `GPU acceleration active!`) were consumed by `EnableDelayedExpansion`.

**Fix:** Applied in the rewritten setup_gpu.bat by the user's separate version which uses temporary Python script files instead of inline `-c` commands.

---

### 4. PyTorch Download Timeout
**Problem:** The PyTorch CUDA 12.8 wheel is 2.75 GB. The default pip timeout (15 seconds) caused `ReadTimeoutError` at ~73% download progress.

**Fix:** Added explicit timeout and retry flags:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --timeout 120 --retries 10
```

---

## Final Working Setup

| Component | Version | Status |
|---|---|---|
| Python | 3.12.8 | ✅ via py launcher |
| PyTorch | 2.11.0+cu128 | ✅ CUDA enabled |
| CUDA | 12.8 | ✅ RTX 5060 Laptop GPU |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU | ✅ Blackwell architecture |
| open-clip-torch | 3.3.0 | ✅ ViT-B-32 model |
| FastAPI | 0.136.3 | ✅ |
| uvicorn | 0.49.0 | ✅ |
| EasyOCR | 1.7.2 | ✅ |
| FAISS | 1.14.2 (CPU) | ✅ |
| Node.js | 22.20.0 | ✅ |
| npm | 11.7.0 | ✅ |
| Vite | 5.4.21 | ✅ |

---

## How to Start the Application

### Option 1: Using start.bat
```bash
# Double-click start.bat or run from terminal:
start.bat
```

### Option 2: Manual Start
```bash
# Terminal 1 — Backend:
cd backend
..\venv_gpu\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# Terminal 2 — Frontend:
cd frontend
npm run dev
```

### Access Points
- **Frontend UI:** http://localhost:5173
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
