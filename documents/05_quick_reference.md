# RECALIX V10 — Quick Reference

> Cheat sheet for common operations

---

## Starting the Application

### Using start.bat (recommended)
```bash
# Just double-click start.bat, or:
cd recalix_v10_v2
start.bat
```

### Manual Start
```bash
# Terminal 1 — Backend
cd backend
..\venv_gpu\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev
```

### Access
- **UI:** http://localhost:5173
- **API:** http://localhost:8000
- **Docs:** http://localhost:8000/docs

---

## Setup (First Time Only)

```bash
# Run the one-time setup:
setup_gpu.bat

# This will:
# 1. Find Python 3.10-3.12
# 2. Detect GPU (NVIDIA/CPU)
# 3. Create venv_gpu
# 4. Install PyTorch (CUDA or CPU)
# 5. Install all dependencies
# 6. Verify installation
```

---

## Training

### From the UI
1. Open http://localhost:5173
2. Go to Training page
3. Select mode (semi_hard recommended)
4. Click Start Training

### Training Modes
| Mode | Use When |
|---|---|
| `basic` | First time, small dataset |
| `semi_hard` ⭐ | Default — best balance |
| `hard` | Large diverse dataset, after semi_hard plateaus |

---

## Useful API Endpoints

```bash
# Health check
curl http://localhost:8000/api/health

# Search
curl "http://localhost:8000/api/search?q=sunset&top_k=10"

# Stats
curl http://localhost:8000/api/stats

# Folders
curl http://localhost:8000/api/folders

# Gallery
curl "http://localhost:8000/api/gallery?page=1&page_size=20"

# Training status
curl http://localhost:8000/api/training/status

# Search history
curl "http://localhost:8000/api/search/history?limit=5"
```

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `RECALIX_DEVICE` | auto | Force device: `cpu`, `cuda`, `mps` |
| `RECALIX_SEARCH_BACKEND` | `auto` | Search backend: `cosine`, `faiss`, `auto` |
| `RECALIX_FUSION_ENABLED` | `true` | Enable/disable fusion model |
| `RECALIX_TRAIN_LIMIT` | 0 (unlimited) | Cap training images per split |
| `RECALIX_NEGATIVE_SAMPLE` | 0 (all) | Cap negative pool size |
| `RECALIX_DISABLE_EXAMPLE_CACHE` | unset | Force rebuild training examples |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `No module named pip` | Delete `venv_gpu`, re-run `setup_gpu.bat` |
| `CUDA not available` | Check `nvidia-smi`, reinstall PyTorch with correct CUDA |
| Port 8000 in use | `netstat -ano | findstr :8000` then kill the PID |
| Port 5173 in use | `netstat -ano | findstr :5173` then kill the PID |
| Search returns 0 results | Index photos first via the UI |
| Training accuracy stuck at 0.5 | Need more diverse photos (not just screenshots) |
