# RECALIX V9 — Setup Guide

## Quick Start

**Windows:** double-click `start.bat`  
**Linux/Mac:** `chmod +x start.sh && ./start.sh`

Then open: **http://localhost:5173** (dev) or **http://localhost:8000** (production)

---

## Manual Setup

### Backend
```bash
cd backend
pip install -r requirements.txt

# GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Start
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend (dev)
```bash
cd frontend
npm install
npm run dev          # → http://localhost:5173
```

### Frontend (production build)
```bash
cd frontend
npm run build        # builds into backend/static/
# Then just run the backend — it serves the UI at http://localhost:8000
```

---

## First Use

1. Open the app
2. Enter your photos directory path in the **Photo Indexer** panel
   - Windows: `C:\Users\YourName\Pictures`
   - Linux/Mac: `/home/yourname/Pictures`
3. Click **Index** and wait
4. Search: *"sunset"*, *"birthday party"*, *"receipt"*

---

## Training the Fusion Model

### From the UI
1. Click **Training** in the nav bar
2. Select **Semi-Hard** mode (recommended)
3. Click **Start Training** — watch live logs in the terminal

### From CLI (recommended for long runs)
```bash
cd training

# Recommended first run:
python train.py --mode semi_hard --epochs 20

# Generate research comparison table (Table 1 for paper):
python train.py --mode baseline --epochs 10

# All modes:
python train.py --mode basic      --epochs 10   # fast
python train.py --mode baseline   --epochs 10   # research table
python train.py --mode semi_hard  --epochs 20   # RECOMMENDED
python train.py --mode hard       --epochs 30   # after semi_hard converges

# Custom data path (if running from a different directory):
python train.py --mode semi_hard --data_dir C:\path\to\backend\data
```

---

## Training Mode Guide

| Mode | Speed | Use when |
|---|---|---|
| `basic` | Fast | First test, verify pipeline works |
| `baseline` | Medium | Generating research Table 1 for paper |
| `semi_hard` | Medium | **Default — best quality/stability tradeoff** |
| `hard` | Slow | After semi_hard converges, for final model |

---

## What Changed vs V5

| Feature | V5 | V9 |
|---|---|---|
| Fusion weights | Fixed (0.55/0.30/0.10/0.05) | **Adaptive per query** |
| FAISS search | ❌ | ✅ Auto-enabled ≥1000 images |
| Training | External script only | **UI + CLI** |
| Live training log | ❌ | ✅ SSE streaming |
| Device support | CUDA/CPU | **CUDA + Apple MPS + CPU** |

---

## Troubleshooting

**Backend won't start**
```bash
python --version      # needs 3.10+
netstat -an | grep 8000   # must be free
```

**CLIP model download fails** — needs internet on first run (~350 MB from HuggingFace)

**FAISS not found**
```bash
pip install faiss-cpu
python -c "import faiss; print('ok')"
```

**GPU not detected**
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"
```

**Training — "No embeddings found"** — Index photos first, then train.

**Training — loss stays at 0** — Fixed in V9 (was a bug where `.mean()` before `ranking_loss` collapsed all gradient signal).

---

## API Reference

```
GET  /api/health                  System status
GET  /api/search?q=...            Semantic search
POST /api/index                   Start indexing
GET  /api/index/status            Live index progress
GET  /api/folders                 List indexed folders
GET  /api/stats                   DB stats
POST /api/open-file               Open file in OS
POST /api/training/start          Start training
GET  /api/training/stream         SSE live log
GET  /api/training/fusion-status  Model load state
POST /api/training/build-faiss    Rebuild FAISS index
DELETE /api/training/model        Delete trained model
```

Interactive docs: **http://localhost:8000/docs**
