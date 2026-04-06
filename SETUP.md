# RECALIX V9 — Setup Guide

> Local AI-powered photo search · Privacy-first · Research-grade

---

## What's New in V9

| Feature | V5 | V9 |
|---|---|---|
| Search backend | Cosine only | **FAISS + Cosine (auto)** |
| Hybrid scoring | Fixed weights | **Fusion model (adaptive)** |
| Training | External scripts | **Built-in (UI + CLI)** |
| Device support | CUDA / CPU | **CUDA + Apple MPS + CPU** |
| Training modes | 1 | **4 (basic/baseline/semi_hard/hard)** |
| Live training log | ❌ | **✅ SSE streaming** |
| UI design | Basic dark | **Glassmorphism dark** |

---

## Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11+ |
| RAM | 4 GB | 8 GB |
| Storage | 2 GB | 5 GB (models + thumbnails) |
| GPU | Optional | CUDA 11.8+ or Apple M1+ |
| Node.js | 18 | 20 (for frontend dev) |

---

## Quick Start

### Windows
```bat
double-click start.bat
```

### Linux / macOS
```bash
chmod +x start.sh
./start.sh
```

Then open: **http://localhost:5173** (dev) or **http://localhost:8000** (production)

---

## Manual Setup

### 1. Backend

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon (M1/M2/M3) — standard PyTorch includes MPS:
pip install torch torchvision

# FAISS (strongly recommended for large collections):
pip install faiss-cpu         # CPU
# pip install faiss-gpu       # GPU (CUDA only)

# Start backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. Frontend (Development)

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### 3. Frontend (Production build)
```bash
cd frontend
npm run build
# Builds into backend/static/ — served automatically by backend
# Then just run the backend: http://localhost:8000
```

---

## First Use

1. Open the app in your browser
2. In the **Photo Indexer** panel, enter your photos directory path
   - Windows: `C:\Users\YourName\Pictures`
   - Linux/Mac: `/home/yourname/Pictures`
3. Click **Index** — wait for indexing to complete
4. Search! Try: *"sunset on beach"*, *"birthday party"*, *"people smiling"*

---

## Training the Fusion Model

The fusion model learns query-adaptive hybrid weights from your own photos.

### Via UI
1. Click **Training** in the top nav
2. Choose a mode (start with **Semi-Hard**)
3. Click **Start Training**
4. Watch live logs in the terminal panel
5. Model auto-loads into search when done

### Via CLI (no backend needed)
```bash
cd training

# Recommended first run:
python train.py --mode semi_hard --epochs 20

# Generate research comparison table:
python train.py --mode baseline --epochs 10

# All modes:
python train.py --mode basic      --epochs 10   # fast
python train.py --mode baseline   --epochs 10   # generates comparison table
python train.py --mode semi_hard  --epochs 20   # RECOMMENDED
python train.py --mode hard       --epochs 30   # highest quality

# Custom settings:
python train.py --mode semi_hard --epochs 30 --lr 0.0005 --batch_size 32

# Custom data path:
python train.py --mode semi_hard --data_dir C:\path\to\backend\data
```

### Training Mode Guide

| Mode | Speed | Quality | Use when |
|---|---|---|---|
| `basic` | Fast | Good | First run, quick test |
| `baseline` | Medium | — | Generating research Table 1 |
| `semi_hard` | Medium | **Best** | Default, production |
| `hard` | Slow | Highest* | After semi_hard converges |

*Hard mode has instability risk. Recommended to run semi_hard first, then hard.

---

## FAISS Index

FAISS provides approximate nearest-neighbor search — much faster than cosine for large collections.

- **Auto-enabled** when you have ≥ 1000 indexed images and FAISS is installed
- Rebuild after indexing new photos:
  - UI: Click **Build FAISS Index** in the indexer panel
  - Or: POST `http://localhost:8000/api/training/build-faiss`

```bash
# Check if FAISS is working:
curl http://localhost:8000/api/health
# Look for: "search_backend": "faiss"
```

---

## Device Configuration

Auto-detection order: **CUDA → Apple MPS → CPU**

Override with environment variable:
```bash
# Force CPU:
RECALIX_DEVICE=cpu python -m uvicorn app.main:app --port 8000

# Force CUDA:
RECALIX_DEVICE=cuda python -m uvicorn app.main:app --port 8000
```

---

## Project Structure

```
recalix_v9/
├── start.bat              Windows launcher
├── start.sh               Linux/Mac launcher
├── backend/
│   ├── app/
│   │   ├── config.py      All configuration (edit this)
│   │   ├── main.py        FastAPI app + startup
│   │   ├── api/           REST endpoints
│   │   │   ├── routes_search.py    GET /api/search
│   │   │   ├── routes_index.py     POST /api/index
│   │   │   ├── routes_files.py     GET /api/thumbnails, POST /api/open-file
│   │   │   ├── routes_system.py    POST /api/select-directory
│   │   │   └── routes_training.py  POST /api/training/start, GET /api/training/stream
│   │   ├── core/
│   │   │   ├── engine.py           CLIP model (CUDA/MPS/CPU)
│   │   │   ├── fusion_model.py     QueryConditionedFusion (148K params)
│   │   │   ├── hybrid.py           Adaptive + fallback hybrid scoring
│   │   │   ├── faiss_search.py     FAISS ANN index
│   │   │   ├── similarity.py       Unified search dispatcher
│   │   │   ├── indexer.py          Recursive directory indexing
│   │   │   ├── embedding_store.py  Memory-mapped .npy storage
│   │   │   ├── ocr.py              EasyOCR text extraction
│   │   │   └── thumbnail.py        Pillow thumbnail generation
│   │   └── db/
│   │       └── database.py         SQLite + WAL mode
│   ├── data/              Created on first run
│   │   ├── metadata.db    SQLite database
│   │   ├── embeddings.npy Memory-mapped embedding store
│   │   ├── faiss.index    FAISS index (built on demand)
│   │   ├── fusion_model.pt Trained model (after training)
│   │   └── thumbnails/    Generated thumbnails
│   ├── static/            Frontend production build (auto-generated)
│   └── requirements.txt
├── training/
│   ├── train.py           Standalone CLI
│   ├── requirements.txt
│   └── modes/
│       ├── base.py        Shared dataset + evaluation
│       ├── basic.py       Random negatives
│       ├── baseline.py    Fixed-weight comparison
│       ├── semi_hard.py   Mixed mining (recommended)
│       └── hard.py        Full hard negative mining
└── frontend/
    ├── src/
    │   ├── pages/
    │   │   ├── SearchPage.tsx   Main search interface
    │   │   └── TrainingPage.tsx Training dashboard
    │   ├── components/
    │   │   ├── SearchBar.tsx
    │   │   ├── FilterSidebar.tsx
    │   │   ├── PhotoGrid.tsx
    │   │   ├── IndexingPanel.tsx
    │   │   └── TrainingPanel.tsx (SSE live logs)
    │   └── lib/
    │       ├── api.ts           All API calls
    │       └── utils.ts
    ├── tailwind.config.ts       Glassmorphism dark theme
    └── package.json
```

---

## API Reference

```
GET  /api/health                  System status + backend info
GET  /api/search?q=...            Natural language search
POST /api/index                   Start indexing directory
GET  /api/index/status            Live indexing progress
DELETE /api/index                 Clear all indexed data
GET  /api/thumbnails/{id}         Serve thumbnail image
POST /api/open-file               Open file in OS
GET  /api/folders                 List all indexed folders
GET  /api/stats                   DB statistics
POST /api/select-directory        Native folder picker dialog
POST /api/training/start          Start training (mode/epochs/lr)
GET  /api/training/status         Training state
GET  /api/training/stream         SSE live log stream
GET  /api/training/fusion-status  Fusion model load state
POST /api/training/build-faiss    Rebuild FAISS index
DELETE /api/training/model        Delete trained model
```

Full interactive docs: **http://localhost:8000/docs**

---

## Research Notes

- **Core finding**: Query-conditioned fusion achieves +2–9% pairwise accuracy over the best fixed-weight baseline
- **3-bucket evaluation**: Queries are classified as visual / OCR / filename — model weights shift measurably between buckets
- **Model checkpoint**: `backend/data/fusion_model.pt` — saved after every epoch improvement
- **Baseline comparison**: Run `python train.py --mode baseline` to generate Table 1 for your paper
- For DTU project reporting: see `3_DAY_EXECUTION_PLAN.md` from V7

---

## Troubleshooting

**Backend won't start**
```bash
# Check Python version
python --version   # needs 3.10+
# Check port
netstat -an | grep 8000   # should be empty before starting
```

**CLIP model download fails**
- First run downloads ~350 MB from Hugging Face — needs internet once
- Cached at `~/.cache/huggingface/`

**EasyOCR slow / using too much memory**
- Disable OCR in `backend/app/config.py`: set `OCR_ENABLED = False`
- Re-index after changing this setting

**FAISS not found**
```bash
pip install faiss-cpu
# Check: python -c "import faiss; print(faiss.__version__)"
```

**GPU not detected**
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"
```

**Training fails immediately**
- Index at least 50+ photos before training
- Ensure `backend/data/embeddings.npy` and `metadata.db` exist
