# RECALIX V10 — Local AI Photo Search

Privacy-first semantic image search. Runs 100% on your computer. No cloud. No data leaves your machine.

## Quick Start

### Windows
```
1. Double-click setup_gpu.bat     ← one-time setup (installs PyTorch for your GPU)
2. Double-click start.bat         ← launch the app
3. Open http://localhost:5173
```

### Linux / macOS
```bash
chmod +x start.sh && ./start.sh
```

---

## First Use

1. Open the app at **http://localhost:5173**
2. In the **Photo Indexer**, paste your photos directory path and click **Index**
3. Wait for indexing to complete (GPU: ~200-400 img/min, CPU: ~20-40 img/min)
4. Search with natural language: *"sunset on beach"*, *"birthday party"*, *"receipt"*

---

## GPU Support

| GPU | Architecture | PyTorch needed | setup_gpu.bat |
|-----|-------------|----------------|---------------|
| RTX 5000 series | Blackwell (sm_120) | 2.7 + CUDA 12.8 | ✅ Auto-detects |
| RTX 4000 series | Ada (sm_89) | ≥ 2.0 + CUDA 12.1 | ✅ Auto-detects |
| RTX 3000 series | Ampere (sm_86) | ≥ 1.9 + CUDA 12.1 | ✅ Auto-detects |
| Apple M1/M2/M3 | MPS | Standard PyTorch | ✅ Auto-detects |
| CPU only | — | Standard PyTorch | ✅ Auto-detects |

---

## Training the Fusion Model

The fusion model learns query-adaptive scoring weights from YOUR indexed photos.

### From the UI
1. Click **Training** in the nav bar
2. Choose **Semi-Hard** mode (recommended)
3. Click **Start Training** — watch live logs
4. Training artifacts are saved to `backend/data/training_logs/`

### From CLI (recommended for long runs)
```bash
cd training

# Recommended:
python train.py --mode semi_hard --epochs 20

# For M.Tech paper Table 1:
python train.py --mode baseline --epochs 10

# All modes:
python train.py --mode basic      --epochs 10   # fast verification
python train.py --mode baseline   --epochs 10   # research comparison table
python train.py --mode semi_hard  --epochs 20   # RECOMMENDED
python train.py --mode hard       --epochs 30   # after semi_hard converges
```

### Does training on the same dataset again help?
Yes — each training run starts from a fresh model. Running multiple times with different modes
lets you compare results. The model that achieves the highest val_acc is your best model.
Training artifacts in `backend/data/training_logs/` track every session.

---

## Search — How It Works

Each search goes through this pipeline:

```
Query text
    ↓
CLIP text encoder → 512-dim embedding
    ↓
Cosine / FAISS search → top-K candidate images
    ↓
Hybrid re-scoring (4 signals):
  • Visual:     CLIP text-image similarity
  • OCR:        Keyword match in extracted text
  • Filename:   Token match in filename
  • Structural: Document/screenshot heuristics
    ↓
Weights predicted by fusion model (or fixed fallback)
    ↓
Final ranked results
```

**Backend badge on results:**
- `cosine` — NumPy dot product search (default, exact)
- `faiss`  — Approximate nearest-neighbor (faster for 1000+ images)

The fusion model shows `(device=cuda)` when your GPU is being used for weight prediction.

---

## Folder Indexing

You can index multiple directories — just run the indexer for each one.
Use the **Folders** filter in the sidebar to search within a specific folder.

---

## File Structure

```
recalix_v10/
├── start.bat / start.sh          Launch the app
├── setup_gpu.bat                 One-time GPU setup (Windows)
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── engine.py         CLIP model (GPU auto-detection + CPU fallback)
│   │   │   ├── fusion_model.py   Query-Conditioned Fusion MLP
│   │   │   ├── hybrid.py         4-signal adaptive scoring
│   │   │   ├── similarity.py     Cosine / FAISS dispatcher
│   │   │   ├── indexer.py        Recursive directory indexer
│   │   │   └── embedding_store.py Memory-mapped embedding storage
│   │   └── data/
│   │       ├── metadata.db       SQLite — all media metadata
│   │       ├── embeddings.npy    Memory-mapped CLIP embeddings
│   │       ├── fusion_model.pt   Trained fusion model (after training)
│   │       ├── faiss.index       FAISS ANN index (after building)
│   │       └── training_logs/    Training artifact JSONs
│   └── requirements.txt
├── training/
│   ├── train.py                  CLI training script
│   └── modes/
│       ├── basic.py              Random negatives
│       ├── baseline.py           Fixed-weight comparison + Table 1
│       ├── semi_hard.py          Mixed mining (RECOMMENDED)
│       └── hard.py               Full hard negative mining
└── frontend/                     React + Vite + Tailwind UI
```

---

## API Reference

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | System status, GPU info, search backend |
| `GET /api/search?q=...` | Semantic search |
| `POST /api/index` | Start indexing a directory |
| `GET /api/index/status` | Live indexing progress |
| `GET /api/folders` | List indexed folders |
| `GET /api/stats` | Database statistics |
| `POST /api/training/start` | Start model training |
| `GET /api/training/stream` | SSE live training log |
| `GET /api/training/artifacts` | List training sessions |
| `POST /api/training/build-faiss` | Build FAISS index |

Interactive docs: **http://localhost:8000/docs**

---

## Troubleshooting

**GPU not working after setup_gpu.bat**
- Verify with: `venv_gpu\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"`
- RTX 5000 series needs `cu128` — setup_gpu.bat handles this automatically

**Search returns bad results**
- Index more photos (50+ minimum for meaningful results)
- Run training after indexing to improve fusion weights
- Use the `Baseline` training mode to benchmark your results

**Training artifact saved but model not loading**
- Check `backend/data/fusion_model.pt` exists
- Restart the backend — it loads the model at startup
- Or: DELETE `/api/training/model` then POST `/api/training/start` again to force reload

**"Fusion predict failed" in logs**
- Fixed in this version — was a GPU/CPU device mismatch
- If still happening, restart the backend
