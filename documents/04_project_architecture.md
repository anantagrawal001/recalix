# RECALIX V10 — Project Architecture

> Local AI Photo Search Engine | Created: 2026-06-04

---

## What is RECALIX?

RECALIX is a **local AI-powered photo search engine** that lets you search your personal photo library using natural language queries like "sunset on beach", "invoice from Amazon", or "group photo at party". Everything runs locally — no cloud, no data leaves your machine.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    RECALIX V10                           │
│                                                          │
│  ┌─────────────┐          ┌──────────────────────────┐   │
│  │  Frontend    │  HTTP    │  Backend (FastAPI)       │   │
│  │  React +     │◄────────►│                          │   │
│  │  TypeScript  │  :5173   │  ┌────────────────────┐  │   │
│  │  Vite        │          │  │ CLIP (ViT-B-32)    │  │   │
│  │  TailwindCSS │          │  │ 151M params, frozen│  │   │
│  └─────────────┘          │  └────────────────────┘  │   │
│                            │  ┌────────────────────┐  │   │
│                            │  │ Fusion Model       │  │   │
│                            │  │ 173K params,trained│  │   │
│                            │  └────────────────────┘  │   │
│                            │  ┌────────────────────┐  │   │
│                            │  │ FAISS Index        │  │   │
│                            │  │ 233K vectors       │  │   │
│                            │  └────────────────────┘  │   │
│                            │  ┌────────────────────┐  │   │
│                            │  │ SQLite DB          │  │   │
│                            │  │ metadata.db        │  │   │
│                            │  └────────────────────┘  │   │
│                            │          :8000           │   │
│                            └──────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Training Module                                  │    │
│  │  Trains fusion model on your indexed photos       │    │
│  │  Modes: basic | semi_hard | hard                  │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
recalix_v10_v2/
├── backend/                    # Python FastAPI backend
│   ├── app/
│   │   ├── main.py            # FastAPI app entry point, lifespan events
│   │   ├── config.py          # All configuration (paths, model, search params)
│   │   ├── api/               # REST API routes
│   │   │   ├── routes_search.py    # GET /api/search — AI search
│   │   │   ├── routes_index.py     # POST /api/index — index photos
│   │   │   ├── routes_files.py     # thumbnails, gallery, folders, open-file
│   │   │   ├── routes_system.py    # system info
│   │   │   └── routes_training.py  # training management API
│   │   ├── core/              # Core AI/ML logic
│   │   │   ├── engine.py          # CLIP model loading, embed_text(), embed_image()
│   │   │   ├── similarity.py      # Cosine/FAISS search dispatch
│   │   │   ├── hybrid.py          # Hybrid scoring with fusion model
│   │   │   ├── fusion_model.py    # QueryConditionedFusion neural network
│   │   │   ├── indexer.py         # Photo indexing pipeline
│   │   │   ├── embedding_store.py # Numpy embedding storage
│   │   │   ├── faiss_search.py    # FAISS vector index
│   │   │   ├── ocr.py            # EasyOCR text extraction
│   │   │   └── thumbnail.py      # Thumbnail generation
│   │   └── db/
│   │       └── database.py        # SQLite database operations
│   ├── data/                  # Runtime data (created at runtime)
│   │   ├── metadata.db            # SQLite: file paths, OCR text, metadata
│   │   ├── embeddings.npy         # CLIP image embeddings (233K × 512)
│   │   ├── faiss.index            # FAISS vector search index
│   │   ├── fusion_model.pt        # Trained fusion model checkpoint
│   │   └── thumbnails/            # JPEG thumbnails for UI
│   ├── static/                # Production frontend build (optional)
│   └── requirements.txt       # Python dependencies
│
├── frontend/                  # React + TypeScript frontend
│   ├── src/
│   │   └── pages/
│   │       └── TrainingPage.tsx   # Training management UI
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.ts
│
├── training/                  # Training module
│   ├── modes/
│   │   ├── base.py               # Shared: dataset, evaluation, query extraction
│   │   ├── basic.py              # Random negative training
│   │   ├── semi_hard.py          # Semi-hard negative training (recommended)
│   │   ├── hard.py               # Hard negative mining
│   │   └── baseline.py           # Fixed-weight baselines for comparison
│   └── artifacts/                # Training outputs (learning curves, metrics)
│
├── documents/                 # Project documentation (you are here!)
├── venv_gpu/                  # Python virtual environment
├── start.bat                  # Launch both servers (Windows)
├── start.sh                   # Launch both servers (macOS/Linux)
├── setup_gpu.bat              # One-time setup: auto-detect GPU + install deps
├── recalix.exe                # Standalone launcher (optional)
└── recalix.py                 # Python launcher script
```

---

## Search Flow (How a Query Works)

```
User types: "sunset on beach"
        ↓
1. CLIP encodes query → 512-dim text embedding
        ↓
2. FAISS searches 233K image embeddings → top 100 candidates
        ↓
3. For each candidate, compute 4 signals:
   • visual: CLIP similarity (text ↔ image)
   • ocr: word overlap with OCR text in image
   • filename: word overlap with filename
   • structural: document/screenshot heuristics
        ↓
4. QueryConditionedFusion(query_embedding) → [0.82, 0.05, 0.08, 0.05]
        ↓
5. final_score = 0.82×visual + 0.05×ocr + 0.08×filename + 0.05×structural
        ↓
6. Re-rank by final_score → return top results
```

---

## Key Technologies

| Component | Technology | Purpose |
|---|---|---|
| **AI Model** | OpenAI CLIP ViT-B-32 | Image & text embeddings |
| **Vector Search** | FAISS | Fast approximate nearest neighbor search |
| **OCR** | EasyOCR | Extract text from images |
| **Backend** | FastAPI + uvicorn | REST API server |
| **Frontend** | React + TypeScript + Vite | Web UI |
| **Styling** | TailwindCSS | UI styling |
| **Database** | SQLite | Metadata storage |
| **Training** | PyTorch + CUDA | Fusion model training |
| **GPU** | NVIDIA CUDA 12.8 | GPU acceleration |

---

## Current System Stats

| Metric | Value |
|---|---|
| Indexed images | 233,192 |
| Embeddings | 233,199 (512-dim each) |
| Unique folders | 221 |
| Search backend | FAISS |
| Fusion model | Active (semi_hard trained) |
| Training accuracy | 94.2% recall@5 |
| CLIP model | ViT-B-32 (openai) |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| PyTorch | 2.11.0+cu128 |
