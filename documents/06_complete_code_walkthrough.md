# RECALIX V10 — Every File Explained (For Examiner)

> This document explains **every single file and folder** in the project — executables, batch scripts, Python files, JSON files, database, virtual environment, configs, frontend, training logs — with WHY each exists and WHAT it does.

---

## Table of Contents

1. [Root Directory — Launch & Setup Files](#1-root-directory--launch--setup-files)
2. [Backend — Python API Application](#2-backend--python-api-application)
3. [Backend Data — Generated Runtime Files](#3-backend-data--generated-runtime-files)
4. [Frontend — React UI Application](#4-frontend--react-ui-application)
5. [Training — ML Training Module](#5-training--ml-training-module)
6. [Tools — Automation Scripts](#6-tools--automation-scripts)
7. [Documentation & Config Files](#7-documentation--config-files)
8. [Virtual Environment (venv_gpu)](#8-virtual-environment-venv_gpu)
9. [Complete Data Flow](#9-complete-data-flow)
10. [Examiner Q&A Reference](#10-examiner-qa-reference)

---

## 1. Root Directory — Launch & Setup Files

### 📦 `recalix.exe` (5.5 MB) — Compiled Launcher Executable

**Why it exists:** So the user can double-click ONE file to set up and run the entire application without knowing Python or command line. Built using PyInstaller from `recalix.py`.

**What it does:**
1. Detects Python 3.10–3.12 on the system
2. Scans GPU hardware (NVIDIA, AMD, Intel) via Windows WMI
3. Creates a virtual environment if first run
4. Installs the correct PyTorch build for the user's GPU
5. Starts the backend server on port 8000
6. Opens the browser to the app

**Why it's important:** Makes the project a "plug and play" desktop application. Users don't need to install anything manually — just double-click.

---

### 🐍 `recalix.py` (40 KB, 946 lines) — Launcher Source Code

**Why it exists:** The Python source code that gets compiled into `recalix.exe`. Also usable directly: `python recalix.py`.

**What it does (detailed):**

```
find_python()          → Searches for Python via py launcher, PATH, common install dirs
discover_hardware()    → WMI GPU scan + nvidia-smi for architecture/compute capability
install_pytorch()      → Tries CUDA wheels in priority order with kernel verification
main()                 → Orchestrates: find Python → scan GPU → create venv → install → launch
```

**Key design — GPU auto-detection:**
- Maps NVIDIA compute capability to architecture: `sm_120 → Blackwell → cu128`
- Tests each CUDA wheel by actually running `torch.zeros(1).cuda()` — not just checking imports
- Falls back gracefully: CUDA → DirectML (AMD) → Intel XPU → CPU

---

### 📜 `setup_gpu.bat` (7.7 KB) — One-Time GPU Setup Script

**Why it exists:** Alternative to `recalix.exe` for manual setup. Useful for debugging GPU detection or if the exe has issues.

**What it does:**
1. Detects if NVIDIA GPU is present via `nvidia-smi`
2. Reads compute capability to pick the right CUDA wheel
3. Creates `venv_gpu` virtual environment
4. Installs PyTorch + all dependencies
5. Verifies GPU kernel execution

---

### 📜 `start.bat` (1.9 KB) — Application Launch Script

**Why it exists:** Quick launcher after initial setup is done. Starts both backend and frontend servers.

**What it does:**
1. Activates `venv_gpu`
2. Verifies CUDA is working
3. Starts backend: `uvicorn app.main:app --port 8000` (in a new console window)
4. Starts frontend: `npm run dev` (Vite dev server on port 5173)
5. Waits for keypress, then kills both servers

---

### 📜 `start.sh` (2.3 KB) — Linux/Mac Launch Script

**Why it exists:** Same as `start.bat` but for Linux and macOS. Shell script equivalent.

---

### 📜 `run_training.bat` (727 bytes) — Manual Training Launcher

**Why it exists:** Quick way to start training from command line without the web UI.

**What it does:**
1. Prompts user for training mode (basic/baseline/semi_hard/hard)
2. Prompts for number of epochs
3. Runs `training/train.py` with the venv Python

---

### 📜 `run_project_agent.bat` (464 bytes) — Project Health Monitor

**Why it exists:** Launches the automated project agent that watches system health.

**What it does:** Runs `tools/project_agent.py` which monitors code changes, build health, and indexing/training job status.

---

### 🐍 `test.py` (84 bytes) — Quick Test Script

**Why it exists:** Simple test to verify Python environment works. Used during development.

---

## 2. Backend — Python API Application

### 📂 `backend/` — The Server

Everything under this folder is the backend Python application that handles AI processing, search, indexing, and API endpoints.

---

### 📄 `backend/requirements.txt` (2.1 KB) — Python Dependencies List

**Why it exists:** Declares all Python packages needed by the backend. Used by `pip install -r requirements.txt`.

**Packages listed:**
| Package | Version | Why Needed |
|---------|---------|------------|
| `fastapi` | ≥0.115 | Web API framework — serves all REST endpoints |
| `uvicorn[standard]` | ≥0.30 | ASGI server — runs the FastAPI application |
| `pydantic` | ≥2.0 | Data validation for API request/response models |
| `open-clip-torch` | ≥2.24 | CLIP model — converts images and text to embeddings |
| `torch` | ≥2.1 | PyTorch — GPU-accelerated deep learning framework |
| `torchvision` | ≥0.16 | Image processing utilities for PyTorch |
| `numpy` | ≥1.24 | Numerical computing — embedding math, matrix operations |
| `Pillow` | ≥10.0 | Image loading, resizing, thumbnail generation |
| `easyocr` | ≥1.7 | Optical Character Recognition — reads text from images |
| `faiss-cpu` | ≥1.8 | Fast Approximate Nearest Neighbor search for vectors |
| `python-multipart` | ≥0.0.9 | File upload handling in FastAPI |
| `aiofiles` | ≥23.0 | Async file serving for thumbnails |

---

### 📂 `backend/app/` — Python Application Package

### 📄 `backend/app/__init__.py` (0 bytes)

**Why it exists:** Makes `app/` a Python package. Required for `from app.core import ...` imports to work. Empty file — just a marker.

---

### 📄 `backend/app/main.py` (4.8 KB) — FastAPI Application Entry Point

**Why it exists:** THE entry point. This is what `uvicorn app.main:app` loads.

**What it does:**
1. **Startup (lifespan):** init database → warm up CLIP model → load FAISS index → load fusion model
2. **CORS middleware:** Allows frontend (port 5173) to call backend (port 8000)
3. **Route registration:** Mounts all API routers (search, index, files, system, training)
4. **Health endpoint:** `GET /api/health` returns system status
5. **SPA fallback:** If frontend build exists in `static/`, serves it; any unknown path returns `index.html`

---

### 📄 `backend/app/config.py` (4.8 KB) — Central Configuration

**Why it exists:** ALL tunable parameters in ONE place. Changing behavior = editing one file.

**Key settings:**
```
CLIP_MODEL_NAME = "ViT-B-32"          # Which CLIP model
EMBEDDING_DIM   = 512                 # Vector dimensions
BATCH_SIZE      = 16                  # Images per GPU batch
DEFAULT_TOP_K   = 50                  # Default search results
MIN_SIMILARITY_THRESHOLD = 0.20      # Minimum score to show
OCR_ENABLED     = True                # OCR on/off
HYBRID_WEIGHT_VISUAL = 0.55          # Fallback weight for visual signal
```

---

### 📂 `backend/app/core/` — AI/ML Core Logic

### 📄 `backend/app/core/__init__.py` (0 bytes)

**Why it exists:** Makes `core/` a Python package.

---

### 📄 `backend/app/core/engine.py` (5.8 KB) — CLIP AI Model

**Why it exists:** This IS the AI brain. Loads the CLIP model and provides `embed_text()` and `embed_images()` functions.

**What it does:**
- **`_detect_device()`** — Auto-detects: CUDA GPU → Apple MPS → CPU. Actually runs a GPU kernel test.
- **`_lazy_load()`** — Loads CLIP ViT-B-32 model on first use (singleton pattern, thread-safe)
- **`embed_text(query)`** — Converts "sunset on beach" → 512-dimensional float vector
- **`embed_images(images)`** — Converts PIL images → 512-dimensional vectors (batched for GPU efficiency)
- **`warmup()`** — Called at startup to pre-load the model into GPU memory (~350MB VRAM)
- **`_cpu_fallback()`** — If GPU crashes mid-inference, automatically moves model to CPU

---

### 📄 `backend/app/core/fusion_model.py` (4.0 KB) — Query-Conditioned Fusion MLP

**Why it exists:** The novel research contribution. A trainable neural network that predicts adaptive weights for 4 search signals based on the query.

**Architecture:** 512 → 256 → 128 → 64 → 4 (MLP with LayerNorm, GELU, Dropout)
**Parameters:** 173,508 trainable parameters
**Output:** 4 weights summing to 1.0 (via softmax) — one for each signal (visual, OCR, filename, structural)

---

### 📄 `backend/app/core/hybrid.py` (6.0 KB) — Hybrid Scoring Engine

**Why it exists:** Combines 4 scoring signals (CLIP visual, OCR text, filename, structural heuristics) into a single final score. Uses the fusion model weights if trained, otherwise falls back to fixed weights.

**The 4 signals:**
1. **Visual** — CLIP cosine similarity between text query and image embedding
2. **Text (OCR)** — Word overlap between query and text extracted from the image
3. **Filename** — Word overlap between query and image filename tokens
4. **Structural** — Heuristic boosts/penalties (document detection, screenshot detection)

---

### 📄 `backend/app/core/similarity.py` (4.3 KB) — Search Dispatcher

**Why it exists:** Decides whether to use FAISS (fast, approximate) or cosine search (exact, slower), runs the search, and returns results with metadata.

**Auto-selection:** Uses FAISS when index is built AND image count > 1000; otherwise cosine.

---

### 📄 `backend/app/core/faiss_search.py` (4.3 KB) — FAISS Vector Index

**Why it exists:** Facebook's FAISS library provides ~10x faster search for 233K+ vectors. Without it, cosine search takes ~50ms; with FAISS IVF it takes ~5ms.

**Index types:**
- < 10K images → `IndexFlatIP` (exact inner product)
- ≥ 10K images → `IndexIVFFlat` (approximate, clusters vectors, searches nearby clusters only)

---

### 📄 `backend/app/core/indexer.py` (17.5 KB) — Photo Indexing Pipeline

**Why it exists:** The batch processing engine. Scans directories, generates CLIP embeddings, extracts OCR text, creates thumbnails, and stores everything.

**Key features:**
- Incremental indexing (skips already-indexed files by checking modified timestamp)
- SHA256 deduplication (skips identical content)
- Batch processing (16 images per GPU batch for efficiency)
- Crash-resumable (checkpoints every 100 files)
- Real-time progress tracking via JSON file

---

### 📄 `backend/app/core/index_cli.py` (665 bytes) — Indexer CLI Wrapper

**Why it exists:** Allows indexing to run as a separate Python process. The backend spawns `python -m app.core.index_cli <directory>` so indexing doesn't block the API server.

---

### 📄 `backend/app/core/faiss_cli.py` (647 bytes) — FAISS CLI Wrapper

**Why it exists:** Same pattern as `index_cli.py`. Builds the FAISS index as a separate process so it doesn't block the server.

---

### 📄 `backend/app/core/embedding_store.py` (3.9 KB) — Embedding Storage

**Why it exists:** Stores 233K × 512 CLIP vectors efficiently using memory-mapped numpy files. Memory mapping avoids loading ~476MB into RAM.

**Design decisions:**
- `memmap` mode lets the OS page in only what's needed
- 1.5x growth factor when file needs to grow (like ArrayList in Java)
- Atomic file swap on resize (write to `.tmp.npy`, then rename — prevents corruption)
- Count is written AFTER successful flush (crash-safe)

---

### 📄 `backend/app/core/ocr.py` (5.3 KB) — Text Extraction

**Why it exists:** Enables searching for text INSIDE images (invoices, signs, screenshots, WhatsApp messages) using EasyOCR.

**Functions:**
- `extract_text()` — Runs EasyOCR on image, returns all text with confidence > 30%
- `extract_filename_tokens()` — "IMG_2023_birthday_cake.jpg" → "img 2023 birthday cake"
- `detect_document()` — Heuristic: >50 words + A4-like aspect ratio = document
- `detect_screenshot()` — Heuristic: extreme aspect ratios or desktop resolutions

---

### 📄 `backend/app/core/thumbnail.py` (1.7 KB) — Thumbnail Generator

**Why it exists:** Creates small 256×256 JPEG previews for fast UI loading. Without thumbnails, the UI would load full-resolution photos and be very slow.

**Key detail:** Uses SHA256 hash of file path as thumbnail filename → stable naming, no collisions.

---

### 📂 `backend/app/api/` — REST API Routes

### 📄 `backend/app/api/__init__.py` (0 bytes)

**Why it exists:** Makes `api/` a Python package.

---

### 📄 `backend/app/api/routes_search.py` (3.6 KB) — Search API

**Why it exists:** Main search endpoint: `GET /api/search?q=sunset&type=image&top_k=50`.

**Flow:** Text → CLIP embedding → FAISS/cosine search → hybrid re-scoring → paginate → return JSON.

Also handles search history (saves queries, 10-day auto-cleanup).

---

### 📄 `backend/app/api/routes_index.py` (2.4 KB) — Indexing API

**Why it exists:** API to start/stop/monitor photo indexing.

**Endpoints:**
- `POST /api/index` — Start indexing a directory (non-blocking, launches background process)
- `GET /api/index/status` — Live progress (polled by frontend every 2 seconds)
- `DELETE /api/index` — Wipe entire index
- `DELETE /api/index/folder` — Remove specific folder from index

---

### 📄 `backend/app/api/routes_files.py` (4.9 KB) — File Operations API

**Why it exists:** Serves thumbnails, gallery browsing, folder tree, file opening, and statistics.

**Endpoints:**
- `GET /api/thumbnails/{id}` — Serve thumbnail image
- `POST /api/open-file` — Open file in OS file explorer (validates path from DB, not user input — security)
- `GET /api/folders` — List all indexed folders with counts
- `GET /api/gallery` — Paginated media browser
- `GET /api/stats` — Database statistics

---

### 📄 `backend/app/api/routes_training.py` (14.4 KB) — Training API

**Why it exists:** Manages ML training runs from the web UI. Training runs as a separate Python process so it doesn't crash the API server.

**Endpoints:**
- `POST /api/training/start` — Launch training process with mode/epochs/lr/batch_size
- `GET /api/training/status` — Training progress (epoch, loss, accuracy)
- `GET /api/training/stream` — Server-Sent Events for real-time log streaming
- `GET /api/training/fusion-status` — Is fusion model loaded? What device?
- `POST /api/training/build-faiss` — Rebuild FAISS index after new training
- `GET /api/training/artifacts` — List all past training runs with metrics
- `DELETE /api/training/model` — Delete fusion model, revert to fixed weights

---

### 📄 `backend/app/api/routes_system.py` (1.3 KB) — System API

**Why it exists:** Opens native OS directory picker dialog using tkinter. Desktop-app feature.

---

### 📂 `backend/app/db/` — Database Layer

### 📄 `backend/app/db/__init__.py` (0 bytes)

**Why it exists:** Makes `db/` a Python package.

---

### 📄 `backend/app/db/database.py` (10.7 KB) — SQLite Database

**Why it exists:** Stores all image metadata, search history, and indexing state. SQLite is perfect for a local desktop app — zero config, single file, fast.

**Schema:**
- `media` table — file_path, filename, type, folder, dimensions, hash, thumbnail_path, embedding_index, ocr_text, filename_tokens, is_document, is_screenshot
- `indexing_state` table — key/value store for checkpoints
- `search_history` table — recent searches with 10-day retention

**Design decisions:**
- WAL mode for concurrent reads during searches while indexing writes
- `UNIQUE(file_path)` with `ON CONFLICT DO UPDATE` for upsert pattern
- Thread-local connections (one per thread, `check_same_thread=False`)

---

### 📂 `backend/static/` — Production Frontend Build

### 📄 `backend/static/index.html` (749 bytes)

**Why it exists:** The compiled React frontend. When you run `npm run build` in the frontend folder, Vite outputs the compiled app here. The backend serves this at `http://localhost:8000`.

### 📄 `backend/static/assets/index-CXV3h5z2.css` (27 KB)

**Why it exists:** Compiled CSS (TailwindCSS → minified). All styles for the UI.

### 📄 `backend/static/assets/index-DCsBZ1xy.js` (297 KB)

**Why it exists:** Compiled JavaScript (React + TypeScript → minified bundle). All frontend logic.

---

## 3. Backend Data — Generated Runtime Files

### 📂 `backend/data/` — Runtime Data (Generated, Not Source Code)

Everything in this folder is **generated at runtime** — not hand-written code. It's created when you index photos and train models.

---

### 📊 `backend/data/metadata.db` (3.3 GB) — SQLite Database

**Why it exists:** Stores metadata for all 233,199 indexed images. Large because it includes OCR text for every image.

**Why 3.3 GB?** Each image stores: file_path, filename, folder, dimensions, SHA256 hash, OCR text (can be 100s of characters), filename tokens, document/screenshot flags, pixel area, timestamps. 233K images × ~14KB average = ~3.2 GB.

**Associated files:**
- `metadata.db-wal` (62 KB) — Write-Ahead Log. SQLite uses this for crash recovery. Temporary.
- `metadata.db-shm` (32 KB) — Shared Memory index. SQLite uses this for WAL reader coordination. Temporary.

---

### 📊 `backend/data/embeddings.npy` (614 MB) — CLIP Embedding Vectors

**Why it exists:** Stores 300,032 × 512 float32 vectors (capacity for 300K images, 233K used). This IS the search index — when you search, the backend compares your query vector against these.

**Why 614 MB?** 300,032 vectors × 512 dimensions × 4 bytes/float = 614 MB. Memory-mapped, so only pages accessed are loaded into RAM.

---

### 📊 `backend/data/embeddings_meta.json` (70 bytes)

**Why it exists:** Tracks how many embeddings are stored and what CLIP model was used.

**Content:** `{"count": 233199, "capacity": 300032, "model": "ViT-B-32"}`

---

### 📊 `backend/data/faiss.index` (480 MB) — FAISS Search Index

**Why it exists:** Pre-built FAISS IVF index for fast approximate nearest-neighbor search. Without this, search is brute-force cosine (slower). With this, search is ~10x faster.

**Built by:** `POST /api/training/build-faiss` or `python -m app.core.faiss_cli`.

---

### 🧠 `backend/data/fusion_model.pt` (699 KB) — Trained Fusion Model

**Why it exists:** The trained neural network weights for query-conditioned fusion. This is the OUTPUT of training. The backend loads this at startup to get adaptive search weights.

**Content:** PyTorch state_dict — 173,508 parameters serialized.

### 🧠 `backend/data/fusion_model_best.pt` (699 KB) — Best Checkpoint

**Why it exists:** Copy of the best-performing model during training (highest validation accuracy). Safety backup.

---

### 📄 `backend/data/training_status.json` (460 bytes) — Training State

**Why it exists:** Persists training status across server restarts. The backend reads this to know if training is running, what epoch it's on, etc.

**Content:** `{"status": "running", "mode": "hard", "epoch": 0, "total_epochs": 20, "pid": 29512, ...}`

---

### 📂 `backend/data/thumbnails/` — Thumbnail Cache (~233K JPEG files)

**Why it exists:** Stores 256×256 JPEG thumbnails for every indexed image. The UI loads these instead of full-resolution photos for performance.

**Naming:** SHA256(file_path)[:16] + ".jpg" — e.g., `208e21703510ba72.jpg`

---

### 📂 `backend/data/training_logs/` — Training Run Artifacts

**Why it exists:** Records every training session — terminal logs, metrics, timestamps.

**File types:**

**`terminal_*.log`** — Raw terminal output from each training run. Contains epoch-by-epoch loss and accuracy.

Example: `terminal_semi_hard_20260603_235059.log`
```
[16:05:23] Epoch 1/20 loss=0.38521 acc=0.612
[16:05:45] Epoch 2/20 loss=0.28103 acc=0.734
...
[16:12:01] Epoch 20/20 loss=0.04218 acc=0.942
```

**`train_*.json`** — Training session metadata (mode, epochs, lr, batch_size, started_at, finished_at, results).

Example: `train_semi_hard_20260603_235059.json`
```json
{
  "session_id": "20260603_235059",
  "mode": "semi_hard",
  "epochs": 20,
  "status": "done",
  "result": {"final_loss": 0.04218, "final_acc": 0.942, "artifact_dir": "..."}
}
```

---

### 📂 `backend/data/training_example_cache/` — Cached Training Data

**Why it exists:** Caches pre-computed training examples (query + positive/negative signal triplets) so re-training doesn't recompute them from scratch. Speeds up training by ~3x on subsequent runs.

---

### 📂 `backend/data/agent_logs/` — Project Agent Logs

**Why it exists:** Stores logs from `tools/project_agent.py` (the automated health monitor).

---

## 4. Frontend — React UI Application

### 📂 `frontend/` — React + TypeScript + Vite + TailwindCSS

---

### 📄 `frontend/package.json` (873 bytes) — NPM Project Definition

**Why it exists:** Defines the frontend project — name, version, dependencies, and build scripts.

**Key dependencies:**
| Package | Why |
|---------|-----|
| `react` 18.3 | UI framework |
| `react-router-dom` 6.26 | Client-side page routing |
| `@tanstack/react-query` 5.56 | Data fetching + caching for API calls |
| `lucide-react` | Icon library |
| `tailwindcss` 3.4 | Utility-first CSS framework |
| `vite` 5.4 | Build tool + dev server |

**Scripts:**
- `npm run dev` → Starts dev server on port 5173 with hot reload
- `npm run build` → Compiles to `backend/static/` for production

---

### 📄 `frontend/package-lock.json` (85 KB) — Dependency Lock File

**Why it exists:** Locks exact versions of ALL npm packages (including nested dependencies) to ensure reproducible builds. Auto-generated by `npm install`.

---

### 📄 `frontend/index.html` (651 bytes) — HTML Entry Point

**Why it exists:** The single HTML page that loads the React application. Contains `<div id="root">` where React mounts.

---

### 📄 `frontend/vite.config.ts` (400 bytes) — Vite Build Configuration

**Why it exists:** Configures the build tool:
- **Dev server proxy:** `/api` requests → forwarded to `http://localhost:8000` (backend)
- **Build output:** Compiled files go to `../backend/static/` (so backend can serve them)
- **Path alias:** `@/` → `./src/` for clean imports

---

### 📄 `frontend/tsconfig.json` (510 bytes) — TypeScript Configuration

**Why it exists:** Configures TypeScript compiler — strict mode, JSX support, path aliases.

### 📄 `frontend/tsconfig.node.json` (171 bytes) — Node TypeScript Config

**Why it exists:** Separate TypeScript config for Node.js files (vite.config.ts). Needed because Vite config runs in Node, not browser.

---

### 📄 `frontend/postcss.config.js` (66 bytes) — PostCSS Configuration

**Why it exists:** Tells PostCSS to use TailwindCSS and Autoprefixer plugins. Required for TailwindCSS to work.

### 📄 `frontend/tailwind.config.ts` (2.0 KB) — TailwindCSS Configuration

**Why it exists:** Configures TailwindCSS — custom colors, fonts, animations, dark mode support.

---

### 📂 `frontend/src/` — Source Code

### 📄 `frontend/src/main.tsx` (558 bytes) — React Entry Point

**Why it exists:** Mounts the React app into the DOM. Sets up React Query provider for API data fetching.

### 📄 `frontend/src/App.tsx` (3.4 KB) — Root Application Component

**Why it exists:** Defines the app layout and routing:
- Navigation bar with Search / Indexing / Training tabs
- React Router routes for each page
- Dark theme with modern styling

### 📄 `frontend/src/index.css` (2.5 KB) — Global Styles

**Why it exists:** TailwindCSS directives + custom CSS variables (colors, fonts, animations).

---

### 📂 `frontend/src/pages/` — Page Components

### 📄 `frontend/src/pages/SearchPage.tsx` (21 KB) — Search Interface

**Why it exists:** The main search UI. Contains:
- Search bar with real-time suggestions
- Results grid showing thumbnails with scores
- Filters (folder, date range, file type, extensions)
- Pagination
- Click to open file in OS file explorer

### 📄 `frontend/src/pages/IndexPage.tsx` (1.4 KB) — Indexing Page

**Why it exists:** Hosts the IndexingPanel component. Simple wrapper.

### 📄 `frontend/src/pages/TrainingPage.tsx` (30.6 KB) — Training Dashboard

**Why it exists:** Full training management UI:
- Start training with mode/epochs/lr selection
- Real-time progress bar and log streaming (SSE)
- Historical training runs with metrics comparison
- Learning curve visualization
- FAISS index rebuild button

---

### 📂 `frontend/src/components/` — Reusable UI Components

### 📄 `frontend/src/components/SearchBar.tsx` (6.4 KB)

**Why it exists:** Search input with keyboard shortcuts, auto-focus, and search history dropdown.

### 📄 `frontend/src/components/PhotoGrid.tsx` (7.2 KB)

**Why it exists:** Responsive image grid. Displays thumbnail, filename, score badge, OCR text preview. Click to open in OS.

### 📄 `frontend/src/components/IndexingPanel.tsx` (16 KB)

**Why it exists:** Directory selection, indexing progress bar, live statistics, error display.

### 📄 `frontend/src/components/TrainingPanel.tsx` (12.4 KB)

**Why it exists:** Training controls — mode selection, parameter inputs, start/stop buttons, real-time status.

---

### 📂 `frontend/src/lib/` — Utility Functions

### 📄 `frontend/src/lib/api.ts` (8.8 KB) — API Client

**Why it exists:** Centralized API functions. Every API call goes through this file.

**Functions:** `searchPhotos()`, `startIndexing()`, `getIndexStatus()`, `startTraining()`, `getTrainingStatus()`, `getFolders()`, `getStats()`, etc.

### 📄 `frontend/src/lib/utils.ts` (413 bytes) — Helper Utilities

**Why it exists:** `cn()` function that merges TailwindCSS class names (using `clsx` + `tailwind-merge`).

---

### 📂 `frontend/node_modules/` — NPM Dependencies (DO NOT SHIP)

**Why it exists:** Contains ALL installed npm packages (~200MB). Auto-generated by `npm install`. Never committed to git or included in distribution zips.

---

## 5. Training — ML Training Module

### 📂 `training/` — Fusion Model Training

---

### 📄 `training/train.py` (7.8 KB) — Training CLI Entry Point

**Why it exists:** Command-line script to run training. Called by the backend as a subprocess: `python train.py --mode semi_hard --epochs 20`.

**What it does:**
1. Parses arguments (mode, epochs, lr, batch_size, data_dir)
2. Sets up colored terminal logging
3. Loads the appropriate trainer class
4. Runs training with progress callbacks
5. Saves training artifact JSON for the frontend dashboard

---

### 📄 `training/requirements.txt` (153 bytes) — Training Dependencies

**Why it exists:** Minimal dependencies for training (torch, torchvision, open-clip-torch, numpy). Subset of backend requirements.

---

### 📄 `training/embed_captions.py` (5.2 KB) — Caption Embedding Generator

**Why it exists:** Pre-computes CLIP text embeddings for image captions. Used to create query-image pairs for training data.

---

### 📂 `training/modes/` — Training Mode Implementations

### 📄 `training/modes/__init__.py` (488 bytes) — Mode Registry

**Why it exists:** Factory function `get_trainer(mode)` that returns the correct trainer class based on mode name.

```python
def get_trainer(mode):
    if mode == "basic": return BasicTrainer()
    elif mode == "baseline": return BaselineTrainer()
    elif mode == "semi_hard": return SemiHardTrainer()
    elif mode == "hard": return HardTrainer()
```

---

### 📄 `training/modes/base.py` (40.8 KB, 965 lines) — Shared Training Infrastructure

**Why it exists:** The largest file. Contains ALL shared training logic:

- **`extract_query_items()`** — Generates text queries from image metadata (captions, OCR, filenames)
- **`compute_4signals()`** — Computes the 4 scoring signals for each query-image pair
- **`RECALIXDataset`** — Builds triplet training examples (query, positive image, negative image)
- **`evaluate()`** — Pairwise ranking accuracy metric (does model rank positive > negative?)
- **`BaseTrainer`** — Setup, training loop, model saving, learning curve generation

---

### 📄 `training/modes/basic.py` (4.1 KB) — Basic Training Mode

**Why it exists:** Simplest training mode. Uses random negatives only. Good for verifying the pipeline works.

---

### 📄 `training/modes/baseline.py` (6.9 KB) — Baseline Training Mode

**Why it exists:** Research comparison mode. Trains with fixed weights AND learned weights, outputs a comparison table ("Table 1") for the paper.

---

### 📄 `training/modes/semi_hard.py` (7.5 KB) — Semi-Hard Training Mode ⭐ RECOMMENDED

**Why it exists:** The production training mode. Uses 50% random + 50% semi-hard negatives. Best balance of speed and quality.

**Why semi-hard?** Pure random negatives are too easy (model learns obvious cases). Pure hard negatives cause training instability. 50/50 mix gives the best results.

---

### 📄 `training/modes/hard.py` (12.2 KB) — Hard Negative Training Mode

**Why it exists:** Advanced mode. Finds the HARDEST negatives (most confusing wrong images). Used after semi_hard converges for the final model.

---

### 📂 `training/artifacts/` — Training Run Outputs

Contains subdirectories for each training run with learning curve plots, summary JSONs, and checkpoints.

---

## 6. Tools — Automation Scripts

### 📄 `tools/project_agent.py` (10.1 KB) — Automated Health Monitor

**Why it exists:** Background agent that monitors the project. Watches for:
- Code file changes
- Running indexing/training jobs
- Build health (backend + frontend)
- Disk space and process status

Logs to `backend/data/agent_logs/`. Launched by `run_project_agent.bat`.

---

## 7. Documentation & Config Files

### 📄 `README.md` (6.3 KB) — Project Overview

**Why it exists:** The main project README. Quick start guide, GPU support table, training instructions, architecture overview, API reference, troubleshooting.

### 📄 `SETUP.md` (3.9 KB) — Detailed Setup Guide

**Why it exists:** Step-by-step manual setup instructions for all platforms (Windows, Linux, Mac). Training mode guide with comparison table.

### 📂 `documents/` — Detailed Documentation

All documentation created for this project:

| File | Purpose |
|------|---------|
| `01_training_explained.md` | How training works — architecture, parameters, re-training logic |
| `02_setup_changelog.md` | All bugs fixed during setup |
| `03_sanity_test_results.md` | All 10 API test results |
| `04_project_architecture.md` | System architecture, search flow, tech stack |
| `05_quick_reference.md` | Operations cheat sheet |
| `06_complete_code_walkthrough.md` | THIS FILE — every file explained |

### 📂 `.vscode/`

### 📄 `.vscode/settings.json` (151 bytes)

**Why it exists:** VS Code editor settings — Python interpreter path, editor preferences. Not part of the application.

---

## 8. Virtual Environment (venv_gpu)

### 📂 `venv_gpu/` — Python Virtual Environment (~3-5 GB)

**Why it exists:** Isolated Python environment with all dependencies installed. Ensures the project uses specific package versions without conflicting with system Python.

**Key contents:**
```
venv_gpu/
├── Scripts/
│   ├── python.exe        ← Python interpreter used by the project
│   ├── pip.exe           ← Package manager
│   ├── activate.bat      ← Activates the venv in terminal
│   └── uvicorn.exe       ← Web server executable
├── Lib/
│   └── site-packages/    ← ALL installed packages
│       ├── torch/        ← PyTorch (1.5 GB with CUDA support)
│       ├── open_clip/    ← CLIP model library
│       ├── easyocr/      ← OCR engine
│       ├── faiss/        ← FAISS search library
│       ├── fastapi/      ← Web framework
│       └── ... (~100 more packages)
└── pyvenv.cfg            ← Venv config (Python version, base path)
```

**Why NOT shipped:** 3-5 GB, platform-specific binaries (Windows only), PyTorch CUDA version tied to specific GPU. The `.exe` and `setup_gpu.bat` recreate this automatically for each user's hardware.

---

## 9. Complete Data Flow

```
User double-clicks recalix.exe
    ↓
recalix.py → Detects GPU (NVIDIA RTX 5060, sm_120, Blackwell)
    ↓
Creates venv_gpu → Installs PyTorch cu128 + all deps
    ↓
Starts uvicorn → backend/app/main.py
    ↓ (startup)
init_db() → Creates metadata.db tables
engine.warmup() → Loads CLIP ViT-B-32 into GPU (350MB VRAM)
faiss_search.load_index() → Loads faiss.index (480MB)
hybrid._load_fusion_model() → Loads fusion_model.pt (699KB)
    ↓
Server ready on http://localhost:8000
    ↓
User searches "sunset on beach"
    ↓
routes_search.py → engine.embed_text("sunset on beach") → 512-dim vector
    ↓
similarity.py → faiss_search.search() → top 300 candidates in ~5ms
    ↓
database.py → Filter by folder/date/type → ~100 valid results
    ↓
hybrid.py → For each result: compute visual + OCR + filename + structural
    ↓
fusion_model.pt → predict_weights(query_embedding) → [0.82, 0.05, 0.08, 0.05]
    ↓
final_score = 0.82×visual + 0.05×ocr + 0.08×filename + 0.05×structural
    ↓
Sort by score → paginate (24/page) → return JSON
    ↓
Frontend renders thumbnail grid with scores
```

---

## 10. Examiner Q&A Reference

| Question | Answer |
|---|---|
| **Why CLIP ViT-B-32?** | Balanced speed vs accuracy. B-32 is 5x faster than L-14 with ~90% quality. Good enough for 233K images. |
| **Why not fine-tune CLIP?** | 151M parameters needs massive data. Our 173K fusion head achieves the same goal much more efficiently. |
| **Why 4 signals?** | Pure CLIP misses text IN images and filename info. "Invoice" search needs OCR. "birthday_photo.jpg" needs filename matching. |
| **Why SQLite?** | Local desktop app — zero config, single file, fast. No need for Postgres/MySQL. |
| **Why is metadata.db 3.3 GB?** | 233K images × (file paths + OCR text + tokens + all metadata) = large. OCR text for each image can be hundreds of characters. |
| **Why FAISS?** | 233K × 512 = 476MB of vectors. Brute-force cosine ~50ms. FAISS IVF ~5ms (10x faster). |
| **Why memory-mapped embeddings?** | 614MB file. Memory mapping loads only pages accessed. App doesn't use 614MB RAM. |
| **Why triplet loss?** | Learning to rank (positive > negative), not classify. Triplet loss directly optimizes ranking quality. |
| **Why separate training process?** | GPU-intensive training inside FastAPI would block search and risk OOM crashes. |
| **Why EasyOCR over Tesseract?** | EasyOCR: GPU-accelerated, better accuracy on photos. Tesseract: faster but worse on non-document images. |
| **Why .exe launcher?** | Users double-click one file. No Python/pip knowledge needed. PyInstaller compiles Python to standalone exe. |
| **Why venv_gpu not shipped?** | 3-5 GB, platform-specific, GPU-tied. The exe recreates it correctly for each user's hardware. |
| **How does search know "sunset" = beach photo?** | CLIP was trained on 400M image-text pairs from the internet. It learned that "sunset" vectors are close to sunset image vectors. |
| **Does re-training help?** | First run: big improvement (0.50 → 0.94 accuracy). Subsequent runs on same data: minimal gain (already converged). New images or new modes can help. |
| **What does the fusion model learn?** | For visual queries ("sunset") → weight visual signal high. For text queries ("invoice") → weight OCR high. It learns query-type detection. |
