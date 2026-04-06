"""
RECALIX V9 — Central Configuration
====================================
All tunable parameters. Override anything with environment variables.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = BASE_DIR / "data"
THUMBNAILS_DIR  = DATA_DIR / "thumbnails"
DB_PATH         = DATA_DIR / "metadata.db"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
FAISS_INDEX_PATH= DATA_DIR / "faiss.index"
FUSION_MODEL_PATH = DATA_DIR / "fusion_model.pt"

DATA_DIR.mkdir(exist_ok=True)
THUMBNAILS_DIR.mkdir(exist_ok=True)

# ── CLIP Model ────────────────────────────────────────────────────────────────
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
EMBEDDING_DIM   = 512

# ── Device  (auto: CUDA → MPS → CPU) ─────────────────────────────────────────
# Set RECALIX_DEVICE=cpu/cuda/mps to override auto-detection
DEVICE_OVERRIDE = os.environ.get("RECALIX_DEVICE", "")

# ── Indexing ──────────────────────────────────────────────────────────────────
BATCH_SIZE        = 5
THUMBNAIL_SIZE    = (256, 256)
THUMBNAIL_QUALITY = 85
CHECKPOINT_EVERY  = 100

# ── Media extensions ──────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic", ".heif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}
ALL_EXTENSIONS   = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

# ── Search ────────────────────────────────────────────────────────────────────
DEFAULT_TOP_K              = 50
MAX_TOP_K                  = 500
MIN_SIMILARITY_THRESHOLD   = 0.23
SCREENSHOT_ASPECT_PENALTY  = 0.03

# Search backend: "cosine" | "faiss" | "auto"
# "auto" uses FAISS when built and count >= FAISS_AUTO_THRESHOLD
SEARCH_BACKEND      = os.environ.get("RECALIX_SEARCH_BACKEND", "auto")
FAISS_ENABLED       = os.environ.get("RECALIX_FAISS_ENABLED", "true").lower() != "false"
FAISS_AUTO_THRESHOLD = 1000   # switch to FAISS when > this many images indexed

# ── Hybrid scoring weights (fallback when no trained model) ───────────────────
HYBRID_SEARCH_ENABLED    = True
HYBRID_WEIGHT_VISUAL     = 0.55
HYBRID_WEIGHT_TEXT       = 0.30
HYBRID_WEIGHT_FILENAME   = 0.10
HYBRID_WEIGHT_STRUCTURAL = 0.05

# ── Fusion model (trained by training/ module) ────────────────────────────────
FUSION_ENABLED       = os.environ.get("RECALIX_FUSION_ENABLED", "true").lower() != "false"
FUSION_MODEL_OVERRIDE = os.environ.get("RECALIX_FUSION_MODEL", "")

# ── OCR ───────────────────────────────────────────────────────────────────────
OCR_ENABLED   = True
OCR_BACKEND   = "easyocr"   # "easyocr" | "tesseract"
OCR_LANGUAGES = ["en"]
OCR_GPU       = True

# ── Structural detection ──────────────────────────────────────────────────────
MIN_DOCUMENT_TEXT_LENGTH = 50
SCREENSHOT_ASPECT_MIN    = 0.33
SCREENSHOT_ASPECT_MAX    = 3.0
TINY_IMAGE_THRESHOLD     = 10000

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:4173",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8080",
    "http://localhost:8000",
]
