"""
RECALIX V10 — FastAPI Application
===================================
Serves REST API + React SPA (production build).
Startup: init DB → warmup CLIP → try load FAISS index → try load fusion model
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.config import CORS_ORIGINS, THUMBNAILS_DIR, API_HOST, API_PORT, BASE_DIR
from app.db.database import init_db
from app.api.routes_index    import router as index_router
from app.api.routes_search   import router as search_router
from app.api.routes_files    import router as files_router
from app.api.routes_system   import router as system_router
from app.api.routes_training import router as training_router

STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("  RECALIX V10 - Local AI Photo Search")
    print("=" * 60)

    init_db()
    print("[Boot] Database ready.")

    loop = asyncio.get_event_loop()

    # CLIP model warmup
    await loop.run_in_executor(None, _warmup_clip)

    # FAISS index (non-blocking, best-effort)
    await loop.run_in_executor(None, _try_load_faiss)

    # Fusion model (non-blocking, best-effort)
    await loop.run_in_executor(None, _try_load_fusion)

    print(f"[Boot] Ready -> http://localhost:{API_PORT}")
    print("=" * 60)
    yield
    print("[Shutdown] RECALIX stopping.")


def _warmup_clip():
    try:
        from app.core.engine import warmup
        warmup()
    except Exception as e:
        print(f"[Boot] CLIP warmup failed (will retry on first request): {e}")


def _try_load_faiss():
    try:
        from app.core import faiss_search
        if faiss_search.load_index():
            print("[Boot] FAISS index loaded.")
        else:
            print("[Boot] No FAISS index - using cosine search.")
    except Exception as e:
        print(f"[Boot] FAISS load skipped: {e}")


def _try_load_fusion():
    try:
        from app.core.hybrid import _load_fusion_model, get_fusion_status
        _load_fusion_model()
        s = get_fusion_status()
        if s["fusion_active"]:
            print(f"[Boot] Fusion model active (device={s['device']}).")
            print(f"[Boot] Search: CLIP text embed -> cosine/FAISS -> fusion re-score (adaptive weights)")
        else:
            fw = s["fallback_weights"]
            print(f"[Boot] No fusion model - fixed weights: visual={fw['visual']}, text={fw['text']}, filename={fw['filename']}")
            print(f"[Boot] Search: CLIP text embed -> cosine/FAISS -> fixed hybrid re-score")
            print(f"[Boot] Tip: Run training to improve search quality with learned weights.")
    except Exception as e:
        print(f"[Boot] Fusion load skipped: {e}")


app = FastAPI(title="RECALIX V10", description="Local AI Photo Search", version="10.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(index_router)
app.include_router(search_router)
app.include_router(files_router)
app.include_router(system_router)
app.include_router(training_router)


@app.get("/api/health")
async def health():
    from app.core.similarity import get_active_backend
    from app.core.hybrid import get_fusion_status
    from app.core import embedding_store
    return {
        "status":         "ok",
        "version":        "10.0.0",
        "indexed":        embedding_store.count(),
        "search_backend": get_active_backend(),
        "fusion":         get_fusion_status()["fusion_active"],
    }


@app.get("/")
async def root():
    if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
        return FileResponse(str(STATIC_DIR / "index.html"))
    return {"message": "RECALIX V10 backend running", "docs": "/docs"}


# Serve static assets + SPA fallback
if STATIC_DIR.exists():
    if THUMBNAILS_DIR.exists():
        app.mount("/thumbs", StaticFiles(directory=str(THUMBNAILS_DIR)), name="thumbs")

    assets = STATIC_DIR / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            return JSONResponse({"detail": "Not found"}, 404)
        idx = STATIC_DIR / "index.html"
        return FileResponse(str(idx)) if idx.exists() else JSONResponse({"error": "Frontend not built"}, 404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=False, workers=1)
