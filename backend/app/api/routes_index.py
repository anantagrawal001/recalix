"""
RECALIX API — Indexing Routes
POST /api/index        → Start indexing a directory
GET  /api/index/status → Get live progress
DELETE /api/index      → Clear the entire index
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from app.core.indexer import start_indexing, get_progress
from app.core import embedding_store
from app.db import database

router = APIRouter(prefix="/api/index", tags=["indexing"])


class IndexRequest(BaseModel):
    directory: str
    reindex: bool = False


@router.post("")
async def start_index(req: IndexRequest):
    """Start indexing a directory. Non-blocking — runs in background thread."""
    path = Path(req.directory)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.directory}")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {req.directory}")

    started = start_indexing(str(path.resolve()), reindex=req.reindex)
    if not started:
        return {"message": "Indexing already in progress", "started": False}

    return {"message": "Indexing started", "started": True, "directory": str(path.resolve())}


@router.get("/status")
async def index_status():
    """Return live indexing progress."""
    progress = get_progress()
    stats = database.get_stats()
    return {
        **progress,
        "db_stats": stats,
        "embedding_count": embedding_store.count(),
    }


@router.delete("")
async def clear_index():
    """Wipe all indexed data and embeddings."""
    database.clear_all()
    embedding_store.clear()
    return {"message": "Index cleared successfully"}


@router.delete("/folder")
async def clear_folder_index(
    folder: str,
    include_subfolders: bool = True,
):
    """Delete indexed media for a specific folder (optionally including subfolders)."""
    if not folder:
        raise HTTPException(status_code=400, detail="folder is required")

    result = database.delete_media_by_folder(folder, include_subfolders=include_subfolders)
    for thumb in result.get("thumbnails", []):
        try:
            Path(thumb).unlink(missing_ok=True)
        except Exception:
            pass

    return {
        "message": "Folder index cleared",
        "deleted": result.get("deleted", 0),
        "folder": folder,
        "include_subfolders": include_subfolders,
    }
