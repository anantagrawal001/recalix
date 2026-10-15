

import os
import sys
import math
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.db import database

router = APIRouter(prefix="/api", tags=["files"])


# ── Thumbnails ────────────────────────────────────────────────────────────────

@router.get("/thumbnails/{emb_index}")
async def get_thumbnail(emb_index: int):
    conn = database.get_conn()
    row = conn.execute(
        "SELECT thumbnail_path FROM media WHERE embedding_index = ?", (emb_index,)
    ).fetchone()
    if not row or not row["thumbnail_path"]:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    thumb_path = Path(row["thumbnail_path"])
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail file missing on disk")
    return FileResponse(
        str(thumb_path),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


# ── Open File (SECURITY FIX) ─────────────────────────────────────────────────

class OpenFileRequest(BaseModel):
    media_id: int                   # ← DB row ID, NOT raw file path from frontend
    show_in_folder: bool = False


@router.post("/open-file")
async def open_file(req: OpenFileRequest):
    """
    Open a file by its DB media_id.
    Security: path is ALWAYS resolved from the database, never trusted from client.
    """
    conn = database.get_conn()
    row = conn.execute("SELECT file_path FROM media WHERE id = ?", (req.media_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Media not found in database")

    path = Path(row["file_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File no longer exists on disk: {path}")

    try:
        if sys.platform == "win32":
            if req.show_in_folder:
                subprocess.Popen(["explorer", "/select,", str(path)])
            else:
                os.startfile(str(path))
        elif sys.platform == "darwin":
            target = str(path.parent) if req.show_in_folder else str(path)
            subprocess.Popen(["open", target])
        else:  # Linux
            target = str(path.parent) if req.show_in_folder else str(path)
            subprocess.Popen(["xdg-open", target])
        return {"success": True, "opened": str(path), "show_in_folder": req.show_in_folder}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OS could not open file: {str(e)}")


# ── Folder Tree ───────────────────────────────────────────────────────────────

@router.get("/folders")
async def get_folders():
    folders = database.get_all_folders()
    return {"folders": folders}


@router.get("/gallery")
async def get_gallery(
    folder: Optional[str] = Query(None),
    file_type: Optional[str] = Query("image", alias="type"),
    ext: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(48, ge=1, le=200),
):
    file_type = file_type if file_type in ("image", "video") else None
    ext_list = None
    if ext:
        ext_list = [e.strip().lower() for e in ext.split(",") if e.strip()]
    data = database.get_media_gallery(
        folder=folder,
        file_type=file_type,
        page=page,
        page_size=page_size,
        extensions=ext_list,
    )
    results = []
    for row in data["items"]:
        emb_idx = row.get("embedding_index")
        results.append({
            **row,
            "score": 1.0,
            "thumbnail_url": f"/thumbs/{Path(row['thumbnail_path']).name}" if row.get("thumbnail_path") else None,
            "_search_backend": "gallery",
        })
    total = data["total"]
    return {
        "folder": folder,
        "results": results,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": math.ceil(total / page_size) if total else 0,
    }


# ── Stats ─────────────────────────────────────────────────────────────────────

@router.get("/stats")
async def get_stats():
    from app.core import embedding_store
    stats = database.get_stats()
    return {
        **stats,
        "embedding_count": embedding_store.count(),
        "model_version": embedding_store.get_model_version(),
    }
