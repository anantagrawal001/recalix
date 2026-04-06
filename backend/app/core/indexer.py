"""
RECALIX Indexer — Production Grade
====================================
Fixes over v1:
- Incremental indexing: skips already-indexed files (by path + mtime)
- Duplicate detection: skips same content via SHA256 hash
- GPU memory management: cuda.empty_cache() after each batch
- Embedding version check: warns if model changed since last index
- Immediate batch flush (no accumulation in RAM)
- Crash-resumable: checkpoint every N files
"""

import os
import sys
import time
import hashlib
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np

from app.config import (
    IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, ALL_EXTENSIONS,
    BATCH_SIZE, CHECKPOINT_EVERY, CLIP_MODEL_NAME,
)
from app.core import engine, embedding_store
from app.core.thumbnail import generate_thumbnail, open_image_for_clip
from app.db import database

# ── Shared progress state ─────────────────────────────────────────────────────
_progress: Dict[str, Any] = {
    "status": "idle",
    "total": 0,
    "processed": 0,
    "skipped": 0,
    "errors": 0,
    "new_indexed": 0,
    "current_file": "",
    "started_at": None,
    "finished_at": None,
    "message": "",
}
_progress_lock = threading.Lock()
_indexing_thread: Optional[threading.Thread] = None


def get_progress() -> Dict[str, Any]:
    with _progress_lock:
        return dict(_progress)


def _set_progress(**kwargs):
    with _progress_lock:
        _progress.update(kwargs)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _file_hash(path: Path, chunk_size: int = 65536) -> str:
    """SHA256 of first 64KB — fast dedup without reading entire file."""
    h = hashlib.sha256()
    try:
        with open(str(path), "rb") as f:
            h.update(f.read(chunk_size))
    except Exception:
        return ""
    return h.hexdigest()


def scan_directory(root: str) -> List[Path]:
    root_path = Path(root)
    files = set()
    for ext in ALL_EXTENSIONS:
        files.update(root_path.rglob(f"*{ext}"))
        files.update(root_path.rglob(f"*{ext.upper()}"))
    return [f.resolve() for f in files if f.is_file()]


def _get_file_meta(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    ext = path.suffix.lower()
    file_type = "image" if ext in IMAGE_EXTENSIONS else "video"
    return {
        "file_path": str(path),
        "filename": path.name,
        "file_type": file_type,
        "folder": str(path.parent),
        "file_size": stat.st_size,
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


# ── Main indexing routine ─────────────────────────────────────────────────────

def run_indexing(directory: str, reindex: bool = False):
    _set_progress(
        status="scanning",
        total=0, processed=0, skipped=0, errors=0, new_indexed=0,
        current_file="",
        started_at=datetime.now().isoformat(),
        finished_at=None,
        message="Scanning directory..."
    )

    try:
        # ── Embedding version check ───────────────────────────────────────────
        stored_model = embedding_store.get_model_version()
        if stored_model and stored_model != CLIP_MODEL_NAME and not reindex:
            _set_progress(
                status="error",
                message=(
                    f"Model mismatch: index was built with '{stored_model}' "
                    f"but current model is '{CLIP_MODEL_NAME}'. "
                    f"Run with reindex=True to rebuild."
                )
            )
            return

        # ── Wipe if reindex ───────────────────────────────────────────────────
        if reindex:
            database.clear_all()
            embedding_store.clear()

        # ── Scan ──────────────────────────────────────────────────────────────
        all_files = scan_directory(directory)
        total = len(all_files)
        _set_progress(total=total, message=f"Found {total} files. Checking which are new...")

        if total == 0:
            _set_progress(status="done", message="No media files found.")
            return

        # ── Separate images / videos ──────────────────────────────────────────
        image_files = [f for f in all_files if f.suffix.lower() in IMAGE_EXTENSIONS]
        video_files = [f for f in all_files if f.suffix.lower() in VIDEO_EXTENSIONS]

        _set_progress(status="indexing")

        processed = 0
        skipped = 0
        errors = 0
        new_indexed = 0

        # ── Track seen hashes this session (dedup within a single run) ────────
        seen_hashes: set = set()

        # ── Process images in batches ─────────────────────────────────────────
        batch_paths: List[Path] = []
        batch_metas: List[Dict] = []

        def flush_batch():
            nonlocal processed, errors, new_indexed
            if not batch_paths:
                return

            # Open images
            pil_images, valid_paths, valid_metas = [], [], []
            for p, m in zip(batch_paths, batch_metas):
                img = open_image_for_clip(str(p))
                if img:
                    pil_images.append(img)
                    valid_paths.append(p)
                    valid_metas.append(m)
                else:
                    errors += 1

            if pil_images:
                try:
                    import torch
                    embeddings = engine.embed_images(pil_images)       # (M, D), normalized

                    # Immediately free GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Append to store → get indices
                    indices = embedding_store.append_embeddings(embeddings, CLIP_MODEL_NAME)

                    # Thumbnails + DB — one at a time, immediate flush
                    for path, meta, emb_idx in zip(valid_paths, valid_metas, indices):
                        thumb = generate_thumbnail(str(path))
                        width, height = None, None
                        ocr_text = ""
                        filename_tokens = ""
                        is_document = 0
                        is_screenshot = 0
                        pixel_area = None
                        
                        try:
                            from PIL import Image as _Img
                            from app.core import ocr
                            
                            with _Img.open(str(path)) as im:
                                width, height = im.size
                                pixel_area = width * height if width and height else None
                                
                                # Phase 2: Extract OCR text
                                if meta["file_type"] == "image":
                                    ocr_text = ocr.extract_text(im)
                                    filename_tokens = ocr.extract_filename_tokens(meta["filename"])
                                    is_document = 1 if ocr.detect_document(
                                        ocr_text, width, height, meta.get("file_size", 0)
                                    ) else 0
                                    is_screenshot = 1 if ocr.detect_screenshot(width, height) else 0
                        
                        except Exception as e:
                            print(f"[Indexer] Metadata extraction failed for {path.name}: {e}")
                            pass

                        database.upsert_media({
                            **meta,
                            "width": width,
                            "height": height,
                            "thumbnail_path": str(thumb) if thumb else None,
                            "embedding_index": emb_idx,
                            "ocr_text": ocr_text,
                            "filename_tokens": filename_tokens,
                            "is_document": is_document,
                            "is_screenshot": is_screenshot,
                            "pixel_area": pixel_area,
                        })
                        new_indexed += 1
                        processed += 1
                        _set_progress(
                            processed=processed, skipped=skipped,
                            errors=errors, new_indexed=new_indexed,
                            current_file=path.name
                        )

                except Exception as e:
                    print(f"[Indexer] Batch error: {e}")
                    errors += len(pil_images)

            batch_paths.clear()
            batch_metas.clear()

        for i, path in enumerate(image_files):
            _set_progress(current_file=path.name)

            meta = _get_file_meta(path)
            modified_at = meta["modified_at"]

            # ── Incremental check: skip if already indexed with same mtime ────
            if not reindex:
                conn = database.get_conn()
                existing = conn.execute(
                    "SELECT modified_at FROM media WHERE file_path = ?",
                    (str(path),)
                ).fetchone()
                if existing and existing["modified_at"] == modified_at:
                    skipped += 1
                    processed += 1
                    _set_progress(processed=processed, skipped=skipped)
                    continue

            # ── Dedup check: skip if same content seen this session ───────────
            file_hash = _file_hash(path)
            if file_hash and file_hash in seen_hashes:
                skipped += 1
                processed += 1
                _set_progress(processed=processed, skipped=skipped)
                continue
            if file_hash:
                seen_hashes.add(file_hash)
            
            # Add file_hash to metadata
            meta["file_hash"] = file_hash

            batch_paths.append(path)
            batch_metas.append(meta)

            if len(batch_paths) >= BATCH_SIZE:
                flush_batch()

            # Checkpoint
            if (i + 1) % CHECKPOINT_EVERY == 0:
                database.set_state("last_checkpoint", i + 1)
                database.set_state("checkpoint_directory", directory)

        flush_batch()  # Remaining batch

        # ── Videos: metadata only (embeddings in future v2) ──────────────────
        for path in video_files:
            processed += 1
            skipped += 1
            try:
                if not reindex:
                    conn = database.get_conn()
                    exists = conn.execute(
                        "SELECT id FROM media WHERE file_path=?", (str(path),)
                    ).fetchone()
                    if exists:
                        _set_progress(processed=processed, skipped=skipped)
                        continue
                meta = _get_file_meta(path)
                database.upsert_media({
                    **meta, "width": None, "height": None,
                    "thumbnail_path": None, "embedding_index": None,
                })
            except Exception:
                errors += 1
            _set_progress(processed=processed, skipped=skipped, errors=errors)

        _set_progress(
            status="done",
            processed=processed,
            skipped=skipped,
            errors=errors,
            new_indexed=new_indexed,
            current_file="",
            finished_at=datetime.now().isoformat(),
            message=(
                f"Done! {new_indexed} new images indexed. "
                f"{skipped} skipped (already up-to-date or duplicate). "
                f"{errors} errors."
            )
        )

    except Exception as e:
        _set_progress(
            status="error",
            message=str(e),
            finished_at=datetime.now().isoformat(),
        )
        print(f"[Indexer] Fatal error: {e}")


def start_indexing(directory: str, reindex: bool = False) -> bool:
    global _indexing_thread
    if _progress.get("status") in ("scanning", "indexing"):
        return False
    _indexing_thread = threading.Thread(
        target=run_indexing, args=(directory, reindex), daemon=True
    )
    _indexing_thread.start()
    return True
