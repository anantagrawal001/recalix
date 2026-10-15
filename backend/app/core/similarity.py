"""
RECALIX V9 — Unified Search Dispatcher
=========================================
Dispatches to FAISS or cosine based on config and collection size.

Pipeline:
  1. Load embeddings (memmap)
  2. Get filtered embedding indices from DB
  3. Run FAISS or cosine search
  4. Apply screenshot aspect-ratio penalty
  5. Threshold filter
  6. Attach metadata + thumbnail URL
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.core import embedding_store
from app.db import database
from app.config import (
    DEFAULT_TOP_K, MAX_TOP_K,
    MIN_SIMILARITY_THRESHOLD, SCREENSHOT_ASPECT_PENALTY,
    SEARCH_BACKEND, FAISS_ENABLED, FAISS_AUTO_THRESHOLD,
)

logger = logging.getLogger("recalix.similarity")


def _use_faiss() -> bool:
    if not FAISS_ENABLED or SEARCH_BACKEND == "cosine":
        return False
    if SEARCH_BACKEND == "faiss":
        from app.core import faiss_search
        return faiss_search.is_ready()
    # auto
    try:
        from app.core import faiss_search
        return faiss_search._check_faiss() and faiss_search.is_ready() and embedding_store.count() >= FAISS_AUTO_THRESHOLD
    except Exception:
        return False


def get_active_backend() -> str:
    return "faiss" if _use_faiss() else "cosine"


def search(
    query_embedding: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    file_type: Optional[str] = None,
    folder: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    extensions: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:

    top_k       = min(top_k, MAX_TOP_K)
    all_emb     = embedding_store.load_all()
    if all_emb is None:
        return []

    total       = len(all_emb)
    valid_idxs  = database.get_all_embedding_indices_with_filters(
        file_type, folder, date_from, date_to, extensions
    )
    if not valid_idxs:
        return []

    valid_idx   = np.array(valid_idxs, dtype=np.int64)
    valid_idx   = valid_idx[valid_idx < total]
    if not len(valid_idx):
        return []

    use_faiss = _use_faiss()
    if use_faiss:
        from app.core import faiss_search
        try:
            result_idx, result_scores = faiss_search.search(query_embedding, valid_idx, top_k)
        except Exception as e:
            logger.warning(f"FAISS failed, falling back: {e}")
            result_idx, result_scores = _cosine(query_embedding, all_emb, valid_idx, top_k)
            use_faiss = False
    else:
        result_idx, result_scores = _cosine(query_embedding, all_emb, valid_idx, top_k)

    media_rows = database.get_media_by_embedding_indices(result_idx.tolist())
    row_map    = {r["embedding_index"]: r for r in media_rows}

    out = []
    for emb_idx, score in zip(result_idx, result_scores):
        row   = row_map.get(int(emb_idx))
        if not row:
            continue
        score = float(score)
        if row.get("width") and row.get("height"):
            ar = row["width"] / row["height"]
            if ar > 3.0 or ar < 0.33:
                score -= SCREENSHOT_ASPECT_PENALTY
        if score < MIN_SIMILARITY_THRESHOLD:
            continue
        out.append({
            **row,
            "score":            round(score, 4),
            "thumbnail_url":    f"/thumbs/{Path(row['thumbnail_path']).name}" if row.get("thumbnail_path") else None,
            "_search_backend":  "faiss" if use_faiss else "cosine",
        })

    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def _cosine(query_embedding, all_emb, valid_idx, top_k):
    total = len(all_emb)
    # If valid_idx represents a large portion of the database, it's faster to do a full memmap dot product
    # and then filter the scores rather than copying all embeddings from the memmap file.
    if len(valid_idx) > total * 0.2:
        scores_all = all_emb @ query_embedding.astype(np.float32)
        scores     = scores_all[valid_idx]
    else:
        filtered = np.array(all_emb[valid_idx], dtype=np.float32)
        scores   = filtered @ query_embedding.astype(np.float32)

    k        = min(top_k, len(scores))
    top_pos  = np.argpartition(scores, -k)[-k:]
    top_pos  = top_pos[np.argsort(scores[top_pos])[::-1]]
    return valid_idx[top_pos], scores[top_pos]
