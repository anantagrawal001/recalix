"""
RECALIX V9 — Search Routes
GET /api/search?q=...&type=image&folder=...&date_from=...&date_to=...&top_k=50&page=1
"""

import math
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.core import engine, embedding_store
from app.core.similarity import search, get_active_backend
from app.core.hybrid import apply_hybrid_scoring
from app.db import database
from app.config import DEFAULT_TOP_K, MAX_TOP_K, HYBRID_SEARCH_ENABLED

router   = APIRouter(prefix="/api/search", tags=["search"])
PAGE_SIZE = 24


@router.get("")
async def search_media(
    q:         str            = Query(...),
    type:      Optional[str]  = Query(None),
    folder:    Optional[str]  = Query(None),
    date_from: Optional[str]  = Query(None),
    date_to:   Optional[str]  = Query(None),
    top_k:     int            = Query(DEFAULT_TOP_K, le=MAX_TOP_K),
    page:      int            = Query(1, ge=1),
):
    if not q.strip():
        raise HTTPException(400, "Query cannot be empty")

    if embedding_store.count() == 0:
        return {"query": q, "results": [], "total": 0, "page": page, "total_pages": 0,
                "message": "No media indexed yet. Index a directory first."}

    try:
        query_vec = engine.embed_text(q)
    except Exception as e:
        raise HTTPException(500, f"Model error: {e}")

    results = search(
        query_embedding=query_vec,
        top_k=top_k,
        file_type=type if type in ("image", "video") else None,
        folder=folder,
        date_from=date_from,
        date_to=date_to,
    )

    if HYBRID_SEARCH_ENABLED and results:
        results = apply_hybrid_scoring(results, q, query_embedding=query_vec)

    total       = len(results)
    start       = (page - 1) * PAGE_SIZE
    page_results= results[start:start + PAGE_SIZE]

    return {
        "query":        q,
        "results":      page_results,
        "total":        total,
        "page":         page,
        "page_size":    PAGE_SIZE,
        "total_pages":  math.ceil(total / PAGE_SIZE) if total else 0,
        "backend":      get_active_backend(),
    }
