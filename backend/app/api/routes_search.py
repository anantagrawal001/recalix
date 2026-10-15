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
    file_type: Optional[str]  = Query(None, alias="type"),
    folder:    Optional[str]  = Query(None),
    date_from: Optional[str]  = Query(None),
    date_to:   Optional[str]  = Query(None),
    ext:       Optional[str]  = Query(None),
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

    ext_list = None
    if ext:
        ext_list = [e.strip().lower() for e in ext.split(",") if e.strip()]

    results = search(
        query_embedding=query_vec,
        top_k=top_k,
        file_type=file_type if file_type in ("image", "video") else None,
        folder=folder,
        date_from=date_from,
        date_to=date_to,
        extensions=ext_list,
    )

    if HYBRID_SEARCH_ENABLED and results:
        # Log which scoring method is active (visible in backend terminal)
        from app.core.hybrid import get_fusion_status
        _fs = get_fusion_status()
        if _fs["fusion_active"]:
            import logging
            logging.getLogger("recalix.search").debug(
                f"[Search] '{q}' → {len(results)} results, "
                f"fusion=ON (device={_fs['device']}), backend={get_active_backend()}"
            )
        results = apply_hybrid_scoring(results, q, query_embedding=query_vec)

    total       = len(results)
    start       = (page - 1) * PAGE_SIZE
    page_results= results[start:start + PAGE_SIZE]

    # Save search history (10-day retention)
    try:
        database.add_search_history(
            query=q.strip(),
            folder=folder,
            date_from=date_from,
            date_to=date_to,
            extensions=ext,
            retention_days=10,
        )
    except Exception:
        pass

    return {
        "query":        q,
        "results":      page_results,
        "total":        total,
        "page":         page,
        "page_size":    PAGE_SIZE,
        "total_pages":  math.ceil(total / PAGE_SIZE) if total else 0,
        "backend":      get_active_backend(),
    }


@router.get("/history")
async def get_search_history(limit: int = Query(3, ge=1, le=50)):
    items = database.get_search_history(limit=limit)
    return {"items": items}


@router.delete("/history")
async def clear_search_history():
    deleted = database.clear_search_history()
    return {"deleted": deleted}


@router.delete("/history/{item_id}")
async def delete_search_history(item_id: int):
    deleted = database.delete_search_history_item(item_id)
    if deleted == 0:
        raise HTTPException(404, "History item not found")
    return {"deleted": deleted}
