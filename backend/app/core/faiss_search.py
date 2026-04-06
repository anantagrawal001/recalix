"""
RECALIX V9 — FAISS Search Index
=================================
ANN search using FAISS. Auto-selects index type by collection size:
  < 10K  → IndexFlatIP  (exact, inner product)
  ≥ 10K  → IndexIVFFlat (approximate, faster)

Falls back gracefully to cosine if faiss not installed.
Embeddings are pre-L2-normalized → inner product == cosine similarity.
"""

import logging
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("recalix.faiss")

_index      = None
_index_lock = threading.Lock()
_available  = None


def _check_faiss() -> bool:
    global _available
    if _available is not None:
        return _available
    try:
        import faiss; _available = True  # noqa
    except ImportError:
        _available = False
        logger.warning("faiss-cpu not installed. `pip install faiss-cpu` to enable.")
    return _available


def is_ready() -> bool:
    return _index is not None and _check_faiss()


def build_index(embeddings: Optional[np.ndarray] = None) -> bool:
    global _index
    if not _check_faiss():
        return False
    import faiss
    from app.config import FAISS_INDEX_PATH, EMBEDDING_DIM
    from app.core import embedding_store

    try:
        if embeddings is None:
            embeddings = embedding_store.load_all()
        if embeddings is None or len(embeddings) == 0:
            logger.info("No embeddings — FAISS index not built.")
            return False

        n, d = embeddings.shape
        emb = embeddings.astype(np.float32)
        logger.info(f"Building FAISS index: {n} vectors, dim={d}")

        if n < 10_000:
            index = faiss.IndexFlatIP(d)
        else:
            nlist = min(int(n ** 0.5), 256)
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            logger.info(f"Training IVF index (nlist={nlist})...")
            index.train(emb)

        index.add(emb)
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        logger.info(f"FAISS index saved → {FAISS_INDEX_PATH}")

        with _index_lock:
            _index = index
        return True

    except Exception as e:
        logger.error(f"FAISS build_index failed: {e}")
        return False


def load_index() -> bool:
    global _index
    if not _check_faiss():
        return False
    import faiss
    from app.config import FAISS_INDEX_PATH
    if not FAISS_INDEX_PATH.exists():
        return False
    try:
        with _index_lock:
            _index = faiss.read_index(str(FAISS_INDEX_PATH))
        logger.info(f"FAISS index loaded ({_index.ntotal} vectors)")
        return True
    except Exception as e:
        logger.error(f"FAISS load_index failed: {e}")
        return False


def search(
    query_embedding: np.ndarray,
    valid_indices: Optional[np.ndarray],
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (embedding_indices, scores) sorted by score descending.
    Searches top_k*6 candidates then filters to valid_indices.
    """
    if not is_ready():
        raise RuntimeError("FAISS index not ready")

    qvec   = query_embedding.astype(np.float32).reshape(1, -1)
    search_k = min(top_k * 6, _index.ntotal) if valid_indices is not None else top_k

    if hasattr(_index, "nprobe"):
        _index.nprobe = min(64, getattr(_index, "nlist", 64))

    scores_raw, idx_raw = _index.search(qvec, search_k)
    scores_raw = scores_raw[0]; idx_raw = idx_raw[0]

    # Remove FAISS invalid (-1) results
    ok = idx_raw >= 0
    idx_raw = idx_raw[ok]; scores_raw = scores_raw[ok]

    if valid_indices is not None and len(valid_indices) > 0:
        vset = set(valid_indices.tolist())
        keep = np.array([i in vset for i in idx_raw], dtype=bool)
        idx_raw = idx_raw[keep]; scores_raw = scores_raw[keep]

    if len(idx_raw) > top_k:
        order = np.argsort(scores_raw)[::-1][:top_k]
        idx_raw = idx_raw[order]; scores_raw = scores_raw[order]

    return idx_raw, scores_raw


def invalidate():
    global _index
    with _index_lock:
        _index = None
    from app.config import FAISS_INDEX_PATH
    if FAISS_INDEX_PATH.exists():
        FAISS_INDEX_PATH.unlink()
    logger.info("FAISS index invalidated.")
