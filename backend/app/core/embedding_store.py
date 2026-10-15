"""
RECALIX Embedding Store — Production Grade
==========================================
Fixes over v1:
- Capacity-based growth (no file recreation per append)
- Atomic swap on resize
- Embedding version tracking (model name stored)
- Thread-safe writes
- Crash-safe: only updates count after successful write
"""

import json
import threading
import numpy as np
from typing import List, Optional

from app.config import EMBEDDINGS_PATH, EMBEDDING_DIM, DATA_DIR

_lock = threading.Lock()

_DATA_FILE = EMBEDDINGS_PATH
_META_FILE = DATA_DIR / "embeddings_meta.json"

_GROWTH_FACTOR = 1.5
_INITIAL_CAPACITY = 100000


def _load_meta():
    default = {
        "count": 0,
        "capacity": _INITIAL_CAPACITY,
        "model": None
    }

    if not _META_FILE.exists():
        return default

    try:
        text = _META_FILE.read_text().strip()
        if text == "":
            return default

        meta = json.loads(text)

        # ensure required keys exist
        meta.setdefault("count", 0)
        meta.setdefault("capacity", _INITIAL_CAPACITY)
        meta.setdefault("model", None)

        return meta

    except Exception:
        return default


def _save_meta(meta: dict):
    _META_FILE.write_text(json.dumps(meta, indent=2))


def _open_map(capacity: int, mode: str) -> np.memmap:
    return np.memmap(
        str(_DATA_FILE), dtype=np.float32, mode=mode, shape=(capacity, EMBEDDING_DIM)
    )


def _grow(old_meta: dict, needed: int) -> dict:
    """
    Grow memmap file capacity. Uses atomic temp-file swap to prevent corruption.
    """
    old_count = old_meta["count"]
    old_capacity = old_meta["capacity"]
    new_capacity = max(
        int(old_capacity * _GROWTH_FACTOR),
        old_capacity + needed + _INITIAL_CAPACITY,
    )
    tmp_path = _DATA_FILE.with_suffix(".tmp.npy")
    new_fp = np.memmap(str(tmp_path), dtype=np.float32, mode="w+", shape=(new_capacity, EMBEDDING_DIM))
    if old_count > 0 and _DATA_FILE.exists():
        old_fp = _open_map(old_capacity, "r")
        new_fp[:old_count] = old_fp[:old_count]
        del old_fp
    new_fp.flush()
    del new_fp
    tmp_path.replace(_DATA_FILE)
    return {"count": old_count, "capacity": new_capacity, "model": old_meta["model"]}


def count() -> int:
    return _load_meta()["count"]


def get_model_version() -> Optional[str]:
    return _load_meta().get("model")


def set_model_version(model_name: str):
    with _lock:
        meta = _load_meta()
        meta["model"] = model_name
        _save_meta(meta)


def load_all() -> Optional[np.ndarray]:
    """Return memory-mapped view of all live embeddings. Shape: (N, D)."""
    meta = _load_meta()
    n, cap = meta["count"], meta["capacity"]
    if n == 0 or cap == 0 or not _DATA_FILE.exists():
        return None
    fp = _open_map(cap, "r")
    return fp[:n]


def append_embeddings(embeddings: np.ndarray, model_name: str) -> List[int]:
    """
    Append (M, D) normalized embeddings. Grows file if needed.
    Returns new indices. Thread-safe. Crash-safe (count written after flush).
    """
    m = len(embeddings)
    if m == 0:
        return []
    with _lock:
        meta = _load_meta()
        n = meta["count"]
        cap = meta["capacity"]
        # create memmap file if it doesn't exist
        if not _DATA_FILE.exists():
            fp = np.memmap(str(_DATA_FILE), dtype=np.float32, mode="w+", shape=(cap, EMBEDDING_DIM))
            fp.flush()
            del fp
        if cap == 0 or n + m > cap:
            meta = _grow(meta, m)
            cap = meta["capacity"]
        fp = _open_map(cap, "r+")
        fp[n : n + m] = embeddings
        fp.flush()
        del fp
        meta["count"] = n + m
        meta["model"] = model_name
        _save_meta(meta)
        return list(range(n, n + m))


def clear():
    if _DATA_FILE.exists():
        _DATA_FILE.unlink()
    if _META_FILE.exists():
        _META_FILE.unlink()
