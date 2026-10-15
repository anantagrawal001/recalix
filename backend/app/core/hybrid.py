"""
RECALIX V9 — Hybrid Scoring Engine
=====================================
Combines 4 signals with query-adaptive weights from the trained fusion model.

If fusion_model.pt exists  → weights predicted per query  (research mode)
If fusion_model.pt missing → fixed heuristic weights       (production fallback)

Signals:
  visual     — CLIP cosine similarity (raw search score)
  text       — OCR keyword match
  filename   — Filename token match
  structural — Document/screenshot heuristics
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger("recalix.hybrid")

FALLBACK_WEIGHTS = {
    "visual":     0.55,
    "text":       0.30,
    "filename":   0.10,
    "structural": 0.05,
}

_fusion_model  = None
_model_loaded  = False
_model_device  = "cpu"


def _load_fusion_model():
    global _fusion_model, _model_loaded, _model_device
    if _model_loaded:
        return
    _model_loaded = True

    from app.config import FUSION_ENABLED, FUSION_MODEL_PATH, FUSION_MODEL_OVERRIDE
    from pathlib import Path

    if not FUSION_ENABLED:
        logger.info("Fusion model disabled by env var")
        return

    model_path = Path(FUSION_MODEL_OVERRIDE) if FUSION_MODEL_OVERRIDE else FUSION_MODEL_PATH
    if not model_path.exists():
        logger.info(f"No fusion model at {model_path} — using fixed weights")
        return

    try:
        import torch
        from app.core.fusion_model import load_model
        _model_device = "cuda" if torch.cuda.is_available() else "cpu"
        _fusion_model = load_model(model_path, _model_device)
        if _fusion_model:
            logger.info(f"✓ Fusion model loaded ({model_path.name}, device={_model_device})")
    except Exception as e:
        logger.error(f"Fusion model load failed: {e} — using fixed weights")


def get_fusion_status() -> Dict:
    _load_fusion_model()
    from app.config import FUSION_MODEL_PATH
    return {
        "fusion_active":  _fusion_model is not None,
        "model_path":     str(FUSION_MODEL_PATH),
        "model_exists":   FUSION_MODEL_PATH.exists(),
        "device":         _model_device if _fusion_model else None,
        "fallback_weights": FALLBACK_WEIGHTS,
    }


def _predict_weights(query_embedding: Optional[np.ndarray]) -> Dict[str, float]:
    _load_fusion_model()
    if _fusion_model is not None and query_embedding is not None:
        try:
            return _fusion_model.predict_weights(query_embedding)
        except Exception as e:
            logger.warning(f"Fusion predict failed: {e}")
    return dict(FALLBACK_WEIGHTS)


# ── Signal scorers ────────────────────────────────────────────────────────────

def _text_score(query: str, ocr_text: str) -> float:
    if not ocr_text:
        return 0.0
    q = query.lower(); o = ocr_text.lower()
    if q in o:
        return 1.0
    qt = set(q.split()); ot = o.split(); ots = set(ot)
    if not qt:
        return 0.0
    m = len(qt & ots)
    if not m:
        return 0.0
    score = m / len(qt)
    for i in range(len(ot) - len(qt) + 1):
        if q in " ".join(ot[i:i+len(qt)]):
            score = min(score * 1.3, 1.0); break
    return min(score, 1.0)


def _filename_score(filename_tokens: str, query: str) -> float:
    if not filename_tokens:
        return 0.0
    qt = set(query.lower().split())
    ft = set(filename_tokens.lower().split())
    if not qt:
        return 0.0
    return min(len(qt & ft) / len(qt), 1.0)


def _structural_score(row: Dict[str, Any], query: str = "") -> float:
    """
    Structural heuristic signal. Range: [-0.2, +0.3].
    No +0.5 bias — only real discriminative signals.
    Matches V5 behaviour: conditional document boost, screenshot penalty.
    """
    from app.config import TINY_IMAGE_THRESHOLD
    score = 0.0
    q = query.lower()

    # Document boost — only for document-type queries
    doc_kw = {"document", "invoice", "receipt", "scan", "paper", "form", "certificate"}
    if any(kw in q for kw in doc_kw):
        if row.get("is_document"):
            score += 0.3

    # Screenshot penalty — unless user is asking for screenshots
    if row.get("is_screenshot") and "screenshot" not in q:
        score -= 0.1

    # Tiny image penalty
    pa = row.get("pixel_area")
    if pa and pa < TINY_IMAGE_THRESHOLD:
        if not any(kw in q for kw in {"icon", "emoji", "logo"}):
            score -= 0.15

    return max(score, -0.2)


# ── Main ──────────────────────────────────────────────────────────────────────

def apply_hybrid_scoring(
    results: List[Dict[str, Any]],
    query: str,
    query_embedding: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Re-score results list using hybrid signals + adaptive weights."""
    w = _predict_weights(query_embedding)

    for item in results:
        v = float(item.get("score", 0.0))
        t = _text_score(query, item.get("ocr_text") or "")
        f = _filename_score(item.get("filename_tokens") or "", query)
        s = _structural_score(item)

        # Normalize signals to a comparable [0.0, 1.0] scale
        v_norm = float(np.clip((v - 0.15) / 0.20, 0.0, 1.0))
        t_norm = float(np.clip(t, 0.0, 1.0))
        f_norm = float(np.clip(f, 0.0, 1.0))
        s_norm = float(np.clip((s + 0.2) / 0.5, 0.0, 1.0))

        item["score"]    = round(w["visual"]*v_norm + w["text"]*t_norm + w["filename"]*f_norm + w["structural"]*s_norm, 4)
        item["_signals"] = {"visual": round(v,4), "text": round(t,4), "filename": round(f,4), "structural": round(s,4)}
        item["_weights"] = {k: round(wv, 4) for k, wv in w.items()}

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
