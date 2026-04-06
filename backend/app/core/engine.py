"""
RECALIX V9 — AI Engine
========================
OpenCLIP model with auto device selection:
  Priority: CUDA GPU → Apple MPS (M-series) → CPU
  Override: RECALIX_DEVICE env var
Thread-safe via lock. Model loaded once at startup via warmup().
"""

import threading
import numpy as np
from typing import List
from PIL import Image

_model      = None
_preprocess = None
_tokenizer  = None
_device     = None
_lock       = threading.Lock()


def _detect_device() -> str:
    from app.config import DEVICE_OVERRIDE
    if DEVICE_OVERRIDE:
        print(f"[Engine] Device override: {DEVICE_OVERRIDE}")
        return DEVICE_OVERRIDE

    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        print(f"[Engine] CUDA GPU: {name} ({vram} GB VRAM)")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[Engine] Apple Silicon MPS detected")
        return "mps"
    print("[Engine] No GPU found — using CPU")
    return "cpu"


def _lazy_load():
    global _model, _preprocess, _tokenizer, _device
    if _model is not None:
        return
    import torch, open_clip
    from app.config import CLIP_MODEL_NAME, CLIP_PRETRAINED
    _device = _detect_device()
    print(f"[Engine] Loading {CLIP_MODEL_NAME} ({CLIP_PRETRAINED}) on {_device} ...")
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=_device
    )
    _model.eval()
    _tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    print(f"[Engine] Ready.")


def get_device() -> str:
    _lazy_load()
    return _device


def embed_images(images: List[Image.Image]) -> np.ndarray:
    """(N, 512) L2-normalized float32 array from a batch of PIL images."""
    _lazy_load()
    import torch
    with _lock:
        tensors = torch.stack([_preprocess(img) for img in images]).to(_device)
        with torch.no_grad():
            features = _model.encode_image(tensors)
        features = features.float().cpu().numpy()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return (features / norms).astype(np.float32)


def embed_text(query: str) -> np.ndarray:
    """(512,) L2-normalized float32 vector from a text query."""
    _lazy_load()
    import torch
    with _lock:
        tokens = _tokenizer([query]).to(_device)
        with torch.no_grad():
            features = _model.encode_text(tokens)
        features = features.float().cpu().numpy()[0]
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
    return features.astype(np.float32)


def warmup():
    _lazy_load()
