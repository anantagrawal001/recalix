"""
RECALIX V10 — AI Engine
========================
CLIP model with auto GPU detection.
Supports RTX 5060 (Blackwell sm_120) via PyTorch cu128.
Falls back to CPU if CUDA kernel fails.
"""

import threading
import numpy as np
from typing import List
from PIL import Image

_model       = None
_preprocess  = None
_tokenizer   = None
_device      = None
_lock        = threading.Lock()
_cuda_broken = False


def _detect_device() -> str:
    from app.config import DEVICE_OVERRIDE
    if DEVICE_OVERRIDE:
        print(f"[Engine] Device override: {DEVICE_OVERRIDE}")
        return DEVICE_OVERRIDE

    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        cap  = torch.cuda.get_device_capability(0)
        print(f"[Engine] CUDA GPU: {name} ({vram} GB VRAM, compute sm_{cap[0]}{cap[1]})")
        # Test actual kernel execution — catches Blackwell (sm_120) with wrong PyTorch
        try:
            t = torch.zeros(1).cuda()
            _ = t + 1
            del t, _
            torch.cuda.empty_cache()
            print(f"[Engine] sm_{cap[0]}{cap[1]} CUDA test passed.")
            return "cuda"
        except Exception as e:
            print(f"[Engine] CUDA kernel failed: {str(e)[:100]}")
            print(f"[Engine] RTX 5060 needs PyTorch cu128. Run setup_gpu.bat.")
            print(f"[Engine] Falling back to CPU.")
            return "cpu"

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
    print(f"[Engine] Ready on {_device}.")


def _cpu_fallback(err: Exception):
    global _device, _cuda_broken
    if _cuda_broken:
        return
    _cuda_broken = True
    _device = "cpu"
    if _model is not None:
        try:
            _model.cpu()
        except Exception:
            pass
    print(f"[Engine] CUDA runtime error — switched to CPU.")
    print(f"[Engine] Fix: run setup_gpu.bat to install correct PyTorch.")


def get_device() -> str:
    _lazy_load()
    return _device


def embed_images(images: List[Image.Image]) -> np.ndarray:
    """Convert list of PIL images to (N, 512) L2-normalized float32 array."""
    _lazy_load()
    import torch

    with _lock:
        for attempt in range(2):
            try:
                dev     = _device
                tensors = torch.stack([_preprocess(img) for img in images]).to(dev)
                with torch.no_grad():
                    features = _model.encode_image(tensors)
                features = features.float().cpu().numpy()
                if dev == "cuda":
                    torch.cuda.empty_cache()
                break
            except Exception as e:
                if attempt == 0 and ("no kernel image" in str(e) or "CUDA error" in str(e)):
                    _cpu_fallback(e)
                    continue
                raise

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (features / norms).astype(np.float32)


def embed_text(query: str) -> np.ndarray:
    """Convert query string to (512,) L2-normalized float32 vector."""
    _lazy_load()
    import torch

    with _lock:
        for attempt in range(2):
            try:
                dev    = _device
                tokens = _tokenizer([query]).to(dev)
                with torch.no_grad():
                    features = _model.encode_text(tokens)
                features = features.float().cpu().numpy()[0]
                break
            except Exception as e:
                if attempt == 0 and ("no kernel image" in str(e) or "CUDA error" in str(e)):
                    _cpu_fallback(e)
                    continue
                raise

    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
    return features.astype(np.float32)


def embed_texts(queries: List[str]) -> np.ndarray:
    """Convert list of query strings to (N, 512) L2-normalized float32 matrix."""
    if not queries:
        return np.zeros((0, 512), dtype=np.float32)
    _lazy_load()
    import torch

    batch_size = 512
    all_features = []

    with _lock:
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            for attempt in range(2):
                try:
                    dev    = _device
                    tokens = _tokenizer(batch).to(dev)
                    with torch.no_grad():
                        features = _model.encode_text(tokens)
                    features = features.float().cpu().numpy()
                    all_features.append(features)
                    break
                except Exception as e:
                    if attempt == 0 and ("no kernel image" in str(e) or "CUDA error" in str(e)):
                        _cpu_fallback(e)
                        continue
                    raise

    features = np.concatenate(all_features, axis=0)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (features / norms).astype(np.float32)


def warmup():
    _lazy_load()
