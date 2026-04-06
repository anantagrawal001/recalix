"""
RECALIX V9 — Training Routes
==============================
POST /api/training/start          Start training (mode: basic|baseline|semi_hard|hard)
GET  /api/training/status         Current training state
GET  /api/training/stream         SSE — live log stream during training
GET  /api/training/fusion-status  Fusion model load state
POST /api/training/build-faiss    Rebuild FAISS index from current embeddings
DELETE /api/training/model        Delete trained model (revert to fixed weights)
"""

import asyncio
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.hybrid import get_fusion_status
from app.config import FUSION_MODEL_PATH, DATA_DIR

router = APIRouter(prefix="/api/training", tags=["training"])

# ── Shared training state ─────────────────────────────────────────────────────
_train_state = {
    "status":     "idle",   # idle | running | done | error
    "mode":       None,
    "epoch":      0,
    "total_epochs": 0,
    "loss":       None,
    "recall_at5": None,
    "message":    "",
    "started_at": None,
    "finished_at": None,
}
_log_queue:   Queue  = Queue(maxsize=2000)
_train_lock           = threading.Lock()
_train_thread: Optional[threading.Thread] = None


def _push_log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    try:
        _log_queue.put_nowait(entry)
    except Exception:
        pass


def _set_state(**kwargs):
    with _train_lock:
        _train_state.update(kwargs)


class TrainRequest(BaseModel):
    mode: str = "semi_hard"   # basic | baseline | semi_hard | hard
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 64


@router.post("/start")
async def start_training(req: TrainRequest):
    global _train_thread

    valid_modes = {"basic", "baseline", "semi_hard", "hard"}
    if req.mode not in valid_modes:
        raise HTTPException(400, f"mode must be one of {valid_modes}")

    with _train_lock:
        if _train_state["status"] == "running":
            return {"started": False, "message": "Training already in progress"}

    # Clear old logs
    while not _log_queue.empty():
        try: _log_queue.get_nowait()
        except Empty: break

    _set_state(
        status="running", mode=req.mode, epoch=0, total_epochs=req.epochs,
        loss=None, recall_at5=None, message=f"Starting {req.mode} training...",
        started_at=datetime.now().isoformat(), finished_at=None,
    )

    _train_thread = threading.Thread(
        target=_run_training,
        args=(req.mode, req.epochs, req.lr, req.batch_size),
        daemon=True,
    )
    _train_thread.start()
    return {"started": True, "mode": req.mode, "epochs": req.epochs}


def _run_training(mode: str, epochs: int, lr: float, batch_size: int):
    """Runs inside background thread. Imports training module dynamically."""
    try:
        _push_log(f"=== RECALIX V9 Training — mode={mode}, epochs={epochs} ===")
        _push_log(f"Loading DB + embeddings from {DATA_DIR} ...")

        # Import the correct training mode
        import sys, importlib
        from pathlib import Path as P

        train_dir = P(__file__).resolve().parent.parent.parent.parent / "training"
        if str(train_dir) not in sys.path:
            sys.path.insert(0, str(train_dir))

        from modes import get_trainer
        trainer = get_trainer(mode)

        def log_cb(msg: str):
            _push_log(msg)

        def epoch_cb(epoch: int, total: int, loss: float, recall: Optional[float]):
            _set_state(epoch=epoch, total_epochs=total, loss=round(loss, 5), recall_at5=recall)
            _push_log(f"Epoch {epoch}/{total} — loss={loss:.5f}" + (f", Recall@5={recall:.3f}" if recall else ""))

        result = trainer.run(
            data_dir=str(DATA_DIR),
            output_path=str(FUSION_MODEL_PATH),
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            log_cb=log_cb,
            epoch_cb=epoch_cb,
        )

        _push_log(f"=== Training complete. Model saved → {FUSION_MODEL_PATH} ===")
        if result:
            for k, v in result.items():
                _push_log(f"  {k}: {v}")

        # Reload fusion model in hybrid module
        try:
            from app.core import hybrid
            hybrid._fusion_model = None
            hybrid._model_loaded = False
            hybrid._load_fusion_model()
            _push_log("✓ Fusion model reloaded into search engine")
        except Exception as e:
            _push_log(f"⚠ Reload warning: {e}")

        _set_state(status="done", message="Training complete", finished_at=datetime.now().isoformat())

    except Exception as e:
        _push_log(f"ERROR: {e}")
        import traceback
        _push_log(traceback.format_exc())
        _set_state(status="error", message=str(e), finished_at=datetime.now().isoformat())


@router.get("/status")
async def training_status():
    with _train_lock:
        return dict(_train_state)


@router.get("/stream")
async def training_stream():
    """Server-Sent Events — live log lines during training."""
    async def event_gen():
        while True:
            try:
                line = _log_queue.get(timeout=0.5)
                yield f"data: {json.dumps({'log': line})}\n\n"
            except Empty:
                # Heartbeat to keep connection alive
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                with _train_lock:
                    if _train_state["status"] in ("done", "error", "idle"):
                        yield f"data: {json.dumps({'done': True, 'status': _train_state['status']})}\n\n"
                        break
            await asyncio.sleep(0)

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/fusion-status")
async def fusion_status():
    return get_fusion_status()


@router.post("/build-faiss")
async def build_faiss_index():
    """Rebuild FAISS index from current embeddings (run after indexing)."""
    from app.core import faiss_search, embedding_store
    count = embedding_store.count()
    if count == 0:
        raise HTTPException(400, "No embeddings — index some photos first")

    def _build():
        _push_log(f"Building FAISS index for {count} vectors...")
        ok = faiss_search.build_index()
        _push_log("✓ FAISS index built." if ok else "✗ FAISS build failed (faiss-cpu not installed?)")

    threading.Thread(target=_build, daemon=True).start()
    return {"message": f"FAISS index build started for {count} vectors"}


@router.delete("/model")
async def delete_model():
    if FUSION_MODEL_PATH.exists():
        FUSION_MODEL_PATH.unlink()
    # Reset hybrid
    from app.core import hybrid
    hybrid._fusion_model = None
    hybrid._model_loaded = False
    return {"message": "Fusion model deleted. Reverted to fixed weights."}
