"""
Training routes.

Long jobs are launched as separate Python processes so browser refreshes and
heavy training/index work do not depend on the FastAPI request process.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.config import DATA_DIR, FUSION_MODEL_PATH
from app.core.hybrid import get_fusion_status

router = APIRouter(prefix="/api/training", tags=["training"])

_train_state = {
    "status": "idle",
    "mode": None,
    "epoch": 0,
    "total_epochs": 0,
    "loss": None,
    "recall_at5": None,
    "message": "",
    "started_at": None,
    "finished_at": None,
    "session_id": None,
    "pid": None,
    "terminal_log_path": None,
}
_log_queue: Queue = Queue(maxsize=2000)
_STATE_FILE = DATA_DIR / "training_status.json"
_EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+).*?loss=([0-9.]+)(?:.*?(?:Recall@5|val_acc|acc)=([0-9.]+))?")
_train_process: Optional[subprocess.Popen] = None


def _creation_flags() -> int:
    if sys.platform != "win32":
        return 0
    return getattr(subprocess, "CREATE_NEW_CONSOLE", 0)


def _python_executable() -> str:
    root_dir = Path(__file__).resolve().parents[3]
    venv_python = root_dir / "venv_gpu" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not process:
                return False
            try:
                code = ctypes.c_ulong()
                if not ctypes.windll.kernel32.GetExitCodeProcess(process, ctypes.byref(code)):
                    return False
                return code.value == 259
            finally:
                ctypes.windll.kernel32.CloseHandle(process)
        except Exception:
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except Exception:
        return False


def _save_state() -> None:
    try:
        DATA_DIR.mkdir(exist_ok=True)
        tmp = _STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_train_state, indent=2), encoding="utf-8")
        tmp.replace(_STATE_FILE)
    except Exception:
        pass


def _load_state() -> None:
    if not _STATE_FILE.exists():
        return
    try:
        data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        _train_state.update(data)
    except Exception:
        pass


def _push_log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] {msg}"
    try:
        _log_queue.put_nowait(entry)
    except Exception:
        pass
    log_path = _train_state.get("terminal_log_path")
    if log_path:
        try:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(entry + "\n")
        except Exception:
            pass


def _set_state(**kwargs) -> None:
    _train_state.update(kwargs)
    _save_state()


def _parse_log_progress(log_path: Optional[Path]) -> None:
    if not log_path or not log_path.exists():
        return
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-120:]
    except Exception:
        return
    for line in lines:
        m = _EPOCH_RE.search(line)
        if not m:
            continue
        epoch, total, loss, acc = m.groups()
        _train_state.update(
            epoch=int(epoch),
            total_epochs=int(total),
            loss=round(float(loss), 5),
            recall_at5=float(acc) if acc is not None else _train_state.get("recall_at5"),
        )


def _artifact_path() -> Optional[Path]:
    session_id = _train_state.get("session_id")
    mode = _train_state.get("mode")
    if not session_id or not mode:
        return None
    return DATA_DIR / "training_logs" / f"train_{mode}_{session_id}.json"


def _reload_fusion_model() -> None:
    try:
        from app.core import hybrid

        hybrid._fusion_model = None
        hybrid._model_loaded = False
        hybrid._load_fusion_model()
    except Exception as e:
        _push_log(f"Fusion reload warning: {e}")


def _sync_training_state() -> None:
    _load_state()
    log_path = Path(_train_state["terminal_log_path"]) if _train_state.get("terminal_log_path") else None
    _parse_log_progress(log_path)

    if _train_state.get("status") == "running":
        artifact = _artifact_path()
        if artifact and artifact.exists():
            _set_state(status="done", message="Training complete", finished_at=datetime.now().isoformat())
            _reload_fusion_model()
        elif not _pid_alive(_train_state.get("pid")):
            _set_state(
                status="error",
                message="Training process stopped before writing an artifact",
                finished_at=datetime.now().isoformat(),
            )
    else:
        _save_state()


class TrainRequest(BaseModel):
    mode: str = "semi_hard"
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 64


@router.post("/start")
async def start_training(req: TrainRequest):
    global _train_process

    valid_modes = {"basic", "baseline", "semi_hard", "hard"}
    if req.mode not in valid_modes:
        raise HTTPException(400, f"mode must be one of {valid_modes}")

    _sync_training_state()
    if _train_state.get("status") == "running" and _pid_alive(_train_state.get("pid")):
        return {"started": False, "message": "Training already in progress", "session_id": _train_state.get("session_id")}

    while not _log_queue.empty():
        try:
            _log_queue.get_nowait()
        except Empty:
            break

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    started_at = datetime.now().isoformat()
    log_path = DATA_DIR / "training_logs" / f"terminal_{req.mode}_{session_id}.log"
    _set_state(
        status="running",
        mode=req.mode,
        epoch=0,
        total_epochs=req.epochs,
        loss=None,
        recall_at5=None,
        message=f"Starting {req.mode} training...",
        started_at=started_at,
        finished_at=None,
        session_id=session_id,
        pid=None,
        terminal_log_path=str(log_path),
    )

    root_dir = Path(__file__).resolve().parents[3]
    cmd = [
        _python_executable(),
        str(root_dir / "training" / "train.py"),
        "--mode",
        req.mode,
        "--epochs",
        str(req.epochs),
        "--lr",
        str(req.lr),
        "--batch_size",
        str(req.batch_size),
        "--data_dir",
        str(DATA_DIR),
        "--run_id",
        session_id,
        "--started_at",
        started_at,
    ]
    try:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        _train_process = subprocess.Popen(
            cmd,
            cwd=str(root_dir / "training"),
            creationflags=_creation_flags(),
            env=env,
        )
        _set_state(pid=_train_process.pid, message=f"{req.mode} training process started.")
        return {"started": True, "mode": req.mode, "epochs": req.epochs, "session_id": session_id}
    except Exception as e:
        _set_state(status="error", message=f"Could not start training process: {e}", finished_at=datetime.now().isoformat())
        raise HTTPException(500, str(e))


@router.get("/status")
async def training_status():
    _sync_training_state()
    return dict(_train_state)


@router.get("/stream")
async def training_stream():
    async def event_gen():
        _sync_training_state()
        log_path = Path(_train_state["terminal_log_path"]) if _train_state.get("terminal_log_path") else None
        pos = 0
        while True:
            if log_path and log_path.exists():
                try:
                    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
                        f.seek(pos)
                        lines = f.readlines()
                        pos = f.tell()
                    for line in lines:
                        line = line.rstrip()
                        if line:
                            yield f"data: {json.dumps({'log': line})}\n\n"
                except Exception:
                    pass

            try:
                line = _log_queue.get(timeout=0.25)
                yield f"data: {json.dumps({'log': line})}\n\n"
            except Empty:
                _sync_training_state()
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                if _train_state["status"] in ("done", "error", "idle"):
                    yield f"data: {json.dumps({'done': True, 'status': _train_state['status']})}\n\n"
                    break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/fusion-status")
async def fusion_status():
    return get_fusion_status()


@router.post("/build-faiss")
async def build_faiss_index():
    from app.core import embedding_store

    count = embedding_store.count()
    if count == 0:
        raise HTTPException(400, "No embeddings - index some photos first")

    log_dir = DATA_DIR / "training_logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"faiss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    backend_dir = Path(__file__).resolve().parents[2]
    cmd = [_python_executable(), "-m", "app.core.faiss_cli"]
    with log_path.open("a", encoding="utf-8") as out:
        subprocess.Popen(
            cmd,
            cwd=str(backend_dir),
            stdout=out,
            stderr=subprocess.STDOUT,
            creationflags=_creation_flags(),
        )
    return {"message": f"FAISS index build started for {count} vectors", "log_path": str(log_path)}


@router.get("/artifacts")
async def list_artifacts():
    artifacts_dir = DATA_DIR / "training_logs"
    if not artifacts_dir.exists():
        return {"artifacts": []}
    arts = []
    for f in sorted(artifacts_dir.glob("train_*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("terminal_log_path"):
                data["terminal_log_url"] = f"/api/training/artifacts/{data.get('session_id')}/terminal.log"
            result = data.get("result") or {}
            artifact_dir = result.get("artifact_dir")
            if artifact_dir:
                run_dir = Path(artifact_dir)
                summary_path = run_dir / "summary.json"
                if summary_path.exists():
                    try:
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        data["summary"] = summary
                        data["metrics"] = summary.get("metrics", [])
                        data["baselines"] = summary.get("baselines", {})
                        data["process"] = summary.get("process", [])
                    except Exception:
                        pass
                if (run_dir / "learning_curve.svg").exists():
                    data["learning_curve_url"] = f"/api/training/artifacts/{data.get('session_id')}/learning-curve.svg"
                elif (run_dir / "learning_curve.png").exists():
                    data["learning_curve_url"] = f"/api/training/artifacts/{data.get('session_id')}/learning-curve.png"
            arts.append(data)
        except Exception:
            pass
    arts.sort(key=lambda a: a.get("finished_at") or a.get("started_at") or a.get("session_id") or "", reverse=True)
    return {"artifacts": arts}


@router.get("/artifacts/{session_id}/terminal.log")
async def artifact_terminal_log(session_id: str):
    artifacts_dir = DATA_DIR / "training_logs"
    if not artifacts_dir.exists():
        raise HTTPException(404, "No training artifacts found")
    for f in artifacts_dir.glob("train_*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("session_id") != session_id:
            continue
        log_path = data.get("terminal_log_path")
        if log_path and Path(log_path).exists():
            return FileResponse(str(log_path), media_type="text/plain")
        fallback = artifacts_dir / f"terminal_{data.get('mode', 'train')}_{session_id}.log"
        if fallback.exists():
            return FileResponse(str(fallback), media_type="text/plain")
        break
    raise HTTPException(404, "Terminal log not found")


@router.get("/artifacts/{session_id}/learning-curve.{ext}")
async def artifact_learning_curve(session_id: str, ext: str):
    if ext not in {"svg", "png"}:
        raise HTTPException(404, "Unsupported graph format")

    artifacts_dir = DATA_DIR / "training_logs"
    if not artifacts_dir.exists():
        raise HTTPException(404, "No training artifacts found")

    for f in artifacts_dir.glob("train_*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("session_id") != session_id:
            continue

        result = data.get("result") or {}
        artifact_dir = result.get("artifact_dir")
        if not artifact_dir:
            break
        graph_path = Path(artifact_dir) / f"learning_curve.{ext}"
        if graph_path.exists() and graph_path.is_file():
            media_type = "image/svg+xml" if ext == "svg" else "image/png"
            return FileResponse(str(graph_path), media_type=media_type)
        break

    raise HTTPException(404, "Learning curve graph not found")


@router.delete("/model")
async def delete_model():
    if FUSION_MODEL_PATH.exists():
        FUSION_MODEL_PATH.unlink()
    from app.core import hybrid

    hybrid._fusion_model = None
    hybrid._model_loaded = False
    return {"message": "Fusion model deleted. Reverted to fixed weights."}
