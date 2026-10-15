#!/usr/bin/env python3
"""
RECALIX V10 — Training CLI
===========================
Run from the training/ directory with venv_gpu active.

Examples:
  python train.py --mode basic      --epochs 5     (verify pipeline works)
  python train.py --mode baseline   --epochs 15    (generate paper Table 1)
  python train.py --mode semi_hard  --epochs 20    (production model)
  python train.py --mode hard       --epochs 30    (after semi_hard converges)

The model is saved to backend/data/fusion_model.pt
Restart the backend after training to load the new model.
"""

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Color helpers ─────────────────────────────────────────────────────────────
R = "\033[0m"; B = "\033[1m"; G = "\033[92m"; Y = "\033[93m"
C = "\033[96m"; D = "\033[2m"; W = "\033[97m"; E = "\033[91m"
LOG_FILE = None
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def save_log_line(line: str):
    if not LOG_FILE:
        return
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(ANSI_RE.sub("", line) + "\n")

def log(msg: str):
    ts    = datetime.now().strftime("%H:%M:%S")
    color = (E if ("ERROR" in msg or "✗" in msg) else
             G if ("✓" in msg or "Done" in msg or "complete" in msg.lower()) else
             Y if ("⚠" in msg or "spike" in msg.lower()) else
             C if "Epoch" in msg and "loss=" in msg else
             B if "===" in msg or "TABLE" in msg else
             D + W)
    line = f"{D}[{ts}]{R} {color}{msg}{R}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    save_log_line(f"[{ts}] {msg}")

def epoch_cb(epoch: int, total: int, loss: float, acc):
    filled  = int(28 * epoch / total)
    bar     = "█" * filled + "░" * (28 - filled)
    acc_str = f"  acc={G}{acc:.3f}{R}" if acc is not None else ""
    print(f"\r  [{C}{bar}{R}] {W}{epoch:3d}/{total}{R}"
          f"  loss={Y}{loss:.5f}{R}{acc_str}",
          end="" if epoch < total else "\n", flush=True)
    save_log_line(f"Epoch {epoch}/{total} loss={loss:.5f}" + (f" acc={acc:.3f}" if acc is not None else ""))

def main():
    global LOG_FILE
    parser = argparse.ArgumentParser(description="RECALIX V10 Training")
    parser.add_argument("--mode",       default="semi_hard",
                        choices=["basic", "baseline", "semi_hard", "hard"])
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--data_dir",   type=str,   default="")
    parser.add_argument("--output_path", type=str,  default="")
    parser.add_argument("--run_id",     type=str,   default="")
    parser.add_argument("--started_at", type=str,   default="")
    args = parser.parse_args()

    print(f"\n{C}{B}  RECALIX V10 — Training CLI{R}")
    print(f"{D}  {'─'*40}{R}")

    # ── Find data directory ───────────────────────────────────────────────────
    script_dir = Path(__file__).resolve().parent
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        for candidate in [
            script_dir.parent / "backend" / "data",
            script_dir.parent.parent / "backend" / "data",
        ]:
            if (candidate / "embeddings.npy").exists():
                data_dir = candidate
                break
        else:
            print(f"{E}ERROR: Cannot find backend/data. Run with --data_dir path/to/backend/data{R}")
            sys.exit(1)

    output_path = Path(args.output_path) if args.output_path else data_dir / "fusion_model.pt"
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = data_dir / "training_logs" / f"terminal_{args.mode}_{run_id}.log"

    # Make backend importable
    backend_dir = data_dir.parent
    for p in [str(backend_dir), str(script_dir.parent)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    print(f"\n  Mode        : {C}{args.mode}{R}")
    print(f"  Epochs      : {W}{args.epochs}{R}")
    print(f"  LR          : {W}{args.lr}{R}")
    print(f"  Batch size  : {W}{args.batch_size}{R}")
    print(f"  Data dir    : {D}{data_dir}{R}")
    print(f"  Output      : {D}{output_path}{R}")
    print(f"  Terminal log: {D}{LOG_FILE}{R}")
    save_log_line(f"RECALIX V10 Training CLI mode={args.mode} epochs={args.epochs} lr={args.lr} batch_size={args.batch_size}")

    # Quick stats
    import json
    meta_f = data_dir / "embeddings_meta.json"
    if meta_f.exists():
        count = json.loads(meta_f.read_text())["count"]
        print(f"  Embeddings  : {G}{count:,}{R}")
    import sqlite3
    conn = sqlite3.connect(str(data_dir / "metadata.db"))
    n = conn.execute("SELECT COUNT(*) FROM media WHERE embedding_index IS NOT NULL").fetchone()[0]
    conn.close()
    print(f"  Media rows  : {G}{n:,}{R}")
    save_log_line(f"Media rows: {n}")

    if n < 10:
        print(f"\n{Y}⚠ Only {n} indexed images. Index more photos for meaningful training.{R}")

    print(f"\n{D}  {'─'*40}{R}")

    t_start = time.time()
    try:
        from modes import get_trainer
        trainer = get_trainer(args.mode)
        def safe_log(msg: str):
            try:
                log(msg)
            except UnicodeEncodeError:
                log(str(msg).encode("ascii", "replace").decode("ascii"))
        result  = trainer.run(
            data_dir    = str(data_dir),
            output_path = str(output_path),
            epochs      = args.epochs,
            lr          = args.lr,
            batch_size  = args.batch_size,
            log_cb      = safe_log,
            epoch_cb    = epoch_cb,
        )
    except KeyboardInterrupt:
        print(f"\n{Y}Training stopped by user.{R}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{E}Training failed: {e}{R}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t_start
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n{D}  {'─'*40}{R}")
    print(f"  {G}✓ Training complete{R}  ({mins}m {secs}s)")
    print(f"  Model → {G}{output_path}{R}")
    save_log_line(f"Training complete ({mins}m {secs}s)")
    save_log_line(f"Model -> {output_path}")
    if result:
        for k, v in result.items():
            if k not in ("mode",):
                print(f"  {k:<20}: {C}{v}{R}")
                save_log_line(f"{k}: {v}")
        artifact_dir = data_dir / "training_logs"
        artifact_dir.mkdir(exist_ok=True)
        artifact = {
            "session_id": run_id,
            "mode": args.mode,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "started_at": args.started_at or None,
            "finished_at": datetime.now().isoformat(),
            "result": result,
            "status": "done",
            "terminal_log_path": str(LOG_FILE),
        }
        import json as _json
        artifact_path = artifact_dir / f"train_{args.mode}_{run_id}.json"
        artifact_path.write_text(_json.dumps(artifact, indent=2), encoding="utf-8")
        save_log_line(f"Training artifact -> {artifact_path}")
    print(f"\n  {D}Restart the backend to load the new model.{R}\n")

if __name__ == "__main__":
    main()
