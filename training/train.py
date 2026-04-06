#!/usr/bin/env python3
"""
RECALIX V9 — Training CLI (Research Edition)
============================================

Adds:
• Extensive logging
• Artifact generation
• Graph creation
• CSV experiment tables
• Run metadata
"""

import argparse
import sys
import time
import json
import csv
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────
# Colors
# ──────────────────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"
    DIM    = "\033[2m"
    WHITE  = "\033[97m"


# ──────────────────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────────────────

def banner():
    print(f"""
{C.CYAN}{C.BOLD}
  ██████╗ ███████╗ ██████╗ █████╗ ██╗     ██╗██╗  ██╗
  ██╔══██╗██╔════╝██╔════╝██╔══██╗██║     ██║╚██╗██╔╝
  ██████╔╝█████╗  ██║     ███████║██║     ██║ ╚███╔╝
  ██╔══██╗██╔══╝  ██║     ██╔══██║██║     ██║ ██╔██╗
  ██║  ██║███████╗╚██████╗██║  ██║███████╗██║██╔╝ ██╗
  ╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═╝
{C.RESET}{C.DIM}  V9 — Query-Conditioned Fusion Training{C.RESET}
""")


# ──────────────────────────────────────────────────────────
# Artifact directories
# ──────────────────────────────────────────────────────────

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

ROOT = Path(__file__).parent

ART_DIR = ROOT / "artifacts" / f"run_{RUN_ID}"
LOG_DIR = ART_DIR / "logs"
GRAPH_DIR = ART_DIR / "graphs"
TABLE_DIR = ART_DIR / "tables"

for d in [LOG_DIR, GRAPH_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "training.log"
CSV_FILE = TABLE_DIR / "epoch_metrics.csv"

history_epochs = []
history_loss = []


# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────

def log(msg: str):

    ts = datetime.now().strftime("%H:%M:%S")

    line = f"[{ts}] {msg}"

    print(line)

    with open(LOG_FILE, "a", encoding="utf8") as f:
        f.write(line + "\n")


# ──────────────────────────────────────────────────────────
# Epoch callback
# ──────────────────────────────────────────────────────────

def epoch_cb(epoch, total, loss, recall):

    history_epochs.append(epoch)
    history_loss.append(loss)

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, loss])

    bar_len = 30
    filled  = int(bar_len * epoch / total)

    bar = "█" * filled + "░" * (bar_len - filled)

    print(
        f"\r  [{bar}] {epoch}/{total} loss={loss:.5f}",
        end="" if epoch < total else "\n",
        flush=True
    )


# ──────────────────────────────────────────────────────────
# Graph generation
# ──────────────────────────────────────────────────────────

def save_graphs():

    if not history_epochs:
        return

    plt.figure(figsize=(8,5))

    plt.plot(history_epochs, history_loss)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.grid(True)

    plt.savefig(GRAPH_DIR / "loss_curve.png")

    plt.close()


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="semi_hard",
                        choices=["basic","baseline","semi_hard","hard"])

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--data_dir", type=str, default="")

    args = parser.parse_args()

    banner()

    script_dir = Path(__file__).resolve().parent

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:

        candidates = [
            script_dir.parent / "backend" / "data",
            script_dir.parent.parent / "backend" / "data",
            script_dir / "data"
        ]

        data_dir = None

        for c in candidates:
            if (c / "embeddings.npy").exists():
                data_dir = c
                break

        if data_dir is None:
            print("ERROR: Cannot find backend/data directory")
            sys.exit(1)

    output_path = data_dir / "fusion_model.pt"

    backend_dir = data_dir.parent

    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    modes_dir = script_dir / "modes"

    if str(modes_dir.parent) not in sys.path:
        sys.path.insert(0, str(modes_dir.parent))

    log("Training configuration")
    log(f"Mode: {args.mode}")
    log(f"Epochs: {args.epochs}")
    log(f"Batch size: {args.batch_size}")
    log(f"Learning rate: {args.lr}")
    log(f"Data directory: {data_dir}")
    log(f"Output model: {output_path}")

    import sqlite3

    db_path = data_dir / "metadata.db"

    conn = sqlite3.connect(str(db_path))

    n_media = conn.execute(
        "SELECT COUNT(*) FROM media WHERE embedding_index IS NOT NULL"
    ).fetchone()[0]

    conn.close()

    log(f"Indexed images: {n_media}")

    t_start = time.time()

    try:

        from modes import get_trainer

        trainer = get_trainer(args.mode)

        result = trainer.run(
            data_dir=str(data_dir),
            output_path=str(output_path),
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            log_cb=log,
            epoch_cb=epoch_cb
        )

    except KeyboardInterrupt:

        print("Training interrupted")
        sys.exit(0)

    elapsed = time.time() - t_start

    save_graphs()

    meta = {
        "run_id": RUN_ID,
        "mode": args.mode,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "time_seconds": elapsed
    }

    with open(ART_DIR / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nTraining complete")

    print(f"Artifacts saved to:\n{ART_DIR}")


if __name__ == "__main__":
    main()