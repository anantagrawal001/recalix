"""
RECALIX Training — Baseline Mode  (Pipeline Edition)
=====================================================
What changed vs the broken version:
  1. Example creation shows live progress every PROGRESS_EVERY images.
  2. Metadata is split into chunks; examples are generated chunk-by-chunk.
  3. Training starts on chunk-1 examples while chunk-2 is still being built
     (producer thread + queue — true pipeline parallelism).
  4. Full 4-signal computation restored (visual / text / filename / structural).
  5. Artifact verified on disk (path + size) after every save.

Tuning knobs at the top of the file — no need to edit the class.
"""

import math
import random
import threading
import queue as _queue
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from modes.base import BaseTrainer, evaluate


# ──────────────────────────────────────────────────────────────
#  TUNING KNOBS
# ──────────────────────────────────────────────────────────────
CHUNK_IMAGES    = 8_000   # images processed per generation chunk
PROGRESS_EVERY  = 500     # log a line every N images inside a chunk
TOP_K_NEG       = 50      # semi-hard negative candidate pool
QUERIES_PER_IMG = 2       # max query strings extracted per image
QUEUE_DEPTH     = 2       # how many chunks the producer can buffer ahead
# ──────────────────────────────────────────────────────────────

_BASELINES = {
    "visual_only": [1.00, 0.00, 0.00, 0.00],
    "text_only":   [0.00, 1.00, 0.00, 0.00],
    "uniform":     [0.25, 0.25, 0.25, 0.25],
    "v5_fixed":    [0.55, 0.30, 0.10, 0.05],
}


# ──────────────────────────────────────────────────────────────
#  PURE HELPERS
# ──────────────────────────────────────────────────────────────

def _extract_queries(meta: dict) -> list:
    """Extract 1-2 natural-language query strings from a metadata row."""
    qs = []
    ocr   = meta.get("ocr_text") or ""
    words = ocr.split()
    if len(words) >= 2:
        qs.append(" ".join(words[:2]))
    parts = (meta.get("filename") or "").split("_")
    if parts and parts[0]:
        qs.append(parts[0])
    return qs[:QUERIES_PER_IMG] or ["image"]


def _compute_4signals(query_emb_np: np.ndarray, query_str: str,
                      meta: dict, embeddings: np.ndarray) -> np.ndarray:
    """
    Compute [visual, text, filename, structural] for one (query, candidate) pair.
    Mirrors V7 compute_signals() exactly so scores are research-comparable.
    """
    img_emb    = embeddings[meta["embedding_index"]]
    visual     = float(np.dot(query_emb_np, img_emb))

    q_tok = set(query_str.lower().split()) if query_str else set()
    o_tok = set((meta.get("ocr_text") or "").lower().split())
    f_tok = set((meta.get("filename") or "").lower().replace("_", " ").split())

    text     = len(q_tok & o_tok) / len(q_tok) if q_tok else 0.0
    filename = len(q_tok & f_tok) / len(q_tok) if q_tok else 0.0

    structural = 0.0
    if meta.get("is_screenshot"):                               structural -= 0.15
    if meta.get("pixel_area") and meta["pixel_area"] < 10_000: structural -= 0.10

    return np.array([visual, text, filename, structural], dtype="float32")


def _generate_chunk(metadata:     list,
                    embeddings:   np.ndarray,
                    emb_to_meta:  dict,
                    start:        int,
                    end:          int,
                    log_cb,
                    chunk_num:    int,
                    total_chunks: int) -> list:
    """
    Generate training examples for metadata[start:end].

    Logs one progress line every PROGRESS_EVERY images showing:
      - images processed so far / total in chunk
      - examples created so far
      - throughput (img/s) and ETA

    Returns list of dicts: {query_emb, pos_signals, neg_signals}
    """
    chunk_n  = end - start
    examples = []
    t0       = time.time()

    log_cb(f"│  ┌─ Chunk {chunk_num}/{total_chunks} "
           f"│ images [{start:,}–{end:,}] ({chunk_n:,} images)")

    for local_i in range(chunk_n):
        meta = metadata[start + local_i]

        if meta.get("embedding_index") is None:
            continue

        query_emb = embeddings[meta["embedding_index"]]

        # Semi-hard negative mining (vectorised dot product)
        sims = embeddings @ query_emb                            # (N,)
        sims[meta["embedding_index"]] = -999.0                  # exclude self
        top_k_idx = np.argpartition(-sims, TOP_K_NEG)[:TOP_K_NEG]

        for q_str in _extract_queries(meta):
            neg_emb_idx  = int(random.choice(top_k_idx))
            neg_meta_idx = emb_to_meta.get(neg_emb_idx)
            if neg_meta_idx is None:
                continue

            examples.append({
                "query_emb":   query_emb,
                "pos_signals": _compute_4signals(
                                   query_emb, q_str, meta, embeddings),
                "neg_signals": _compute_4signals(
                                   query_emb, q_str, metadata[neg_meta_idx], embeddings),
            })

        # Progress log every PROGRESS_EVERY images (and on final image)
        done = local_i + 1
        if done % PROGRESS_EVERY == 0 or done == chunk_n:
            elapsed = time.time() - t0
            rate    = done / max(elapsed, 0.01)
            eta     = (chunk_n - done) / max(rate, 0.01)
            width   = len(str(chunk_n))
            log_cb(f"│  │  {done:{width},}/{chunk_n:,} images "
                   f"→ {len(examples):,} examples "
                   f"│ {rate:,.0f} img/s "
                   f"│ ETA {eta:.0f}s")

    elapsed = time.time() - t0
    log_cb(f"│  └─ Chunk {chunk_num}/{total_chunks} ✓ "
           f"{len(examples):,} examples in {elapsed:.1f}s")
    return examples


def _verify_artifact(path: str, log_cb, label: str = "artifact") -> bool:
    """Check file exists on disk; log size or a warning."""
    p = Path(path)
    if p.exists():
        log_cb(f"│  ✓ {label} → {p.name}  ({p.stat().st_size / 1024:.1f} KB)  [{p}]")
        return True
    log_cb(f"│  ✗ WARNING: {label} NOT found at {path} — save may have failed!")
    return False


# ──────────────────────────────────────────────────────────────
#  DATASET WRAPPER
# ──────────────────────────────────────────────────────────────

class _ListDataset(Dataset):
    """Wraps a frozen Python list of example dicts for DataLoader use."""

    def __init__(self, examples: list):
        self._ex = examples

    def __len__(self):
        return len(self._ex)

    def __getitem__(self, idx):
        ex = self._ex[idx]
        return (
            torch.tensor(ex["query_emb"],   dtype=torch.float32),
            torch.tensor(ex["pos_signals"], dtype=torch.float32),  # (4,)
            torch.tensor(ex["neg_signals"], dtype=torch.float32),  # (4,)
        )


# ──────────────────────────────────────────────────────────────
#  TRAINER
# ──────────────────────────────────────────────────────────────

class BaselineTrainer(BaseTrainer):
    MODE = "baseline"

    # ----------------------------------------------------------
    # Fixed-baseline evaluation  (vectorised — one device transfer)
    # ----------------------------------------------------------
    def _eval_fixed_baselines(self, examples: list, log_cb) -> dict:
        subset = examples[:300]
        if not subset:
            log_cb("│  ⚠ No examples yet — skipping fixed baseline eval")
            return {k: 0.0 for k in _BASELINES}

        pos_t = torch.tensor(
            np.stack([e["pos_signals"] for e in subset]),
            dtype=torch.float32).to(self.device)               # (N, 4)
        neg_t = torch.tensor(
            np.stack([e["neg_signals"] for e in subset]),
            dtype=torch.float32).to(self.device)               # (N, 4)

        results = {}
        for name, weights in _BASELINES.items():
            w   = torch.tensor(weights, dtype=torch.float32).to(self.device)
            acc = ((pos_t @ w) > (neg_t @ w)).float().mean().item()
            results[name] = acc
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            log_cb(f"│  {name:15s}  [{bar}]  {acc:.3f}  {weights}")

        return results

    # ----------------------------------------------------------
    # Training n_epochs on a frozen example snapshot
    # ----------------------------------------------------------
    def _train_epochs(self, examples: list, n_epochs: int, batch_size: int,
                      global_ep_start: int, total_epochs: int,
                      log_cb, epoch_cb) -> tuple:
        """
        Train model for n_epochs on `examples`.
        Returns (best_loss_seen, updated_global_epoch).
        """
        loader    = DataLoader(_ListDataset(examples), batch_size=batch_size,
                               shuffle=True, drop_last=False, num_workers=0)
        n_batches = len(loader)
        log_freq  = max(1, n_batches // 5)
        best_loss = float("inf")

        for local_ep in range(1, n_epochs + 1):
            global_ep  = global_ep_start + local_ep
            self.model.train()
            epoch_loss = 0.0
            t0         = time.time()

            for bi, (q_emb, pos_sig, neg_sig) in enumerate(loader):
                q_emb   = q_emb.to(self.device)    # (B, 512)
                pos_sig = pos_sig.to(self.device)   # (B, 4)
                neg_sig = neg_sig.to(self.device)   # (B, 4)

                w = self.model(q_emb)               # (B, 4)  — weights from query

                # All 4 signals fused with their learned weights
                pos_score = (w * pos_sig).sum(dim=1).mean()
                neg_score = (w * neg_sig).sum(dim=1).mean()
                loss      = self.loss_fn(pos_score, neg_score)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

                if (bi + 1) % log_freq == 0:
                    running = epoch_loss / (bi + 1)
                    pct     = int(100 * (bi + 1) / n_batches)
                    log_cb(f"│    [Ep {global_ep:02d}/{total_epochs}] "
                           f"{pct:3d}%  batch {bi+1}/{n_batches}  "
                           f"loss={running:.4f}")

            self.scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed  = time.time() - t0
            eta_min  = elapsed * (total_epochs - global_ep) / 60
            log_cb(f"│  ── Ep {global_ep:02d}/{total_epochs}  "
                   f"loss={avg_loss:.4f}  ({elapsed:.0f}s  ETA≈{eta_min:.0f}min)")

            if avg_loss < best_loss:
                best_loss = avg_loss

            epoch_cb(global_ep, total_epochs, avg_loss, None)

        return best_loss, global_ep_start + n_epochs

    # ----------------------------------------------------------
    # Val accuracy helper (vectorised)
    # ----------------------------------------------------------
    def _val_acc(self, val_examples: list, limit: int = 200) -> float:
        if not val_examples:
            return 0.0
        sub   = val_examples[:limit]
        pos_t = torch.tensor(np.stack([e["pos_signals"] for e in sub]),
                             dtype=torch.float32).to(self.device)
        neg_t = torch.tensor(np.stack([e["neg_signals"] for e in sub]),
                             dtype=torch.float32).to(self.device)
        q_t   = torch.tensor(np.stack([e["query_emb"] for e in sub]),
                             dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            w = self.model(q_t)
            return ((w * pos_t).sum(1) > (w * neg_t).sum(1)).float().mean().item()

    # ----------------------------------------------------------
    # Main entry point
    # ----------------------------------------------------------
    def run(self, data_dir, output_path, epochs, lr, batch_size, log_cb, epoch_cb):
        t_total = time.time()
        self._setup(data_dir, epochs, lr, batch_size, log_cb)

        log_cb("╔══════════════════════════════════════════════════════╗")
        log_cb("║  BASELINE MODE — Pipeline Edition                    ║")
        log_cb("╚══════════════════════════════════════════════════════╝")
        log_cb(f"  Epochs: {epochs}  LR: {lr}  Batch: {batch_size}  Device: {self.device}")
        log_cb(f"  Images: {len(self.metadata):,}  Embeddings: {len(self.embeddings):,}")
        log_cb(f"  Chunk: {CHUNK_IMAGES:,} images  "
               f"Progress every: {PROGRESS_EVERY} images  "
               f"Top-K neg: {TOP_K_NEG}")
        log_cb(f"  Output: {output_path}")
        log_cb("")

        # Build embedding_index → row-index map once
        emb_to_meta = {
            m["embedding_index"]: i
            for i, m in enumerate(self.metadata)
            if m.get("embedding_index") is not None
        }

        N             = len(self.metadata)
        n_chunks      = math.ceil(N / CHUNK_IMAGES)
        ep_per_chunk  = max(1, epochs // n_chunks)
        leftover_ep   = max(0, epochs - ep_per_chunk * n_chunks)

        log_cb(f"  Chunks: {n_chunks}  │  "
               f"~{ep_per_chunk} epoch(s)/chunk  "
               f"(+{leftover_ep} leftover after all chunks)")
        log_cb("")

        # ══════════════════════════════════════════════════════
        # PRODUCER THREAD — generates chunks into the queue
        # ══════════════════════════════════════════════════════
        chunk_queue = _queue.Queue(maxsize=QUEUE_DEPTH)

        def _producer():
            for cn in range(1, n_chunks + 1):
                s   = (cn - 1) * CHUNK_IMAGES
                e   = min(s + CHUNK_IMAGES, N)
                exs = _generate_chunk(
                    self.metadata, self.embeddings, emb_to_meta,
                    s, e, log_cb, cn, n_chunks
                )
                chunk_queue.put((cn, exs))
            chunk_queue.put(None)   # sentinel: producer finished

        prod_thread = threading.Thread(target=_producer,
                                       daemon=True,
                                       name="ExampleProducer")
        prod_thread.start()

        log_cb("┌─── Pipeline started ─────────────────────────────────┐")
        log_cb("│  Producer thread: generating examples chunk by chunk")
        log_cb("│  Main thread:     trains on each chunk as it arrives")
        log_cb("│")

        # ══════════════════════════════════════════════════════
        # CONSUMER — trains on each chunk as it arrives
        # ══════════════════════════════════════════════════════
        all_train: list = []
        all_val:   list = []
        global_epoch     = 0
        overall_best     = float("inf")
        baseline_results = {}

        while True:
            # Block here until the next chunk (or sentinel) arrives
            log_cb(f"│  ⏳ Waiting for chunk from producer …")
            item = chunk_queue.get()

            # ── Sentinel: all chunks done ───────────────────────
            if item is None:
                if leftover_ep > 0 and all_train:
                    log_cb("│")
                    log_cb(f"│  All chunks done — {leftover_ep} leftover epoch(s) "
                           f"on full dataset ({len(all_train):,} examples)")
                    best, global_epoch = self._train_epochs(
                        list(all_train), leftover_ep, batch_size,
                        global_epoch, epochs, log_cb, epoch_cb
                    )
                    overall_best = min(overall_best, best)
                break

            chunk_num, new_examples = item

            # ── Split 85 / 15 train / val ───────────────────────
            random.shuffle(new_examples)
            cut = max(1, int(len(new_examples) * 0.15))
            new_val   = new_examples[:cut]
            new_train = new_examples[cut:]
            all_val.extend(new_val)
            all_train.extend(new_train)

            log_cb("│")
            log_cb(f"│  ▶ Chunk {chunk_num}/{n_chunks} received")
            log_cb(f"│    New:   +{len(new_train):,} train  +{len(new_val):,} val")
            log_cb(f"│    Total: {len(all_train):,} train  {len(all_val):,} val")

            # ── Fixed baseline eval (once, after chunk 1) ───────
            if chunk_num == 1:
                log_cb("│")
                log_cb("│  ── Fixed Baseline Evaluation ─────────────────────")
                baseline_results = self._eval_fixed_baselines(all_train, log_cb)
                log_cb("│")

            # ── Train on current accumulated data ───────────────
            # (producer is already generating chunk N+1 in the background)
            ep_now = ep_per_chunk
            log_cb(f"│  ── Training {ep_now} epoch(s) on {len(all_train):,} examples")
            log_cb(f"│     (producer building chunk {chunk_num + 1} in background …)")

            best, global_epoch = self._train_epochs(
                list(all_train),        # frozen snapshot passed to DataLoader
                ep_now, batch_size,
                global_epoch, epochs,
                log_cb, epoch_cb
            )
            overall_best = min(overall_best, best)

            # ── Save checkpoint + verify ─────────────────────────
            best_path = str(output_path).replace(".pt", "_best.pt")
            self.save_model(self.model, best_path,
                            {"mode": "baseline", "chunk": chunk_num,
                             "epoch": global_epoch, "loss": overall_best})
            _verify_artifact(best_path, log_cb, "best checkpoint")

            # ── Val accuracy ─────────────────────────────────────
            acc = self._val_acc(all_val)
            log_cb(f"│    Val accuracy after chunk {chunk_num}: {acc:.3f}")

        # ── Save final model ────────────────────────────────────
        log_cb("│")
        self.save_model(self.model, output_path,
                        {"mode": "baseline", "epochs": epochs,
                         "best_loss": overall_best})
        _verify_artifact(str(output_path), log_cb, "final model")
        log_cb("└" + "─" * 54 + "┘")

        # ══════════════════════════════════════════════════════
        # RESEARCH TABLE 1
        # ══════════════════════════════════════════════════════
        trained_acc      = self._val_acc(all_val, limit=300)
        best_fixed_acc   = max(baseline_results.values()) if baseline_results else 0.0
        gain             = trained_acc - best_fixed_acc

        log_cb("")
        log_cb("╔══════════════════════════════════════════════════════╗")
        log_cb("║  Research Table 1 — Baseline Comparison              ║")
        log_cb("╠══════════════════════════════════════════════════════╣")
        for name, acc in baseline_results.items():
            tag = "  ← best fixed" if acc == best_fixed_acc else ""
            log_cb(f"║  {name:15s}  {acc:.3f}{tag}")
        log_cb(f"║  {'adaptive':15s}  {trained_acc:.3f}  ← trained model")
        log_cb("╠══════════════════════════════════════════════════════╣")
        log_cb(f"║  Gain over best fixed:  {gain:+.3f}")
        log_cb(f"║  Best training loss:    {overall_best:.4f}")
        log_cb(f"║  Total train examples:  {len(all_train):,}")
        log_cb(f"║  Total time:            {(time.time() - t_total) / 60:.1f} min")
        log_cb("╚══════════════════════════════════════════════════════╝")

        return {
            "mode":               "baseline",
            "best_loss":          overall_best,
            "best_val_acc":       trained_acc,
            "baselines":          baseline_results,
            "gain_vs_best_fixed": gain,
            "total_examples":     len(all_train),
        }