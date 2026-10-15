"""
RECALIX V10 — Semi-Hard Negative Training (RECOMMENDED)
========================================================
50% random negatives + 50% semi-hard negatives.

Semi-hard negative: an image that is visually similar to the query
but NOT the correct match (closer than positive by at most margin=0.30).
This gives harder gradients than random but avoids training instability
from fully hard negatives.

All CLIP calls happen in _build() (main thread, before DataLoader).
__getitem__ is pure numpy — no GPU calls inside DataLoader workers.
"""

import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from modes.base import (
    BaseTrainer,
    RECALIXDataset,
    compute_4signals,
    dataset_summary,
    evaluate,
    evaluate_fixed_baselines,
    process_log,
    save_research_artifacts,
)


class SemiHardDataset(Dataset):
    def __init__(self, base: RECALIXDataset, embeddings: np.ndarray, metadata: list, margin: float = 0.30):
        self.base        = base
        self.embeddings  = embeddings   # CPU numpy — safe in DataLoader
        self.metadata    = metadata
        self.emb_to_meta = base._emb_to_meta
        self.candidate_embedding_indices = base.candidate_embedding_indices
        self.margin      = margin

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        q_emb, pos_sig, neg_sig, q_str = self.base[i]
        # The base dataset already chooses negatives from the top-K visually
        # similar candidates. Reusing those precomputed negatives keeps large
        # research runs tractable and avoids a full embedding scan per batch item.
        return q_emb, pos_sig, neg_sig


class SemiHardTrainer(BaseTrainer):
    MODE = "semi_hard"

    def run(self, data_dir, output_path, epochs, lr, batch_size, log_cb, epoch_cb):
        self._setup(data_dir, epochs, lr, batch_size, log_cb)
        log_cb("Mode: SEMI-HARD (50% random + 50% semi-hard negatives)  [RECOMMENDED]")

        log_cb("Building datasets — computing CLIP text embeddings for all images…")
        t0 = time.time()
        base_train = RECALIXDataset(
            self.embeddings, self.metadata, self.embed_fn,
            split="train", top_k_neg=50, log_cb=log_cb,
            data_dir=data_dir, negative_mode="mixed",
        )
        base_val = RECALIXDataset(
            self.embeddings, self.metadata, self.embed_fn,
            split="val", top_k_neg=50, log_cb=log_cb,
            data_dir=data_dir, negative_mode="mixed",
        )
        log_cb(f"Dataset build: {time.time()-t0:.0f}s")

        # Pre-training diagnostic — detect too-easy negatives early
        if base_train.examples:
            pos_sims = [e["pos_sig"][0] for e in base_train.examples[:100]]
            neg_sims = [e["neg_sig"][0] for e in base_train.examples[:100]]
            avg_gap  = float(np.mean([p - n for p, n in zip(pos_sims, neg_sims)]))
            log_cb(f"Pre-training: avg pos-neg visual gap = {avg_gap:.4f} (margin = 0.40)")
            if avg_gap > 0.40:
                log_cb("⚠ Gap already larger than margin — negatives may be too easy.")
                log_cb("  Training will still run. For better results, index diverse photos.")
            else:
                log_cb("✓ Training difficulty is reasonable.")

        train_ds = SemiHardDataset(base_train, self.embeddings, self.metadata)
        loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

        log_cb(f"Train: {len(train_ds)}  Val: {len(base_val)}")
        if len(train_ds) == 0:
            log_cb("ERROR: No training examples."); return {}

        baseline_results = evaluate_fixed_baselines(base_val.examples)
        best_fixed_acc = max(baseline_results.values()) if baseline_results else 0.0
        log_cb("Fixed baselines: " + "  ".join(
            f"{name}={acc:.3f}" for name, acc in baseline_results.items()
        ))

        best_loss = float("inf")
        best_acc = 0.0
        best_path = str(output_path).replace(".pt", "_best.pt")
        history = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for q_emb, pos_sig, neg_sig in loader:
                q_emb   = q_emb.to(self.device)
                pos_sig = pos_sig.to(self.device)
                neg_sig = neg_sig.to(self.device)

                w         = self.model(q_emb)              # (B, 4)
                pos_score = (w * pos_sig).sum(dim=1)       # (B,)
                neg_score = (w * neg_sig).sum(dim=1)       # (B,)
                loss      = self.loss_fn(pos_score, neg_score)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            self.scheduler.step()
            avg_loss = epoch_loss / max(len(loader), 1)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model(self.model, best_path, {"mode": "semi_hard", "epoch": epoch, "loss": best_loss})

            acc = evaluate(self.model, base_val.examples, self.device)
            best_acc = max(best_acc, acc)
            history.append({"epoch": epoch, "loss": avg_loss, "val_acc": acc})
            log_cb(f"  Epoch {epoch}/{epochs}: loss={avg_loss:.4f}  val_acc={acc:.3f}")

            # Detailed weight stats every 5 epochs
            if epoch % 5 == 0 or epoch == epochs:
                with torch.no_grad():
                    sample = base_val.examples[:50]
                    if sample:
                        sq = torch.tensor(
                            np.stack([e["query_emb"] for e in sample]), dtype=torch.float32
                        ).to(self.device)
                        ww = self.model(sq).cpu().numpy()
                        wm, ws = ww.mean(0), ww.std(0)
                        keys = ["visual", "text", "filename", "structural"]
                        log_cb("  Weights: " + "  ".join(
                            f"{k}={wm[j]:.3f}±{ws[j]:.3f}" for j, k in enumerate(keys)
                        ))
                        if ws.max() < 0.01:
                            log_cb("  ⚠ Weights not varying — model not differentiating signals")
                            log_cb("  Tip: index more diverse photos (not just screenshots)")

            epoch_cb(epoch, epochs, avg_loss, acc)

        self.save_model(self.model, output_path, {"mode": "semi_hard", "epochs": epochs, "best_loss": best_loss})
        log_cb(f"✓ Done. Best loss: {best_loss:.5f}")
        result = {
            "mode": "semi_hard",
            "best_loss": best_loss,
            "best_val_acc": best_acc,
            "best_fixed_acc": best_fixed_acc,
            "gain": best_acc - best_fixed_acc,
            "epochs": epochs,
        }
        artifact_paths = save_research_artifacts(
            mode="semi_hard",
            output_path=str(output_path),
            metrics=history,
            summary=dataset_summary(base_train.examples, base_val.examples),
            baseline_results=baseline_results,
            result=result,
            process=process_log(base_train, base_val, "semi_hard"),
        )
        log_cb(f"Research artifacts saved → {artifact_paths['artifact_dir']}")
        result["artifact_dir"] = artifact_paths["artifact_dir"]
        return result
