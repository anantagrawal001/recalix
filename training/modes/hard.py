"""
RECALIX Training — Hard Negative Mode
========================================
Full hard negative mining: always picks the hardest negatives (closest to anchor).
Highest training difficulty → best potential quality, but risk of training collapse.

Use after basic/semi_hard converges. Reduces LR automatically if loss spikes.

Research insight: hard negatives maximize discrimination between query types.
"""

import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from modes.base import BaseTrainer, RECALIXDataset, evaluate


class HardNegDataset(Dataset):
    def __init__(self, base: RECALIXDataset, embeddings: np.ndarray):
        self.base       = base
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        q_emb, pos_emb, _, q_type = self.base[i]
        with torch.no_grad():
            sims = self.embeddings @ q_emb
            pos_sim = (self.embeddings * pos_emb).sum(dim=1).max().item()
            # Hard negative: highest sim but < positive sim
            mask  = sims < pos_sim
            if mask.any():
                best_neg_idx = (sims * mask.float()).argmax().item()
            else:
                best_neg_idx = random.randint(0, len(self.embeddings) - 1)
            neg_emb = self.embeddings[best_neg_idx]
        return q_emb, pos_emb, neg_emb, q_type


class HardTrainer(BaseTrainer):
    MODE = "hard"

    def run(self, data_dir, output_path, epochs, lr, batch_size, log_cb, epoch_cb):
        self._setup(data_dir, epochs, lr, batch_size, log_cb)
        log_cb("Mode: HARD (fully hard negative mining)")
        log_cb("⚠ Hard negatives risk training instability — LR auto-reduced on loss spike")

        base_train = RECALIXDataset(self.embeddings, self.metadata, "train")
        train_ds   = HardNegDataset(base_train, self.embeddings)
        loader     = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        log_cb(f"Train: {len(train_ds)}")

        best_loss  = float("inf")
        prev_loss  = float("inf")
        best_path  = str(output_path).replace(".pt", "_best.pt")
        patience   = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for q_emb, pos_emb, neg_emb, _ in loader:
                q_emb   = q_emb.to(self.device)
                pos_emb = pos_emb.to(self.device)
                neg_emb = neg_emb.to(self.device)

                w = self.model(q_emb)
                pos_score = (w[:, 0] * (pos_emb * q_emb).sum(dim=1)).mean()
                neg_score = (w[:, 0] * (neg_emb * q_emb).sum(dim=1)).mean()
                loss = self.loss_fn(pos_score, neg_score)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                epoch_loss += loss.item()

            self.scheduler.step()
            avg_loss = epoch_loss / max(len(loader), 1)

            # Instability detection
            if avg_loss > prev_loss * 1.5 and epoch > 3:
                for g in self.optimizer.param_groups:
                    g["lr"] *= 0.5
                log_cb(f"⚠ Loss spike at epoch {epoch} — LR halved")
                patience += 1
                if patience >= 3:
                    log_cb("⚠ 3 consecutive spikes — stopping early")
                    break
            else:
                patience = 0

            prev_loss = avg_loss

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model(self.model, best_path, {"mode": "hard", "epoch": epoch, "loss": best_loss})

            recall = None
            if epoch % 5 == 0 or epoch == epochs:
                res = evaluate(self.model, self.embeddings, self.metadata, self.device)
                for bkt, metrics in res.items():
                    log_cb(f"  [{bkt}] R@1={metrics['recall@1']:.3f} R@5={metrics['recall@5']:.3f}")
                if "visual" in res:
                    recall = res["visual"].get("recall@5")

            epoch_cb(epoch, epochs, avg_loss, recall)

        self.save_model(self.model, output_path, {"mode": "hard", "epochs": epoch, "best_loss": best_loss})
        log_cb(f"✓ Done. Best loss: {best_loss:.5f}")
        return {"mode": "hard", "best_loss": best_loss}
