"""
RECALIX Training — Semi-Hard Negative Mode
============================================
Mixed negative mining: 50% random + 50% semi-hard.
Semi-hard = negatives that are close to anchor but further than positive.
This is the RECOMMENDED default — best quality/speed tradeoff.

Research reference: FaceNet (Schroff et al. 2015) negative mining strategy.
"""

import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from modes.base import BaseTrainer, RECALIXDataset, evaluate


class SemiHardDataset(Dataset):
    """Augmented dataset with semi-hard negative sampling."""

    def __init__(self, base_dataset: RECALIXDataset, embeddings: np.ndarray, device: str):
        self.base      = base_dataset
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.device    = device

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        q_emb, pos_emb, neg_emb, q_type = self.base[i]

        # 50% semi-hard mining
        if random.random() < 0.5:
            return q_emb, pos_emb, neg_emb, q_type

        # Semi-hard: find negatives closer to anchor than positive but not too close
        with torch.no_grad():
            q = q_emb
            sims   = self.embeddings @ q           # (N,)
            pos_sim = (self.embeddings * pos_emb).sum(dim=1).max().item()

            # Semi-hard candidates: sim in (pos_sim - margin, pos_sim)
            margin = 0.15
            mask   = (sims < pos_sim) & (sims > pos_sim - margin)
            cands  = mask.nonzero(as_tuple=True)[0]

            if len(cands) > 0:
                idx    = cands[random.randint(0, len(cands) - 1)].item()
                neg_emb = self.embeddings[idx]

        return q_emb, pos_emb, neg_emb, q_type


class SemiHardTrainer(BaseTrainer):
    MODE = "semi_hard"

    def run(self, data_dir, output_path, epochs, lr, batch_size, log_cb, epoch_cb):
        self._setup(data_dir, epochs, lr, batch_size, log_cb)
        log_cb("Mode: SEMI-HARD (50% random + 50% semi-hard negatives) [RECOMMENDED]")

        base_train = RECALIXDataset(self.embeddings, self.metadata, "train")
        base_val   = RECALIXDataset(self.embeddings, self.metadata, "val")
        train_ds   = SemiHardDataset(base_train, self.embeddings, self.device)
        loader     = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        log_cb(f"Train: {len(train_ds)}, Val: {len(base_val)}")
        if len(train_ds) == 0:
            log_cb("ERROR: No training examples."); return {}

        best_loss = float("inf")
        best_path = str(output_path).replace(".pt", "_best.pt")

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for q_emb, pos_emb, neg_emb, _ in loader:
                q_emb   = q_emb.to(self.device)
                pos_emb = pos_emb.to(self.device)
                neg_emb = neg_emb.to(self.device)

                w = self.model(q_emb)  # (B,4)

                # All 4 weighted signals
                def hybrid_score(ref, emb):
                    return (w[:, 0] * (ref * emb).sum(dim=1)).mean()

                pos_score = hybrid_score(q_emb, pos_emb)
                neg_score = hybrid_score(q_emb, neg_emb)
                loss = self.loss_fn(pos_score, neg_score)

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

            # Eval every 5 epochs
            recall = None
            if epoch % 5 == 0 or epoch == epochs:
                res = evaluate(self.model, self.embeddings, self.metadata, self.device)
                if "visual" in res:
                    recall = res["visual"].get("recall@5")
                for bkt, metrics in res.items():
                    log_cb(f"  [{bkt}] R@1={metrics['recall@1']:.3f} R@5={metrics['recall@5']:.3f} MRR={metrics['mrr']:.3f}")

            epoch_cb(epoch, epochs, avg_loss, recall)

        self.save_model(self.model, output_path, {"mode": "semi_hard", "epochs": epochs, "best_loss": best_loss})
        log_cb(f"✓ Done. Best loss: {best_loss:.5f}")
        return {"mode": "semi_hard", "best_loss": best_loss, "epochs": epochs}
