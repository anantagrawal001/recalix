"""
RECALIX Training — Basic Mode
================================
Random negative sampling. Fast, good starting point.
Best for: quick first run, verifying the pipeline works.
"""

import random
import torch
from torch.utils.data import DataLoader
from modes.base import BaseTrainer, RECALIXDataset, evaluate


class BasicTrainer(BaseTrainer):
    MODE = "basic"

    def run(self, data_dir, output_path, epochs, lr, batch_size, log_cb, epoch_cb):
        self._setup(data_dir, epochs, lr, batch_size, log_cb)
        log_cb("Mode: BASIC (random negatives)")

        train_ds = RECALIXDataset(self.embeddings, self.metadata, "train")
        val_ds   = RECALIXDataset(self.embeddings, self.metadata, "val")
        log_cb(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

        if len(train_ds) == 0:
            log_cb("ERROR: No training examples — index more photos first.")
            return {}

        loader    = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        best_loss = float("inf")
        best_path = str(output_path).replace(".pt", "_best.pt")

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for batch in loader:
                q_emb, pos_emb, neg_emb, _ = batch
                q_emb   = q_emb.to(self.device)
                pos_emb = pos_emb.to(self.device)
                neg_emb = neg_emb.to(self.device)

                w = self.model(q_emb)  # (B, 4)

                # Weighted visual score (w[:,0] = visual weight)
                pos_score = (w[:, 0] * (pos_emb * q_emb).sum(dim=1)).mean()
                neg_score = (w[:, 0] * (neg_emb * q_emb).sum(dim=1)).mean()

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
                self.save_model(self.model, best_path, {"mode": "basic", "epoch": epoch, "loss": best_loss})

            epoch_cb(epoch, epochs, avg_loss, None)

        # Final eval
        self.save_model(self.model, output_path, {"mode": "basic", "epochs": epochs, "best_loss": best_loss})
        log_cb(f"Best loss: {best_loss:.5f}")
        return {"mode": "basic", "best_loss": best_loss, "epochs": epochs}
