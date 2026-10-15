"""
RECALIX V10 — Basic Training Mode
===================================
Random negatives. Fast. Good first run to verify the pipeline works.
Use this before semi_hard to confirm loss is actually moving.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from modes.base import (
    BaseTrainer,
    RECALIXDataset,
    dataset_summary,
    evaluate,
    evaluate_fixed_baselines,
    process_log,
    save_research_artifacts,
)


class BasicTrainer(BaseTrainer):
    MODE = "basic"

    def run(self, data_dir, output_path, epochs, lr, batch_size, log_cb, epoch_cb):
        self._setup(data_dir, epochs, lr, batch_size, log_cb)
        log_cb("Mode: BASIC (random negatives — fast first run)")

        log_cb("Building training dataset…")
        base_train = RECALIXDataset(
            self.embeddings, self.metadata, self.embed_fn,
            split="train", top_k_neg=50, log_cb=log_cb,
            data_dir=data_dir, negative_mode="random",
        )
        base_val = RECALIXDataset(
            self.embeddings, self.metadata, self.embed_fn,
            split="val", top_k_neg=50, log_cb=log_cb,
            data_dir=data_dir, negative_mode="random",
        )

        log_cb(f"Train: {len(base_train)}  Val: {len(base_val)}")
        if len(base_train) == 0:
            log_cb("ERROR: No training examples. Index more photos first."); return {}

        loader    = DataLoader(base_train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
        best_loss = float("inf")
        best_acc = 0.0
        best_path = str(output_path).replace(".pt", "_best.pt")
        baseline_results = evaluate_fixed_baselines(base_val.examples)
        best_fixed_acc = max(baseline_results.values()) if baseline_results else 0.0
        history = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for q_emb, pos_sig, neg_sig, _ in loader:
                q_emb   = q_emb.to(self.device)
                pos_sig = pos_sig.to(self.device)
                neg_sig = neg_sig.to(self.device)

                w         = self.model(q_emb)
                pos_score = (w * pos_sig).sum(dim=1)
                neg_score = (w * neg_sig).sum(dim=1)
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
                self.save_model(self.model, best_path, {"mode": "basic", "epoch": epoch, "loss": best_loss})

            acc = evaluate(self.model, base_val.examples, self.device)
            best_acc = max(best_acc, acc)
            history.append({"epoch": epoch, "loss": avg_loss, "val_acc": acc})
            log_cb(f"  Epoch {epoch}/{epochs}: loss={avg_loss:.4f}  val_acc={acc:.3f}")
            epoch_cb(epoch, epochs, avg_loss, acc)

        self.save_model(self.model, output_path, {"mode": "basic", "epochs": epochs, "best_loss": best_loss})
        log_cb(f"✓ Done. Best loss: {best_loss:.5f}")
        result = {
            "mode": "basic",
            "best_loss": best_loss,
            "best_val_acc": best_acc,
            "best_fixed_acc": best_fixed_acc,
            "gain": best_acc - best_fixed_acc,
            "epochs": epochs,
        }
        artifact_paths = save_research_artifacts(
            mode="basic",
            output_path=str(output_path),
            metrics=history,
            summary=dataset_summary(base_train.examples, base_val.examples),
            baseline_results=baseline_results,
            result=result,
            process=process_log(base_train, base_val, "basic"),
        )
        log_cb(f"Research artifacts saved → {artifact_paths['artifact_dir']}")
        result["artifact_dir"] = artifact_paths["artifact_dir"]
        return result
