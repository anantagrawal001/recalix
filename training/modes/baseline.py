"""
RECALIX V10 — Baseline Comparison Mode
========================================
PURPOSE: Generate Table 1 for your conference paper.

Evaluates 4 fixed-weight baselines AND trains the adaptive fusion model,
then prints a comparison showing fusion model gain.

Table 1 output:
  Method               | Pairwise Acc | Notes
  ---------------------|-------------|-------
  Visual only          | 0.XXX       | CLIP similarity only
  OCR text only        | 0.XXX       | text matching only
  Uniform weights      | 0.XXX       | 25/25/25/25
  V5 fixed weights     | 0.XXX       | 55/30/10/5 (original RECALIX)
  Fusion model (ours)  | 0.XXX       | adaptive per query ← should be highest

Run: python train.py --mode baseline --epochs 15
"""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from modes.base import (
    BaseTrainer,
    RECALIXDataset,
    dataset_summary,
    evaluate,
    evaluate_fixed_baselines,
    evaluate_fixed_weights,
    process_log,
    save_research_artifacts,
)


class BaselineTrainer(BaseTrainer):
    MODE = "baseline"

    def run(self, data_dir, output_path, epochs, lr, batch_size, log_cb, epoch_cb):
        self._setup(data_dir, epochs, lr, batch_size, log_cb)
        log_cb("Mode: BASELINE — evaluating all methods for paper Table 1")

        log_cb("Building evaluation dataset…")
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
        log_cb(f"Train: {len(base_train)}  Val: {len(base_val)}")

        if len(base_train) == 0:
            log_cb("ERROR: No training examples."); return {}

        val_ex = base_val.examples

        # ── Step 1: Evaluate all fixed-weight baselines ──────────────────────
        log_cb("")
        log_cb("=" * 50)
        log_cb("STEP 1: Fixed-weight baselines (no training needed)")
        log_cb("=" * 50)

        baselines = {
            "Visual only    [1,0,0,0]":        [1.0, 0.0, 0.0, 0.0],
            "OCR text only  [0,1,0,0]":        [0.0, 1.0, 0.0, 0.0],
            "Uniform        [.25,.25,.25,.25]": [0.25, 0.25, 0.25, 0.25],
            "V5 fixed       [.55,.30,.10,.05]": [0.55, 0.30, 0.10, 0.05],
        }

        baseline_results = evaluate_fixed_baselines(val_ex)
        for name, weights in baselines.items():
            acc = evaluate_fixed_weights(weights, val_ex)
            log_cb(f"  {name}  acc={acc:.4f}")
        for name, acc in baseline_results.items():
            if name not in {k.split("[", 1)[0].strip() for k in baselines}:
                log_cb(f"  {name:<29} acc={acc:.4f}")

        best_baseline_acc = max(baseline_results.values())
        log_cb(f"  Best fixed baseline: {best_baseline_acc:.4f}")

        # ── Step 2: Train fusion model ────────────────────────────────────────
        log_cb("")
        log_cb("=" * 50)
        log_cb("STEP 2: Training adaptive fusion model")
        log_cb("=" * 50)

        loader    = DataLoader(base_train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
        best_loss = float("inf")
        best_acc  = 0.0
        best_path = str(output_path).replace(".pt", "_best.pt")
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

            acc = evaluate(self.model, val_ex, self.device)
            history.append({"epoch": epoch, "loss": avg_loss, "val_acc": acc})
            log_cb(f"  Epoch {epoch}/{epochs}: loss={avg_loss:.4f}  val_acc={acc:.3f}")
            epoch_cb(epoch, epochs, avg_loss, acc)

            if acc > best_acc:
                best_acc = acc
                self.save_model(self.model, best_path, {"mode": "baseline", "epoch": epoch, "acc": best_acc})
            if avg_loss < best_loss:
                best_loss = avg_loss

        self.save_model(self.model, output_path, {"mode": "baseline", "epochs": epochs, "best_acc": best_acc})

        # ── Step 3: Print Table 1 ─────────────────────────────────────────────
        gain = best_acc - best_baseline_acc
        log_cb("")
        log_cb("=" * 60)
        log_cb("TABLE 1 — RESULTS FOR PAPER")
        log_cb("=" * 60)
        log_cb(f"  {'Method':<40} {'Pairwise Acc':>12}")
        log_cb(f"  {'-'*40} {'-'*12}")
        for name, acc in baseline_results.items():
            log_cb(f"  {name:<40} {acc:>12.4f}")
        log_cb(f"  {'-'*40} {'-'*12}")
        log_cb(f"  {'Fusion model (QCF) — OURS':<40} {best_acc:>12.4f}  ←")
        log_cb(f"  {'-'*40} {'-'*12}")
        log_cb(f"  Gain vs best fixed baseline: +{gain:.4f} ({gain*100:.2f}%)")
        log_cb("=" * 60)
        log_cb("Copy these numbers into your paper Table 1.")

        result = {
            "mode":             "baseline",
            "best_fusion_acc":  best_acc,
            "best_fixed_acc":   best_baseline_acc,
            "gain":             gain,
            "epochs":           epochs,
        }
        artifact_paths = save_research_artifacts(
            mode="baseline",
            output_path=str(output_path),
            metrics=history,
            summary=dataset_summary(base_train.examples, base_val.examples),
            baseline_results=baseline_results,
            result=result,
            process=process_log(base_train, base_val, "baseline"),
        )
        log_cb(f"Research artifacts saved → {artifact_paths['artifact_dir']}")
        result["artifact_dir"] = artifact_paths["artifact_dir"]
        return result
