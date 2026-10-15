"""
RECALIX V10 — Hard Negative Training
======================================
Fully hard negatives: the most visually similar image that is NOT the correct match.
Highest potential quality. Risk of instability.
LR auto-halved on loss spike. Early stop after 3 consecutive spikes.
Use after semi_hard converges.
"""

import random
import time
import pickle
import hashlib
import json
import os
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


class HardNegDataset(Dataset):
    def __init__(self, base: RECALIXDataset, embeddings: np.ndarray, metadata: list, log_cb=None):
        self.base        = base
        self.embeddings  = embeddings
        self.metadata    = metadata
        self.emb_to_meta = base._emb_to_meta
        self.candidate_embedding_indices = base.candidate_embedding_indices
        self.log_cb = log_cb or (lambda s: None)
        self._mined = 0
        self._last_log = time.time()
        self.examples = self._build_hard_examples()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        e = self.examples[i]
        q_emb = torch.tensor(e["query_emb"], dtype=torch.float32)
        pos_sig = torch.tensor(e["pos_sig"], dtype=torch.float32)
        neg_sig = torch.tensor(e["neg_sig"], dtype=torch.float32)
        return q_emb, pos_sig, neg_sig

    def _cache_path(self):
        data_dir = getattr(self.base, "data_dir", None)
        if not data_dir or not getattr(self.base, "build_info", None):
            return None
        key_data = {
            "version": "hard-neg-v1",
            "base_cache": self.base.build_info.get("cache_path"),
            "base_examples": len(self.base.examples),
            "candidate_count": len(self.candidate_embedding_indices),
            "embedding_shape": tuple(self.embeddings.shape),
        }
        key = hashlib.sha1(json.dumps(key_data, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return data_dir / "training_example_cache" / f"hard_mined_{key}.pkl"

    def _build_hard_examples(self):
        if not self.base.examples:
            return []

        cache_path = self._cache_path()
        if cache_path and cache_path.exists() and not bool(int(os.environ.get("RECALIX_DISABLE_HARD_CACHE", "0") or 0)):
            try:
                payload = pickle.loads(cache_path.read_bytes())
                examples = payload.get("examples", [])
                if examples:
                    self.log_cb(f"  Hard mining: loaded {len(examples)} cached fully-hard examples")
                    return examples
            except Exception as e:
                self.log_cb(f"  Hard mining cache ignored: {e}")

        candidate_ids = np.asarray(self.candidate_embedding_indices, dtype=np.int64)
        if len(candidate_ids) < 2:
            return list(self.base.examples)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = int(os.environ.get("RECALIX_HARD_MINE_BATCH", "128") or 128)
        batch_size = max(8, batch_size)
        self.log_cb(
            f"  Hard mining: starting full hard-negative precompute for {len(self.base.examples)} examples "
            f"against {len(candidate_ids)} candidates on {device} (batch={batch_size})"
        )

        examples = []
        t0 = time.time()
        last_log = t0
        cand_tensor = None
        cand_ids_tensor = None
        try:
            cand_tensor = torch.tensor(self.embeddings[candidate_ids], dtype=torch.float32, device=device)
            cand_ids_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=device)
        except Exception as e:
            self.log_cb(f"  Hard mining: GPU candidate tensor failed ({e}); using CPU")
            device = "cpu"
            cand_tensor = torch.tensor(self.embeddings[candidate_ids], dtype=torch.float32)
            cand_ids_tensor = torch.tensor(candidate_ids, dtype=torch.long)

        i = 0
        while i < len(self.base.examples):
            chunk = self.base.examples[i:i + batch_size]
            try:
                q = torch.tensor(np.stack([e["query_emb"] for e in chunk]), dtype=torch.float32, device=device)
                scores = q @ cand_tensor.T
                pos_sims = torch.tensor([float(e["pos_sig"][0]) for e in chunk], dtype=torch.float32, device=device)
                pos_ids = torch.tensor([int(e["pos_emb_idx"]) for e in chunk], dtype=torch.long, device=device)
                scores = scores.masked_fill(scores >= pos_sims[:, None], -float("inf"))
                scores = scores.masked_fill(cand_ids_tensor[None, :] == pos_ids[:, None], -float("inf"))
                best_pos = torch.argmax(scores, dim=1).detach().cpu().numpy()
                best_scores = torch.max(scores, dim=1).values.detach().cpu().numpy()
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 8:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    batch_size = max(8, batch_size // 2)
                    self.log_cb(f"  Hard mining: CUDA OOM, reducing batch to {batch_size}")
                    continue
                raise

            for local_idx, base_example in enumerate(chunk):
                out = dict(base_example)
                if np.isfinite(best_scores[local_idx]):
                    neg_emb_idx = int(candidate_ids[int(best_pos[local_idx])])
                    neg_meta_idx = self.emb_to_meta.get(neg_emb_idx)
                    if neg_meta_idx is not None:
                        out["neg_emb_idx"] = neg_emb_idx
                        out["neg_sig"] = compute_4signals(
                            out["query_emb"],
                            out["query_str"],
                            self.metadata[neg_meta_idx],
                            self.embeddings,
                        )
                        out["negative_kind"] = "full_hard"
                examples.append(out)

            i += len(chunk)
            now = time.time()
            if i >= len(self.base.examples) or now - last_log >= 15:
                last_log = now
                self.log_cb(
                    f"  Hard mining: {i}/{len(self.base.examples)} examples "
                    f"({100 * i / max(len(self.base.examples), 1):.1f}%) elapsed={(now - t0) / 60:.1f}m"
                )

        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(pickle.dumps({"examples": examples}, protocol=pickle.HIGHEST_PROTOCOL))
                self.log_cb(f"  Hard mining: cached fully-hard examples -> {cache_path}")
            except Exception as e:
                self.log_cb(f"  Hard mining cache save skipped: {e}")
        return examples


class HardTrainer(BaseTrainer):
    MODE = "hard"

    def run(self, data_dir, output_path, epochs, lr, batch_size, log_cb, epoch_cb):
        self._setup(data_dir, epochs, lr, batch_size, log_cb)
        log_cb("Mode: HARD (fully hard negative mining)")
        log_cb("⚠ Hard negatives risk instability — LR auto-halved on loss spike")

        base_train = RECALIXDataset(
            self.embeddings, self.metadata, self.embed_fn,
            split="train", top_k_neg=50, log_cb=log_cb,
            data_dir=data_dir, negative_mode="topk",
        )
        base_val = RECALIXDataset(
            self.embeddings, self.metadata, self.embed_fn,
            split="val", top_k_neg=50, log_cb=log_cb,
            data_dir=data_dir, negative_mode="topk",
        )
        train_ds = HardNegDataset(base_train, self.embeddings, self.metadata, log_cb=log_cb)
        loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
        log_cb(f"Train: {len(train_ds)}  Val: {len(base_val)}")
        if len(train_ds) == 0:
            log_cb("ERROR: No training examples. Index more photos or clear stale training_example_cache.")
            return {}

        best_loss = float("inf")
        best_acc = 0.0
        prev_loss = float("inf")
        best_path = str(output_path).replace(".pt", "_best.pt")
        patience  = 0
        log_cb("Evaluating fixed baselines before training...")
        baseline_results = evaluate_fixed_baselines(base_val.examples)
        best_fixed_acc = max(baseline_results.values()) if baseline_results else 0.0
        log_cb("Fixed baselines: " + "  ".join(
            f"{name}={acc:.3f}" for name, acc in baseline_results.items()
        ))
        history = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_start = time.time()
            log_cb(
                f"Epoch {epoch}/{epochs} started: {len(loader)} batches, "
                "fully hard negatives mined during batch loading"
            )

            for batch_idx, (q_emb, pos_sig, neg_sig) in enumerate(loader, start=1):
                q_emb   = q_emb.to(self.device)
                pos_sig = pos_sig.to(self.device)
                neg_sig = neg_sig.to(self.device)

                w         = self.model(q_emb)
                pos_score = (w * pos_sig).sum(dim=1)
                neg_score = (w * neg_sig).sum(dim=1)
                loss      = self.loss_fn(pos_score, neg_score)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                epoch_loss += loss.item()
                if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == len(loader):
                    elapsed = time.time() - epoch_start
                    avg_so_far = epoch_loss / max(batch_idx, 1)
                    log_cb(
                        f"  Epoch {epoch}/{epochs}: batch {batch_idx}/{len(loader)} "
                        f"loss={avg_so_far:.4f} elapsed={elapsed/60:.1f}m"
                    )

            self.scheduler.step()
            avg_loss = epoch_loss / max(len(loader), 1)

            if avg_loss > prev_loss * 1.5 and epoch > 3:
                for g in self.optimizer.param_groups:
                    g["lr"] *= 0.5
                log_cb(f"⚠ Loss spike at epoch {epoch} — LR halved")
                patience += 1
                if patience >= 3:
                    log_cb("⚠ 3 consecutive spikes — stopping early"); break
            else:
                patience = 0
            prev_loss = avg_loss

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model(self.model, best_path, {"mode": "hard", "epoch": epoch, "loss": best_loss})

            acc = evaluate(self.model, base_val.examples, self.device)
            best_acc = max(best_acc, acc)
            history.append({"epoch": epoch, "loss": avg_loss, "val_acc": acc})
            log_cb(f"  Epoch {epoch}/{epochs}: loss={avg_loss:.4f}  val_acc={acc:.3f}")
            epoch_cb(epoch, epochs, avg_loss, acc)

        self.save_model(self.model, output_path, {"mode": "hard", "epochs": epoch, "best_loss": best_loss})
        log_cb(f"✓ Done. Best loss: {best_loss:.5f}")
        result = {
            "mode": "hard",
            "best_loss": best_loss,
            "best_val_acc": best_acc,
            "best_fixed_acc": best_fixed_acc,
            "gain": best_acc - best_fixed_acc,
            "epochs": epoch,
        }
        artifact_paths = save_research_artifacts(
            mode="hard",
            output_path=str(output_path),
            metrics=history,
            summary=dataset_summary(base_train.examples, base_val.examples),
            baseline_results=baseline_results,
            result=result,
            process=process_log(base_train, base_val, "hard"),
        )
        log_cb(f"Research artifacts saved → {artifact_paths['artifact_dir']}")
        result["artifact_dir"] = artifact_paths["artifact_dir"]
        return result
