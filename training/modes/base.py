"""
RECALIX V10 — Training Base
============================
Shared code used by all training modes.

What this does:
  1. extract_queries()   - generates text queries from photo metadata
  2. compute_4signals()  - computes [visual, ocr, filename, structural] for a (query, photo) pair
  3. RECALIXDataset      - builds triplets: (query_emb, pos_signals, neg_signals)
  4. evaluate()          - measures pairwise accuracy on validation set
  5. BaseTrainer         - loads CLIP + data + model for any training mode

Key design decision:
  Queries come from OCR text and filenames — NOT image embeddings.
  This matches inference: at search time, query is always a TEXT embedding.
  Training with image embeddings as queries = training/inference mismatch = loss=0.
"""

import json
import re
import csv
import hashlib
import os
import pickle
import random
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_signals(visual: float, text: float, filename: float, structural: float) -> np.ndarray:
    """Normalize signals to a comparable [0.0, 1.0] scale."""
    v_norm = float(np.clip((visual - 0.15) / 0.20, 0.0, 1.0))
    t_norm = float(np.clip(text, 0.0, 1.0))
    f_norm = float(np.clip(filename, 0.0, 1.0))
    s_norm = float(np.clip((structural + 0.2) / 0.5, 0.0, 1.0))
    return np.array([v_norm, t_norm, f_norm, s_norm], dtype=np.float32)


def compute_4signals(
    query_text_emb: np.ndarray,
    query_str:      str,
    meta:           Dict,
    embeddings:     np.ndarray,
) -> np.ndarray:
    """
    Compute [visual, text, filename, structural] for one (query, candidate) pair.

    visual:     cosine sim between query TEXT embedding and image embedding
    text:       keyword overlap between query and OCR text in the image
    filename:   keyword overlap between query and filename tokens
    structural: heuristic score based on image type (document/screenshot)

    Returns (4,) float32
    """
    emb_idx = meta["embedding_index"]
    img_emb = embeddings[emb_idx]

    # 1. Visual: CLIP text-image similarity (pre-normalized, so dot = cosine)
    visual = float(np.dot(query_text_emb, img_emb))

    # 2. OCR text overlap
    q_tok  = set(query_str.lower().split()) if query_str else set()
    o_tok  = set((meta.get("ocr_text") or "").lower().split())
    text   = len(q_tok & o_tok) / len(q_tok) if q_tok else 0.0

    # 3. Filename token overlap
    f_tok     = set((meta.get("filename") or "").lower().replace("_", " ").split())
    filename  = len(q_tok & f_tok) / len(q_tok) if q_tok else 0.0

    # 4. Structural heuristic (no +0.5 bias — only real discriminative signal)
    q_lower    = query_str.lower()
    structural = 0.0
    doc_kw     = {"document", "invoice", "receipt", "scan", "form", "certificate", "paper"}
    if any(kw in q_lower for kw in doc_kw) and meta.get("is_document"):
        structural += 0.3
    if meta.get("is_screenshot") and "screenshot" not in q_lower:
        structural -= 0.1
    pa = meta.get("pixel_area")
    if pa and pa < 10000:
        structural -= 0.15

    return normalize_signals(visual, text, filename, structural)


def extract_query_items(meta: Dict) -> List[Dict]:
    """
    Generate meaningful text queries from a photo's metadata, balanced across
    caption, OCR text, and filename sources.
    """
    queries = []
    fname   = (meta.get("filename") or "")
    ocr     = (meta.get("ocr_text") or "").strip()
    captions = meta.get("captions") or []

    # 1. Caption query: extract at most 1 caption query
    if captions:
        caption = captions[0]
        if isinstance(caption, dict):
            text = " ".join(str(caption.get("text") or "").split())
            emb = caption.get("embedding")
        else:
            text = " ".join(str(caption).split())
            emb = None
        if len(text) > 8:
            queries.append({"text": text, "embedding": emb, "source": "caption"})

    # Words that appear in almost every screenshot filename — useless as queries
    SKIP = {
        "screenshot", "img", "image", "photo", "dsc", "pic",
        "whatsapp", "telegram", "download", "file", "scan",
        "capture", "screen", "snap", "copy", "saved", "received",
        "signal", "instagram", "facebook", "twitter", "tiktok",
    }

    # 2. OCR query: extract at most 1 OCR query
    ocr_words = ocr.split()
    if len(ocr_words) >= 3:
        mid    = max(0, len(ocr_words) // 3)
        phrase = " ".join(ocr_words[mid: mid + 4])
        if len(phrase) > 8:
            queries.append({"text": phrase, "embedding": None, "source": "ocr"})
    elif len(ocr_words) >= 2:
        phrase = " ".join(ocr_words[:3])
        if len(phrase) > 5:
            queries.append({"text": phrase, "embedding": None, "source": "ocr"})

    # 3. Filename query: extract at most 1 filename query
    stem = Path(fname).stem
    parts = re.split(r"[^A-Za-z0-9]+", stem)
    meaningful = [
        p for p in parts
        if p.lower() not in SKIP
        and len(p) > 3
        and not p.isdigit()
        and any(ch.isalpha() for ch in p)
        and not re.fullmatch(r"[0-9a-fA-F]{8,}", p)
    ]
    if meaningful:
        queries.append({"text": " ".join(meaningful[:3]), "embedding": None, "source": "filename"})

    return queries


def extract_queries(meta: Dict) -> List[str]:
    """Backward-compatible helper for diagnostics/tests."""
    return [q["text"] for q in extract_query_items(meta)]


# ── Dataset ───────────────────────────────────────────────────────────────────

class RECALIXDataset(Dataset):
    """
    Triplet dataset for QCF training.

    Each item: (query_text_emb, pos_signals[4], neg_signals[4], query_str)

    query_text_emb: CLIP text embedding of a query phrase (512,)
    pos_signals:    4-signal vector for the image that should match
    neg_signals:    4-signal vector for a random non-matching image
    query_str:      raw query text (needed for semi-hard mining)

    All CLIP calls happen at build time (single thread).
    __getitem__ returns pre-computed tensors — no GPU calls inside DataLoader.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata:   List[Dict],
        embed_fn:   Callable[[str], np.ndarray],
        split:      str = "train",
        top_k_neg:  int = 50,
        log_cb:     Optional[Callable] = None,
        data_dir:   Optional[str] = None,
        negative_mode: Optional[str] = None,
    ):
        self.embeddings = embeddings
        self.metadata   = metadata
        self.embed_fn   = embed_fn
        self.split      = split
        self.top_k_neg  = top_k_neg
        self.negative_mode = (negative_mode or os.environ.get("RECALIX_NEGATIVE_MODE", "") or "topk").lower()
        self._log       = log_cb or (lambda s: None)
        self.candidate_embedding_indices = np.array([], dtype=np.int64)
        self.data_dir = Path(data_dir) if data_dir else None
        self.build_info = {
            "split": split,
            "negative_mode": self.negative_mode,
            "cache": "disabled",
            "cache_path": None,
            "images_scanned": 0,
            "examples": 0,
        }

        # embedding_index → position in metadata list
        self._emb_to_meta = {
            m["embedding_index"]: i
            for i, m in enumerate(metadata)
            if m.get("embedding_index") is not None
        }

        self.examples = self._build()

    def _build(self) -> List[Dict]:
        N = len(self.metadata)
        if N < 3:
            return []

        cache_path = self._cache_path()
        if cache_path and cache_path.exists():
            try:
                payload = pickle.loads(cache_path.read_bytes())
                examples = payload.get("examples", [])
                self.candidate_embedding_indices = np.asarray(
                    payload.get("candidate_embedding_indices", []), dtype=np.int64
                )
                self.build_info.update(payload.get("build_info", {}))
                self.build_info.update({"cache": "hit", "cache_path": str(cache_path), "examples": len(examples)})
                self._log(f"  {self.split}: loaded {len(examples)} cached examples ({self.negative_mode})")
                return examples
            except Exception as e:
                self._log(f"  Cache ignored for {self.split}: {e}")

        # 80/10/10 train/val/test split (deterministic)
        indices = list(range(N))
        rng = random.Random(42)
        rng.shuffle(indices)
        if self.split == "train":
            indices = indices[: int(0.80 * N)]
        elif self.split == "val":
            indices = indices[int(0.80 * N): int(0.90 * N)]
        else:
            indices = indices[int(0.90 * N):]
        limit = int(os.environ.get("RECALIX_TRAIN_LIMIT", "0") or 0)
        if limit > 0:
            indices = indices[:limit]
            self._log(f"  Split {self.split}: limited to {len(indices)} rows by RECALIX_TRAIN_LIMIT")
        self._log(
            f"  Split {self.split}: {len(indices)}/{N} indexed rows "
            f"({len(indices) / max(N, 1):.0%}) using {self.negative_mode} negatives"
        )

        candidate_embedding_indices = [
            int(self.metadata[i]["embedding_index"])
            for i in indices
            if self.metadata[i].get("embedding_index") is not None
            and self.metadata[i]["embedding_index"] < len(self.embeddings)
        ]
        self.candidate_embedding_indices = np.array(candidate_embedding_indices, dtype=np.int64)
        candidate_pool = self.candidate_embedding_indices

        # Use ALL candidates by default (0 = unlimited).  Set RECALIX_NEGATIVE_SAMPLE=5000
        # to revert to the old capped behaviour for faster builds on very large datasets.
        sample_limit = int(os.environ.get("RECALIX_NEGATIVE_SAMPLE", "0") or 0)
        if sample_limit > 0 and len(candidate_pool) > sample_limit:
            rng = np.random.default_rng(42)
            candidate_pool = rng.choice(candidate_pool, size=sample_limit, replace=False)
            self._log(
                f"  {self.split}: sampled {len(candidate_pool)} of {len(self.candidate_embedding_indices)} candidates (RECALIX_NEGATIVE_SAMPLE={sample_limit})"
            )
        else:
            self._log(
                f"  {self.split}: using ALL {len(candidate_pool)} candidates as negative pool (full library coverage)"
            )
        self.candidate_embedding_indices = np.array(candidate_pool, dtype=np.int64)
        coverage_pct = round(len(self.candidate_embedding_indices) / max(len(self.embeddings), 1) * 100, 1)
        self.build_info.update({
            "cache": "miss" if cache_path else "disabled",
            "cache_path": str(cache_path) if cache_path else None,
            "images_scanned": len(indices),
            "indexed_rows": N,
            "total_embeddings": len(self.embeddings),
            "split_fraction": round(len(indices) / max(N, 1), 4),
            "candidate_images": len(self.candidate_embedding_indices),
            "negative_pool_size": len(self.candidate_embedding_indices),
            "negative_pool_coverage_pct": coverage_pct,
            "negative_sample_limit": sample_limit,
        })

        # ── 1. Collect all queries to embed ──
        query_requests = []
        raw_queries_to_embed = set()

        for idx in indices:
            meta = self.metadata[idx]
            emb_idx = meta.get("embedding_index")
            if emb_idx is None or emb_idx >= len(self.embeddings):
                continue

            for q_item in extract_query_items(meta):
                q_str = q_item["text"]
                q_emb = q_item.get("embedding")

                if q_emb is not None:
                    q_emb = np.asarray(q_emb, dtype=np.float32)
                else:
                    raw_queries_to_embed.add(q_str)

                query_requests.append({
                    "meta": meta,
                    "emb_idx": int(emb_idx),
                    "q_str": q_str,
                    "q_emb": q_emb,
                    "source": q_item.get("source", "unknown")
                })

        # ── 2. Batched text encoding of queries ──
        query_to_emb = {}
        if raw_queries_to_embed:
            raw_list = list(raw_queries_to_embed)
            self._log(f"  {self.split}: batched encoding {len(raw_list)} unique query texts...")
            t_enc_start = time.time()

            try:
                from app.core import engine
                if hasattr(engine, "embed_texts"):
                    embs = engine.embed_texts(raw_list)
                else:
                    embs_list = []
                    for k in range(0, len(raw_list), 256):
                        batch = raw_list[k:k+256]
                        embs_list.extend([self.embed_fn(q) for q in batch])
                    embs = np.array(embs_list, dtype=np.float32)
            except Exception as e:
                self._log(f"  Batched text embedding failed ({e}), falling back to serial...")
                embs = np.array([self.embed_fn(q) for q in raw_list], dtype=np.float32)

            for q, emb in zip(raw_list, embs):
                query_to_emb[q] = emb
            self._log(f"  {self.split}: encoded query texts in {time.time()-t_enc_start:.1f}s")

        # Fill in missing query embeddings
        valid_requests = []
        for req in query_requests:
            if req["q_emb"] is None:
                req["q_emb"] = query_to_emb.get(req["q_str"])
            if req["q_emb"] is not None:
                valid_requests.append(req)

        # ── 2.5. Balance Query Sources ──
        from collections import defaultdict
        grouped = defaultdict(list)
        for req in valid_requests:
            grouped[req["source"]].append(req)

        self._log(f"  {self.split}: initial query source counts: " + ", ".join(f"{k}: {len(v)}" for k, v in grouped.items()))

        # Scale query caps proportionally to dataset size — larger datasets
        # produce proportionally more training data for better variety coverage.
        base_cap = 8000 if self.split == "train" else 1000
        target_cap = max(base_cap, len(indices) // 3) if self.split == "train" else max(base_cap, len(indices) // 5)
        balanced_requests = []
        rng_balance = random.Random(42)
        for src, reqs in grouped.items():
            if len(reqs) > target_cap:
                balanced_requests.extend(rng_balance.sample(reqs, target_cap))
            else:
                balanced_requests.extend(reqs)

        rng_balance.shuffle(balanced_requests)
        valid_requests = balanced_requests

        # ── 3. Chunked Negative Mining (memory-safe for large datasets) ──
        #
        # For large candidate pools (100K+ images), the full similarity matrix
        # q_embs @ cand_embs.T would be too large (e.g., 4096 × 238K = ~3.6 GB).
        # We process candidates in chunks of CAND_CHUNK_SIZE and merge top-K
        # across chunks, keeping peak memory under ~1 GB.
        #
        examples = []
        t0 = time.time()
        last_log = t0
        source_counts: Dict[str, int] = {}
        neg_counts = {"random": 0, "topk": 0}

        candidate_pool = self.candidate_embedding_indices
        n_cand = len(candidate_pool)
        self._log(f"  {self.split}: processing {len(valid_requests)} queries against {n_cand:,} candidates...")

        QUERY_BATCH = 2048
        # Chunk size for candidate embeddings — keeps memory under ~1 GB
        # 2048 queries × 8192 candidates × 4 bytes ≈ 64 MB per chunk
        CAND_CHUNK = min(n_cand, 8192)
        use_chunked = n_cand > CAND_CHUNK
        if use_chunked:
            self._log(f"  {self.split}: using chunked similarity ({n_cand:,} candidates in chunks of {CAND_CHUNK:,})")

        # Pre-compute positive index positions in candidate_pool for fast masking
        cand_pool_set = {int(v): i for i, v in enumerate(candidate_pool)}

        for start_idx in range(0, len(valid_requests), QUERY_BATCH):
            end_idx = min(start_idx + QUERY_BATCH, len(valid_requests))
            batch = valid_requests[start_idx:end_idx]
            B = len(batch)

            q_embs = np.stack([q["q_emb"] for q in batch])  # (B, 512)

            top_k = min(self.top_k_neg, n_cand - 1)

            if top_k >= 1 and self.negative_mode != "random":
                if use_chunked:
                    # ── Chunked top-K: process candidates in chunks ──
                    # We keep a running top-K buffer per query across all chunks
                    merged_topk_idx = np.zeros((B, top_k), dtype=np.int64)  # pool-relative indices
                    merged_topk_sim = np.full((B, top_k), -np.inf, dtype=np.float32)

                    for c_start in range(0, n_cand, CAND_CHUNK):
                        c_end = min(c_start + CAND_CHUNK, n_cand)
                        chunk_embs = self.embeddings[candidate_pool[c_start:c_end]]  # (chunk, 512)
                        chunk_sims = q_embs @ chunk_embs.T  # (B, chunk)

                        # Mask out positives within this chunk
                        for b_idx in range(B):
                            pos_global = batch[b_idx]["emb_idx"]
                            pos_in_pool = cand_pool_set.get(pos_global)
                            if pos_in_pool is not None and c_start <= pos_in_pool < c_end:
                                chunk_sims[b_idx, pos_in_pool - c_start] = -np.inf

                        # Merge this chunk's top-K with running buffer
                        chunk_k = min(top_k, chunk_sims.shape[1])
                        if chunk_k < 1:
                            continue
                        chunk_topk_local = np.argpartition(-chunk_sims, chunk_k - 1, axis=1)[:, :chunk_k]
                        chunk_topk_global = chunk_topk_local + c_start  # convert to pool-relative index

                        # Gather similarity values
                        chunk_topk_vals = np.take_along_axis(chunk_sims, chunk_topk_local, axis=1)

                        # Concatenate with running buffer and keep top-K
                        combined_idx = np.concatenate([merged_topk_idx, chunk_topk_global], axis=1)
                        combined_sim = np.concatenate([merged_topk_sim, chunk_topk_vals], axis=1)
                        keep_k = min(top_k, combined_sim.shape[1])
                        keep_pos = np.argpartition(-combined_sim, keep_k - 1, axis=1)[:, :keep_k]
                        merged_topk_idx = np.take_along_axis(combined_idx, keep_pos, axis=1)
                        merged_topk_sim = np.take_along_axis(combined_sim, keep_pos, axis=1)

                    top_k_pos = merged_topk_idx
                else:
                    # ── Small pool: single matrix multiply ──
                    cand_embs = self.embeddings[candidate_pool]  # (N, 512)
                    sims = q_embs @ cand_embs.T  # (B, N)

                    for b_idx in range(B):
                        pos_in_pool = cand_pool_set.get(batch[b_idx]["emb_idx"])
                        if pos_in_pool is not None:
                            sims[b_idx, pos_in_pool] = -np.inf

                    top_k_pos = np.argpartition(-sims, top_k - 1, axis=1)[:, :top_k]
            else:
                top_k_pos = None

            for b_idx, q in enumerate(batch):
                emb_idx = q["emb_idx"]
                meta = q["meta"]
                q_emb = q["q_emb"]
                q_str = q["q_str"]
                source = q["source"]

                use_random = self.negative_mode == "random"
                if self.negative_mode in {"mixed", "semi_hard"}:
                    use_random = random.random() < 0.5

                candidate_ids = candidate_pool[candidate_pool != emb_idx]
                if len(candidate_ids) < 1:
                    continue

                if use_random or top_k_pos is None or top_k < 1:
                    neg_emb_idx = int(random.choice(candidate_ids))
                    neg_kind = "random"
                else:
                    pool_positions = top_k_pos[b_idx]
                    neg_emb_idx = int(candidate_pool[random.choice(pool_positions)])
                    neg_kind = "topk"

                neg_meta_idx = self._emb_to_meta.get(neg_emb_idx)
                if neg_meta_idx is None:
                    continue

                pos_sig = compute_4signals(q_emb, q_str, meta, self.embeddings)
                neg_sig = compute_4signals(q_emb, q_str, self.metadata[neg_meta_idx], self.embeddings)

                source_counts[source] = source_counts.get(source, 0) + 1
                neg_counts[neg_kind] = neg_counts.get(neg_kind, 0) + 1

                examples.append({
                    "query_emb": q_emb.astype(np.float32),
                    "query_str": q_str,
                    "query_source": source,
                    "negative_kind": neg_kind,
                    "pos_emb_idx": int(emb_idx),
                    "neg_emb_idx": int(neg_emb_idx),
                    "pos_sig": pos_sig,
                    "neg_sig": neg_sig,
                })

            now = time.time()
            if now - last_log >= 10:
                last_log = now
                self._log(
                    f"  Building {self.split}: {end_idx}/{len(valid_requests)} queries "
                    f"-> {len(examples)} examples ({now - t0:.0f}s)"
                )

        self._log(f"  {self.split}: {len(examples)} examples from {len(indices)} images")
        self.build_info.update({
            "examples": len(examples),
            "query_sources": source_counts,
            "negative_kinds": neg_counts,
            "build_seconds": round(time.time() - t0, 2),
        })
        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(pickle.dumps({
                    "examples": examples,
                    "candidate_embedding_indices": self.candidate_embedding_indices,
                    "build_info": self.build_info,
                }, protocol=pickle.HIGHEST_PROTOCOL))
                self._log(f"  {self.split}: cached examples -> {cache_path}")
            except Exception as e:
                self._log(f"  Cache save skipped for {self.split}: {e}")
        return examples

    def _cache_path(self) -> Optional[Path]:
        if not self.data_dir or os.environ.get("RECALIX_DISABLE_EXAMPLE_CACHE"):
            return None
        meta = {
            "version": "recalix-example-cache-v2",
            "split": self.split,
            "negative_mode": self.negative_mode,
            "top_k_neg": self.top_k_neg,
            "embeddings_shape": tuple(self.embeddings.shape),
            "metadata_count": len(self.metadata),
        }
        for name in ("embeddings_meta.json", "metadata.db"):
            p = self.data_dir / name
            if p.exists():
                stat = p.stat()
                meta[name] = {"mtime": stat.st_mtime_ns, "size": stat.st_size}
        key = hashlib.sha1(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return self.data_dir / "training_example_cache" / f"{self.split}_{self.negative_mode}_{key}.pkl"

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        e = self.examples[i]
        return (
            torch.tensor(e["query_emb"], dtype=torch.float32),
            torch.tensor(e["pos_sig"],   dtype=torch.float32),
            torch.tensor(e["neg_sig"],   dtype=torch.float32),
            e["query_str"],
        )


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir: str) -> Tuple[np.ndarray, List[Dict]]:
    """Load embeddings and metadata from the RECALIX data directory."""
    data_dir = Path(data_dir)
    emb_path = data_dir / "embeddings.npy"
    db_path  = data_dir / "metadata.db"
    meta_f   = data_dir / "embeddings_meta.json"

    if not emb_path.exists():
        raise FileNotFoundError(f"embeddings.npy not found at {emb_path}\nIndex photos first via the UI.")
    if not db_path.exists():
        raise FileNotFoundError(f"metadata.db not found at {db_path}")
    if not meta_f.exists():
        raise FileNotFoundError("embeddings_meta.json missing. Re-index to regenerate.")

    meta_json = json.loads(meta_f.read_text())
    count     = meta_json["count"]
    capacity  = meta_json["capacity"]
    EMBED_DIM = 512

    emb_raw    = np.memmap(str(emb_path), dtype=np.float32, mode="r", shape=(capacity, EMBED_DIM))
    embeddings = np.array(emb_raw[:count], dtype=np.float32)

    # Ensure L2 normalized
    norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms      = np.where(norms < 1e-8, 1.0, norms)
    embeddings = embeddings / norms

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM media WHERE embedding_index IS NOT NULL ORDER BY embedding_index"
    ).fetchall()
    caption_rows = []
    has_captions = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='captions'"
    ).fetchone()
    if has_captions:
        caption_rows = conn.execute(
            """
            SELECT image_id, caption, embedding
            FROM captions
            WHERE caption IS NOT NULL
              AND embedding IS NOT NULL
              AND length(trim(caption)) > 0
            ORDER BY id
            """
        ).fetchall()
    conn.close()

    metadata = [dict(r) for r in rows]
    metadata = [m for m in metadata if m["embedding_index"] < count]
    caption_map: Dict[int, List[str]] = {}
    for r in caption_rows:
        image_id = int(r["image_id"])
        if len(caption_map.get(image_id, [])) >= 2:
            continue
        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        if emb.shape[0] != EMBED_DIM:
            continue
        emb = np.array(emb, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 1e-8:
            emb = emb / norm
        caption_map.setdefault(image_id, []).append({
            "text": str(r["caption"]),
            "embedding": emb,
        })
    for m in metadata:
        m["captions"] = caption_map.get(int(m["id"]), [])

    return embeddings, metadata


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, examples: List[Dict], device: str) -> float:
    """
    Pairwise accuracy: fraction of examples where pos_score > neg_score.
    This is the main metric for your paper.
    Random baseline = 0.5. Perfect = 1.0.
    """
    if not examples:
        return 0.0

    subset = examples[: min(500, len(examples))]
    q_t    = torch.tensor(np.stack([e["query_emb"] for e in subset]), dtype=torch.float32).to(device)
    pos_t  = torch.tensor(np.stack([e["pos_sig"]   for e in subset]), dtype=torch.float32).to(device)
    neg_t  = torch.tensor(np.stack([e["neg_sig"]   for e in subset]), dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        w         = model(q_t)
        pos_score = (w * pos_t).sum(dim=1)
        neg_score = (w * neg_t).sum(dim=1)
        acc       = (pos_score > neg_score).float().mean().item()
    model.train()
    return acc


def evaluate_fixed_weights(weights: list, examples: List[Dict], limit: int = 500) -> float:
    """Pairwise accuracy with fixed weights, used for baseline tables."""
    if not examples:
        return 0.0
    w = np.array(weights, dtype=np.float32)
    subset = examples[: min(limit, len(examples))]
    correct = 0
    for e in subset:
        pos_score = float(np.dot(w, e["pos_sig"]))
        neg_score = float(np.dot(w, e["neg_sig"]))
        if pos_score > neg_score:
            correct += 1
    return correct / len(subset)


def evaluate_fixed_baselines(examples: List[Dict]) -> Dict[str, float]:
    baselines = {
        "Visual only": [1.0, 0.0, 0.0, 0.0],
        "OCR text only": [0.0, 1.0, 0.0, 0.0],
        "Filename only": [0.0, 0.0, 1.0, 0.0],
        "Uniform": [0.25, 0.25, 0.25, 0.25],
        "V5 fixed": [0.55, 0.30, 0.10, 0.05],
    }
    return {name: evaluate_fixed_weights(weights, examples) for name, weights in baselines.items()}


def dataset_summary(train_examples: List[Dict], val_examples: List[Dict]) -> Dict:
    def source_counts(examples):
        counts: Dict[str, int] = {}
        for e in examples:
            src = e.get("query_source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return counts

    return {
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "train_query_sources": source_counts(train_examples),
        "val_query_sources": source_counts(val_examples),
    }


def process_log(train_ds, val_ds, mode: str) -> List[Dict]:
    """Paper/debug friendly explanation of how this run was constructed."""
    train_info = getattr(train_ds, "build_info", {})
    val_info = getattr(val_ds, "build_info", {})
    return [
        {
            "stage": "data_load",
            "detail": "Loaded normalized image embeddings and metadata rows from backend/data.",
            "train_images_scanned": train_info.get("images_scanned"),
            "val_images_scanned": val_info.get("images_scanned"),
        },
        {
            "stage": "query_generation",
            "detail": "Generated query phrases from captions first, then OCR snippets, then meaningful filename tokens.",
            "train_query_sources": train_info.get("query_sources", {}),
            "val_query_sources": val_info.get("query_sources", {}),
        },
        {
            "stage": "example_cache",
            "detail": "Reused cached training examples when embeddings and metadata were unchanged.",
            "train_cache": train_info.get("cache"),
            "val_cache": val_info.get("cache"),
            "train_cache_path": train_info.get("cache_path"),
            "val_cache_path": val_info.get("cache_path"),
        },
        {
            "stage": "negative_mining",
            "detail": "Created each pair as one positive image and one non-matching negative image.",
            "mode": mode,
            "negative_mode": train_info.get("negative_mode"),
            "train_negative_kinds": train_info.get("negative_kinds", {}),
            "val_negative_kinds": val_info.get("negative_kinds", {}),
        },
        {
            "stage": "optimization",
            "detail": "Trained QueryConditionedFusion to assign per-query weights over visual, OCR, filename, and structural signals.",
            "train_examples": train_info.get("examples"),
            "val_examples": val_info.get("examples"),
        },
    ]


def save_research_artifacts(
    mode: str,
    output_path: str,
    metrics: List[Dict],
    summary: Dict,
    baseline_results: Optional[Dict[str, float]] = None,
    result: Optional[Dict] = None,
    process: Optional[List[Dict]] = None,
) -> Dict:
    """
    Save paper-friendly artifacts: summary JSON, epoch CSV, baseline CSV,
    Markdown table, and a learning-curve PNG when matplotlib is available.
    """
    root = Path(output_path).resolve().parents[2]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = root / "training" / "artifacts" / "research_runs" / f"{mode}_{run_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "run_id": run_id,
        "mode": mode,
        "created_at": datetime.now().isoformat(),
        "model_path": str(output_path),
        "dataset": summary,
        "result": result or {},
        "baselines": baseline_results or {},
        "metrics": metrics,
        "process": process or [],
    }
    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    metrics_path = artifact_dir / "metrics.csv"
    metric_fields = sorted({k for row in metrics for k in row.keys()}) or ["epoch"]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metrics)

    baseline_path = artifact_dir / "baseline_table.csv"
    with baseline_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "pairwise_accuracy"])
        writer.writeheader()
        for name, acc in (baseline_results or {}).items():
            writer.writerow({"method": name, "pairwise_accuracy": acc})

    table_path = artifact_dir / "table.md"
    lines = [
        f"# RECALIX Training Run: {mode}",
        "",
        "## Dataset",
        "",
        f"- Train examples: {summary.get('train_examples', 0)}",
        f"- Validation examples: {summary.get('val_examples', 0)}",
        f"- Train query sources: {summary.get('train_query_sources', {})}",
        f"- Validation query sources: {summary.get('val_query_sources', {})}",
        "",
        "## Baselines",
        "",
        "| Method | Pairwise Accuracy |",
        "|---|---:|",
    ]
    for name, acc in (baseline_results or {}).items():
        lines.append(f"| {name} | {acc:.4f} |")
    if result:
        lines.extend(["", "## Result", "", "| Metric | Value |", "|---|---:|"])
        for k, v in result.items():
            if isinstance(v, (int, float)):
                lines.append(f"| {k} | {v:.6f} |")
            else:
                lines.append(f"| {k} | {v} |")
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_path = artifact_dir / "learning_curve.png"
    svg_path = artifact_dir / "learning_curve.svg"
    try:
        import matplotlib  # type: ignore[import-not-found]
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]

        epochs = [m.get("epoch") for m in metrics if m.get("epoch") is not None]
        losses = [m.get("loss") for m in metrics if m.get("loss") is not None]
        accs = [m.get("val_acc") for m in metrics if m.get("val_acc") is not None]
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        if epochs and losses:
            ax1.plot(epochs, losses, marker="o", label="loss", color="#1f77b4")
            ax1.set_ylabel("Loss")
        ax2 = ax1.twinx()
        if epochs and accs:
            ax2.plot(epochs, accs, marker="s", label="val_acc", color="#2ca02c")
            ax2.set_ylabel("Validation accuracy")
            ax2.set_ylim(0, 1.05)
        ax1.set_xlabel("Epoch")
        ax1.set_title(f"RECALIX {mode} training")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
    except Exception:
        plot_path = None
        if metrics:
            w, h = 760, 420
            pad = 50
            epochs = [float(m.get("epoch", i + 1)) for i, m in enumerate(metrics)]
            losses = [float(m.get("loss", 0.0)) for m in metrics]
            accs = [float(m.get("val_acc", 0.0)) for m in metrics]
            x_min, x_max = min(epochs), max(epochs)
            loss_min, loss_max = min(losses), max(losses)
            if abs(loss_max - loss_min) < 1e-9:
                loss_max = loss_min + 1.0

            def sx(x):
                return pad + (x - x_min) / max(x_max - x_min, 1.0) * (w - 2 * pad)

            def sy_loss(y):
                return h - pad - (y - loss_min) / (loss_max - loss_min) * (h - 2 * pad)

            def sy_acc(y):
                return h - pad - y * (h - 2 * pad)

            loss_points = " ".join(f"{sx(x):.1f},{sy_loss(y):.1f}" for x, y in zip(epochs, losses))
            acc_points = " ".join(f"{sx(x):.1f},{sy_acc(y):.1f}" for x, y in zip(epochs, accs))
            svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<rect width="100%" height="100%" fill="white"/>
<text x="{w/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">RECALIX {mode} training</text>
<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#333"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#333"/>
<polyline points="{loss_points}" fill="none" stroke="#1f77b4" stroke-width="3"/>
<polyline points="{acc_points}" fill="none" stroke="#2ca02c" stroke-width="3"/>
<text x="{pad}" y="{pad-12}" font-family="Arial" font-size="12" fill="#1f77b4">loss</text>
<text x="{w-pad-90}" y="{pad-12}" font-family="Arial" font-size="12" fill="#2ca02c">val_acc</text>
<text x="{w/2}" y="{h-12}" text-anchor="middle" font-family="Arial" font-size="12">epoch</text>
</svg>
"""
            svg_path.write_text(svg, encoding="utf-8")

    paths = {
        "artifact_dir": str(artifact_dir),
        "summary_json": str(summary_path),
        "metrics_csv": str(metrics_path),
        "baseline_csv": str(baseline_path),
        "table_md": str(table_path),
    }
    if plot_path:
        paths["learning_curve_png"] = str(plot_path)
    elif svg_path.exists():
        paths["learning_curve_svg"] = str(svg_path)
    return paths


# ── Base Trainer ──────────────────────────────────────────────────────────────

class BaseTrainer:
    """All training modes extend this. Handles setup: device, CLIP, data, model."""

    MODE = "base"

    def _setup(self, data_dir: str, epochs: int, lr: float, batch_size: int, log_cb: Callable):
        backend = Path(data_dir).parent.parent / "backend"
        if backend.exists() and str(backend) not in sys.path:
            sys.path.insert(0, str(backend))

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log_cb(f"Device: {self.device}")

        # Try GPU kernel test
        if self.device == "cuda":
            try:
                t = torch.zeros(1).cuda(); _ = t + 1; del t, _
            except Exception:
                log_cb("⚠ CUDA kernel failed — switching to CPU")
                self.device = "cpu"

        # Load CLIP text encoder
        log_cb("Loading CLIP text encoder…")
        try:
            from app.core.engine import embed_text, warmup
            warmup()
            self.embed_fn = embed_text
            log_cb("✓ CLIP loaded from backend engine")
        except Exception as e:
            log_cb(f"  Backend engine unavailable ({e}), loading open_clip directly…")
            import open_clip
            _m, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            _m = _m.to(self.device).eval()
            _tok = open_clip.get_tokenizer("ViT-B-32")
            def _embed(text: str) -> np.ndarray:
                with torch.no_grad():
                    tok = _tok([text]).to(self.device)
                    e   = _m.encode_text(tok)
                    e   = e / e.norm(dim=-1, keepdim=True)
                return e.cpu().numpy()[0]
            self.embed_fn = _embed
            log_cb("✓ CLIP loaded via open_clip")

        # Load data
        log_cb("Loading embeddings + metadata…")
        self.embeddings, self.metadata = load_data(data_dir)
        log_cb(f"  Embeddings: {self.embeddings.shape}  |  Rows: {len(self.metadata)}")

        if len(self.metadata) < 10:
            raise RuntimeError(f"Only {len(self.metadata)} indexed images. Index at least 50+ photos first.")

        # Model + optimizer
        from app.core.fusion_model import QueryConditionedFusion, ranking_loss, save_model
        self.model      = QueryConditionedFusion().to(self.device)
        self.loss_fn    = ranking_loss
        self.save_model = save_model
        n_params = sum(p.numel() for p in self.model.parameters())
        log_cb(f"Model: QueryConditionedFusion ({n_params:,} parameters)")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

    def run(self, data_dir, output_path, epochs, lr, batch_size, log_cb, epoch_cb) -> Dict:
        raise NotImplementedError
