"""
RECALIX V9 Training — Shared Base
====================================
Shared dataset loading, triplet generation, and evaluation utilities.
All mode trainers inherit from BaseTrainer.
"""

import json
import random
import sqlite3
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from skimage import data_dir
import torch
from torch.utils.data import Dataset, DataLoader


# ── Dataset ───────────────────────────────────────────────────────────────────

class RECALIXDataset(Dataset):
    """
    Triplet dataset built from existing RECALIX DB + embeddings.
    
    Each sample: (query_embedding, positive_embedding, negative_embedding, query_type)
    query_type: "visual" | "ocr" | "filename" | "mixed"
    
    Queries are synthetic — generated from filename tokens and OCR text.
    This works entirely from your existing indexed data (no external data needed).
    """

    VISUAL_KW  = {"sunset","sunrise","landscape","mountain","ocean","beach","forest","sky",
                  "flower","dog","cat","animal","people","smiling","portrait","birthday",
                  "cake","party","wedding","snow","rain","cloud","colorful","outdoor","nature"}
    OCR_KW     = {"receipt","invoice","document","text","note","letter","form","contract",
                  "certificate","bank","statement","report","page","handwritten","whiteboard",
                  "sign","label","menu","ticket","screenshot"}
    FNAME_KW   = {"img","dsc","photo","image","file","backup","whatsapp","telegram",
                  "download","save","export","scan"}

    def __init__(self, embeddings: np.ndarray, metadata: List[Dict], split: str = "train"):
        self.embeddings = embeddings
        self.metadata   = metadata
        self.split      = split
        self.examples   = self._build_triplets()

    def _classify_query(self, query: str) -> str:
        tokens = set(query.lower().split())
        v = len(tokens & self.VISUAL_KW)
        o = len(tokens & self.OCR_KW)
        f = len(tokens & self.FNAME_KW)
        if v == o == f == 0: return "mixed"
        mx = max(v, o, f)
        if mx == v: return "visual"
        if mx == o: return "ocr"
        return "filename"

    def _build_triplets(self) -> List[Dict]:
        examples = []
        n = len(self.metadata)
        if n < 3:
            return examples

        # Split 80/10/10
        indices = list(range(n))
        random.seed(42)
        random.shuffle(indices)
        if self.split == "train":
            indices = indices[:int(0.8 * n)]
        elif self.split == "val":
            indices = indices[int(0.8 * n):int(0.9 * n)]
        else:
            indices = indices[int(0.9 * n):]

        # Build type-based pools for smarter negative sampling
        type_pools: Dict[str, List[int]] = {"image": [], "video": []}
        for i, m in enumerate(self.metadata):
            type_pools.setdefault(m.get("file_type", "image"), []).append(i)

        for idx in indices:
            meta   = self.metadata[idx]
            anchor = self.embeddings[idx]

            # Generate a plausible query from this item
            query_candidates = []
            if meta.get("ocr_text"):
                words = meta["ocr_text"].lower().split()[:6]
                if len(words) >= 2:
                    query_candidates.append((" ".join(words[:3]), "ocr"))
            if meta.get("filename_tokens"):
                words = meta["filename_tokens"].split()[:4]
                if words:
                    query_candidates.append((" ".join(words[:2]), "filename"))
            query_candidates.append(("photo image", "visual"))

            q_text, q_type = random.choice(query_candidates)
            q_type = self._classify_query(q_text)

            # Positive: same folder or same type (semantically close)
            same_folder = [i for i, m in enumerate(self.metadata)
                           if i != idx and m.get("folder") == meta.get("folder")
                           and self.embeddings[idx] is not None]
            if same_folder:
                pos_idx = random.choice(same_folder)
            else:
                pool    = [i for i in range(n) if i != idx]
                pos_idx = random.choice(pool)

            # Negative: random other item
            neg_candidates = [i for i in range(n) if i != idx and i != pos_idx]
            if not neg_candidates:
                continue
            neg_idx = random.choice(neg_candidates)

            examples.append({
                "query_emb": anchor.copy(),
                "pos_emb":   self.embeddings[pos_idx].copy(),
                "neg_emb":   self.embeddings[neg_idx].copy(),
                "query_type": q_type,
            })

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        e = self.examples[i]
        return (
            torch.tensor(e["query_emb"], dtype=torch.float32),
            torch.tensor(e["pos_emb"],   dtype=torch.float32),
            torch.tensor(e["neg_emb"],   dtype=torch.float32),
            e["query_type"],
        )


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir: str) -> Tuple[np.ndarray, List[Dict]]:
    """Load embeddings + metadata from RECALIX data directory."""
    data_dir = Path(data_dir)
    emb_path = data_dir / "embeddings.npy"
    db_path  = data_dir / "metadata.db"

    if not emb_path.exists():
        raise FileNotFoundError(f"embeddings.npy not found at {emb_path}\nIndex some photos first.")
    if not db_path.exists():
        raise FileNotFoundError(f"metadata.db not found at {db_path}\nIndex some photos first.")

    # Load meta file for count
    meta_file = data_dir / "embeddings_meta.json"
    count = None
    if meta_file.exists():
        import json as _json
        count = _json.loads(meta_file.read_text())["count"]

    # Load embeddings using memmap (because RECALIX stores them as raw memmap)
    # Load embedding metadata
    meta_file = data_dir / "embeddings_meta.json"
    if not meta_file.exists():
        raise FileNotFoundError("embeddings_meta.json missing. Re-index your dataset.")

    meta = json.loads(meta_file.read_text())

    count = meta["count"]
    capacity = meta["capacity"]
    EMBEDDING_DIM = 512

    # Load raw memmap embeddings
    emb_raw = np.memmap(
        str(emb_path),
        dtype=np.float32,
        mode="r",
        shape=(capacity, EMBEDDING_DIM)
    )

    # Only use the valid rows
    embeddings = np.array(emb_raw[:count], dtype=np.float32)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM media WHERE embedding_index IS NOT NULL ORDER BY embedding_index"
    ).fetchall()
    conn.close()

    metadata = [dict(r) for r in rows]
    # Align to embedding count
    metadata = [m for m in metadata if m["embedding_index"] < len(embeddings)]

    return embeddings, metadata


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, embeddings: np.ndarray, metadata: List[Dict], device: str) -> Dict:
    """
    3-bucket evaluation: visual / ocr / filename queries.
    Returns Recall@1, Recall@5, MRR per bucket.
    """
    from torch.utils.data import DataLoader as DL
    model.eval()

    dataset = RECALIXDataset(embeddings, metadata, split="test")
    if not dataset.examples:
        return {}

    buckets = {"visual": [], "ocr": [], "filename": [], "mixed": []}
    for ex in dataset.examples:
        buckets[ex["query_type"]].append(ex)

    results = {}
    all_emb = torch.tensor(embeddings, dtype=torch.float32).to(device)

    for bucket, examples in buckets.items():
        if not examples:
            continue
        r1s, r5s, mrrs = [], [], []

        for ex in examples[:200]:  # cap per bucket
            with torch.no_grad():
                q    = torch.tensor(ex["query_emb"], dtype=torch.float32).unsqueeze(0).to(device)
                w    = model(q)[0]
                # Use visual weight to re-score (simplified: weighted dot product)
                sims = (all_emb @ q.squeeze()).cpu().numpy()
            # Recall@K: is the positive in top K?
            pos_idx   = np.where((embeddings == ex["pos_emb"]).all(axis=1))[0]
            if not len(pos_idx):
                continue
            pos        = pos_idx[0]
            rank       = int(np.sum(sims >= sims[pos]))
            r1s.append(1.0 if rank <= 1 else 0.0)
            r5s.append(1.0 if rank <= 5 else 0.0)
            mrrs.append(1.0 / rank if rank > 0 else 0.0)

        if r1s:
            results[bucket] = {
                "recall@1": round(np.mean(r1s), 4),
                "recall@5": round(np.mean(r5s), 4),
                "mrr":      round(np.mean(mrrs), 4),
                "n":        len(r1s),
            }

    return results


# ── Base trainer ──────────────────────────────────────────────────────────────

class BaseTrainer:
    """All mode trainers extend this."""

    MODE = "base"

    def _setup(self, data_dir: str, epochs: int, lr: float, batch_size: int, log_cb):
        import sys
        from pathlib import Path as P
        # Make backend importable
        backend = P(data_dir).parent.parent / "backend"
        if backend.exists() and str(backend) not in sys.path:
            sys.path.insert(0, str(backend))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log_cb(f"Device: {self.device}")

        self.embeddings, self.metadata = load_data(data_dir)
        log_cb(f"Embedding shape: {self.embeddings.shape}")
        log_cb(f"Metadata rows: {len(self.metadata)}")
        log_cb(f"Loaded {len(self.embeddings)} embeddings, {len(self.metadata)} metadata rows")

        from pathlib import Path as P2
        backend_path = P2(data_dir).parent.parent / "backend"
        if str(backend_path) not in sys.path:
            sys.path.insert(0, str(backend_path))

        from app.core.fusion_model import QueryConditionedFusion, ranking_loss, save_model
        self.model      = QueryConditionedFusion().to(self.device)
        self.loss_fn    = ranking_loss
        self.save_model = save_model
        self.QCF        = QueryConditionedFusion

        self.optimizer  = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        log_cb(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def run(self, data_dir, output_path, epochs, lr, batch_size, log_cb, epoch_cb) -> Dict:
        raise NotImplementedError
