"""
RECALIX V10 — Query-Conditioned Fusion Model
=============================================
This is the core research contribution.

Architecture:
  Input:  512-dim CLIP text embedding of the search query
  Hidden: 256 → 128 → 64
  Output: 4 weights (visual, text, filename, structural) via softmax

At search time:
  final_score = w[0]*visual + w[1]*ocr + w[2]*filename + w[3]*structural

The model learns query-adaptive weights:
  "sunset on beach" → high visual weight
  "invoice amount"  → high OCR weight
  "whatsapp msg"    → high filename weight
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryConditionedFusion(nn.Module):
    """
    Lightweight MLP: 512 → 256 → 128 → 64 → 4 (softmax)
    ~148K parameters. Trained with triplet ranking loss.
    """
    OUTPUT_KEYS = ["visual", "text", "filename", "structural"]

    def __init__(self, input_dim: int = 512, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )
        # Initialize to uniform weights
        with torch.no_grad():
            self.net[-1].bias.fill_(0.0)
            self.net[-1].weight.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 512) → (B, 4) weights summing to 1.0"""
        logits = self.net(x)
        return F.softmax(logits / self.temperature, dim=-1)

    def predict_weights(self, query_embedding: np.ndarray) -> Dict[str, float]:
        """Convenience: numpy (512,) in → weight dict out. Device-safe."""
        self.eval()
        # Move input to same device as model
        device = next(self.parameters()).device
        with torch.no_grad():
            t = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0).to(device)
            w = self(t)[0].cpu().numpy()
        return dict(zip(self.OUTPUT_KEYS, w.tolist()))


def ranking_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float = 0.4,
) -> torch.Tensor:
    """
    Triplet ranking loss (per-sample, then averaged).
    margin=0.4: forces strong separation between positive and negative.
    Higher margin = harder training = model must actually learn signal differences.

    loss = mean(max(0, neg_score - pos_score + margin))
    """
    return F.relu(neg_scores - pos_scores + margin).mean()


def save_model(model: QueryConditionedFusion, path: str, meta: dict = None):
    """Save model checkpoint with metadata."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "saved_at":   datetime.now().isoformat(),
        "meta":        meta or {},
    }
    try:
        torch.save(checkpoint, path)
    except (OSError, RuntimeError) as e:
        target = Path(path)
        repo_root = Path(__file__).resolve().parents[3]
        fallback_dir = repo_root / "training" / "artifacts" / "checkpoints"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback = fallback_dir / f"{target.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{target.suffix}"
        checkpoint["meta"]["save_warning"] = f"Primary path locked/unavailable: {e}"
        checkpoint["meta"]["intended_path"] = str(target)
        torch.save(checkpoint, fallback)


def load_model(path: str, device: str = "cpu") -> QueryConditionedFusion:
    """Load model from checkpoint. Returns model on specified device."""
    checkpoint = torch.load(path, map_location=device)
    model = QueryConditionedFusion()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
