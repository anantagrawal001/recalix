"""
RECALIX V9 — Query-Conditioned Fusion Model
============================================
Lightweight MLP predicting query-adaptive hybrid weights.

Architecture:
    Input  : 512-dim CLIP text embedding
    Layers : Linear → LayerNorm → GELU → Dropout  (×3 stages)
    Output : 4 weights via temperature-scaled softmax
             [visual, text, filename, structural]

Key research property: model learns WHEN each signal matters.
  Visual queries  → high visual weight
  OCR queries     → high text weight
  Filename queries→ high filename weight

Parameters: ~148K
Temperature scaling prevents weight collapse (all-0.25 local minima).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


class QueryConditionedFusion(nn.Module):

    OUTPUT_KEYS = ["visual", "text", "filename", "structural"]

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        temperature: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, 4))
        self.net = nn.Sequential(*layers)

        # Start near uniform weights
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """(B,512) → (B,4) sum-to-1 weights"""
        return F.softmax(self.net(query_embedding) / self.temperature, dim=-1)

    def predict_weights(self, query_embedding: np.ndarray) -> Dict[str, float]:
        """Convenience wrapper: numpy array in → weight dict out."""
        self.eval()
        with torch.no_grad():
            t = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0)
            w = self(t)[0].cpu().numpy()
        return dict(zip(self.OUTPUT_KEYS, w.tolist()))


def ranking_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """Triplet ranking loss: positive score should exceed negative by margin."""
    return F.relu(neg_scores - pos_scores + margin).mean()


def save_model(model: "QueryConditionedFusion", path: Path, metadata: Optional[dict] = None):
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": 512,
            "hidden_dims": (256, 128, 64),
            "temperature": model.temperature,
        },
        "metadata": metadata or {},
    }, str(path))


def load_model(path: Path, device: str = "cpu") -> Optional["QueryConditionedFusion"]:
    if not Path(path).exists():
        return None
    try:
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        cfg = ckpt.get("model_config", {})
        model = QueryConditionedFusion(
            input_dim=cfg.get("input_dim", 512),
            hidden_dims=tuple(cfg.get("hidden_dims", (256, 128, 64))),
            temperature=cfg.get("temperature", 0.5),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        import logging
        logging.getLogger("recalix.fusion").error(f"load_model failed: {e}")
        return None
