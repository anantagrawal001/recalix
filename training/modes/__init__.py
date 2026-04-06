"""Training mode registry — returns the right trainer by name."""

from modes.basic     import BasicTrainer
from modes.baseline  import BaselineTrainer
from modes.semi_hard import SemiHardTrainer
from modes.hard      import HardTrainer

_REGISTRY = {
    "basic":     BasicTrainer,
    "baseline":  BaselineTrainer,
    "semi_hard": SemiHardTrainer,
    "hard":      HardTrainer,
}


def get_trainer(mode: str):
    if mode not in _REGISTRY:
        raise ValueError(f"Unknown mode '{mode}'. Choose: {list(_REGISTRY.keys())}")
    return _REGISTRY[mode]()


def list_modes():
    return list(_REGISTRY.keys())
