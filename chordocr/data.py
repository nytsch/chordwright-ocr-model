"""Die erzeugten Scheiben (`scripts/generate.py`) als Tensoren."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import encode


def load(directory: Path, limit: int | None = None) -> tuple[np.ndarray, list[str]]:
    images, labels = [], []
    for part in sorted(Path(directory).glob("part-*.npz")):
        with np.load(part) as data:
            images.append(data["images"])
            labels.extend(str(label) for label in data["labels"])
        if limit and len(labels) >= limit:
            break
    if not images:
        raise SystemExit(f"Keine Daten in {directory} — zuerst scripts/generate.py")
    stacked = np.concatenate(images)
    return (stacked[:limit], labels[:limit]) if limit else (stacked, labels)


def to_input(images: np.ndarray) -> torch.Tensor:
    """uint8 Papier=255 → float Tinte=1, (n, 1, 32, 192)."""
    return torch.from_numpy(1.0 - images.astype(np.float32) / 255.0).unsqueeze(1)


def batch_targets(labels: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [encode(label) for label in labels]
    flat = torch.tensor([k for e in encoded for k in e], dtype=torch.long)
    lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    return flat, lengths
