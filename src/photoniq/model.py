"""Compact mask-aware convolutional regression; no simulation-based pretraining."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MaskAwareSpectralCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, 7, padding=3), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 5, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 8, 64), nn.GELU(),
                                  nn.Dropout(.15), nn.Linear(64, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(features)).squeeze(1)


def mask_spectra(x: torch.Tensor, keep: int | None = None,
                 rng: np.random.Generator | None = None,
                 indices: np.ndarray | None = None) -> torch.Tensor:
    """Two channels: standardized measured intensity (zeros if hidden) and measurement mask."""
    rng = rng or np.random.default_rng()
    batch, points = x.shape
    if indices is not None:
        selected = np.asarray(indices, dtype=int)
        if not len(selected) or np.any(selected < 0) or np.any(selected >= points):
            raise ValueError('Invalid wavelength indices')
        masks = np.zeros((batch, points), dtype='float32')
        masks[:, selected] = 1
    elif keep is None:
        counts = rng.choice([16, 32, 64, 96, points], size=batch, p=[.10, .15, .2, .2, .35])
        masks = np.zeros((batch, points), dtype='float32')
        for i, count in enumerate(counts):
            masks[i, rng.choice(points, size=min(int(count), points), replace=False)] = 1
    else:
        if keep < 1 or keep > points:
            raise ValueError('keep must be in 1..n_wavelengths')
        chosen = np.unique(np.rint(np.linspace(0, points - 1, keep)).astype(int))
        masks = np.zeros((batch, points), dtype='float32')
        masks[:, chosen] = 1
    m = torch.tensor(masks, device=x.device, dtype=x.dtype)
    return torch.stack([x * m, m], dim=1)
