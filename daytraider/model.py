"""LSTM classifier predicting the probability the next bar closes up."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config
from data import FEATURE_COLUMNS


class DayTraderNet(nn.Module):
    def __init__(self, n_features: int = len(FEATURE_COLUMNS),
                 hidden_size: int = config.HIDDEN_SIZE,
                 num_layers: int = config.NUM_LAYERS,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)  # logits


class Scaler:
    """Per-feature standardization fit on training data only."""

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, X: np.ndarray) -> "Scaler":
        flat = X.reshape(-1, X.shape[-1])
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def save(self, path: Path = config.SCALER_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"mean": self.mean.tolist(), "std": self.std.tolist()}))

    @classmethod
    def load(cls, path: Path = config.SCALER_PATH) -> "Scaler":
        d = json.loads(path.read_text())
        return cls(np.array(d["mean"], np.float32), np.array(d["std"], np.float32))


def save_threshold(value: float, path: Path = config.THRESHOLD_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"prob_threshold": value}))


def load_threshold(path: Path = config.THRESHOLD_PATH) -> float:
    if path.exists():
        return float(json.loads(path.read_text())["prob_threshold"])
    return config.PROB_THRESHOLD


def seed_path(seed: int) -> Path:
    return config.MODEL_DIR / f"model_seed{seed}.pt"


def load_trained_model(path: Path = config.MODEL_PATH) -> DayTraderNet:
    model = DayTraderNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_ensemble() -> list[DayTraderNet]:
    """All trained seed models; falls back to the single model file."""
    paths = [p for p in (seed_path(s) for s in config.ENSEMBLE_SEEDS)
             if p.exists()]
    if not paths:
        paths = [config.MODEL_PATH]
    return [load_trained_model(p) for p in paths]


def ensemble_probs(models: list[DayTraderNet], X: torch.Tensor) -> torch.Tensor:
    """Mean sigmoid probability across the ensemble (X already scaled)."""
    with torch.no_grad():
        return torch.stack([torch.sigmoid(m(X)) for m in models]).mean(dim=0)
