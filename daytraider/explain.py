"""What does the network actually key on? (permutation importance)

For each input feature, shuffle it across the test set and measure how much
the ensemble's accuracy drops. Big drop = the model relies on that feature;
~zero = it ignores it. This is the closest honest answer to "show me the
model's reasoning": it reveals which of the tools we built the network
actually uses to make its predictions.

Usage: python explain.py
"""
from __future__ import annotations

import numpy as np
import torch

from data import FEATURE_COLUMNS, load_dataset
from model import Scaler, ensemble_probs, load_ensemble


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def acc(models, X, y):
    p = torch.cat([ensemble_probs(models, X[i:i + 4096].to(DEVICE)).cpu()
                   for i in range(0, len(X), 4096)]).numpy()
    return ((p > 0.5) == (y > 0.5)).mean()


def main():
    # VALIDATION split, not test: features get SELECTED from this ranking
    # (pure_ml_slim.py), so measuring it on test would leak the test set into
    # model selection and flatter every later test score.
    X, y, *_ = load_dataset()["val"]
    scaler = Scaler.load()
    models = [m.to(DEVICE) for m in load_ensemble()]
    Xs = scaler.transform(X)
    base = acc(models, torch.from_numpy(Xs), y)
    print(f"Baseline validation accuracy: {base:.4f}\n")

    rng = np.random.default_rng(0)
    drops = []
    for j, name in enumerate(FEATURE_COLUMNS):
        Xp = Xs.copy()
        # Shuffle feature j across samples (break its signal, keep its
        # marginal distribution), averaged over a few shuffles.
        d = 0.0
        for _ in range(3):
            perm = rng.permutation(len(Xp))
            Xp[:, :, j] = Xs[perm][:, :, j]
            d += base - acc(models, torch.from_numpy(Xp), y)
        drops.append((name, d / 3))
        Xp[:, :, j] = Xs[:, :, j]

    drops.sort(key=lambda kv: -kv[1])
    print("Feature importance (accuracy drop when shuffled):")
    for name, d in drops:
        bar = "#" * max(0, int(d * 2000))
        print(f"  {name:16s} {d * 1e4:+6.1f} bps  {bar}")


if __name__ == "__main__":
    main()
