"""Old vs new NN ensemble on the SAME held-out test set (session-capped labels).

    python compare_models.py [old_dir]

The bot uses the network to RANK fired setups, so what matters is whether a
higher probability means a better same-day trade. Reports, for each ensemble:
  - AUC of prob vs win/lose on all test windows and on setup-fired windows
  - average trade return of the top-20% ranked setup signals vs all of them
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

import config
from data import load_dataset
from model import Scaler, ensemble_probs, load_trained_model
from strategies import STRATEGY_NAMES, is_short


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def probs_for(model_dir: Path, X: np.ndarray) -> np.ndarray:
    scaler = Scaler.load(model_dir / "scaler.json")
    models = [load_trained_model(model_dir / f"model_seed{s}.pt") for s in config.ENSEMBLE_SEEDS
              if (model_dir / f"model_seed{s}.pt").exists()]
    out = []
    for i in range(0, len(X), 8192):
        out.append(ensemble_probs(models, torch.from_numpy(scaler.transform(X[i:i + 8192]))).numpy())
    return np.concatenate(out)


def report(name: str, p: np.ndarray, rl, rs, setups) -> None:
    longs = [i for i, s in enumerate(STRATEGY_NAMES) if not is_short(s)]
    shorts = [i for i, s in enumerate(STRATEGY_NAMES) if is_short(s)]
    fired_l = setups[:, longs].max(axis=1) > 0
    fired_s = setups[:, shorts].max(axis=1) > 0
    # candidate score exactly as the live bot ranks: long -> p, short -> 1 - p
    score = np.concatenate([p[fired_l], 1 - p[fired_s]])
    ret = np.concatenate([rl[fired_l], rs[fired_s]])
    top = score >= np.quantile(score, 0.8)
    print(f"{name:8s} AUC all-bars {auc(p, rl > 0):.4f} | setup signals: n={len(ret):,} "
          f"AUC {auc(score, ret > 0):.4f} | avg ret all {ret.mean() * 1e4:+.2f} bps, "
          f"top-20% ranked {ret[top].mean() * 1e4:+.2f} bps")


def main() -> None:
    old_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        config.MODEL_DIR / "backup_overnight_labels_2026-11-22"
    ds = load_dataset()
    X, _, rl, rs, setups = ds["test"]
    print(f"test windows: {len(X):,} (session-capped labels)\n")
    report("OLD", probs_for(old_dir, X), rl, rs, setups)
    report("NEW", probs_for(config.MODEL_DIR, X), rl, rs, setups)


if __name__ == "__main__":
    main()
