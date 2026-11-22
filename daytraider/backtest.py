"""Backtest on the held-out test period, per strategy (long and short).

Each fired setup is scored with its simulated ATR-bracket outcome in the
strategy's direction (take-profit / stop-loss / horizon exit — same rules
the live bot trades), with costs charged per trade.
"""
from __future__ import annotations

import numpy as np
import torch

import config
from data import load_dataset
from model import Scaler, ensemble_probs, load_ensemble, load_threshold
from strategies import STRATEGY_COLUMNS, STRATEGY_NAMES, is_short


def stats(rets: np.ndarray, mask: np.ndarray) -> dict:
    r = rets[mask] - config.COST_PER_TRADE
    if len(r) == 0:
        return {"trades": 0, "hit": float("nan"), "avg": float("nan"),
                "total": 0.0}
    return {"trades": len(r),
            "hit": float((r > 0).mean()),
            "avg": float(r.mean()),
            "total": float(np.exp(r.sum()) - 1)}


def report(label: str, s: dict):
    print(f"  {label:22s} trades={s['trades']:5d}  hit={s['hit']:.3f}  "
          f"avg={s['avg'] * 1e4:+.1f} bps  total={s['total']:+.2%}")


def main():
    print("Downloading data (same splits as training)...")
    X_test, y_test, ret_long, ret_short, setups = load_dataset()["test"]
    scaler = Scaler.load()
    models = load_ensemble()
    threshold = load_threshold()
    print(f"Ensemble of {len(models)} models")

    X = torch.from_numpy(scaler.transform(X_test))
    probs = torch.cat([ensemble_probs(models, X[i:i + 4096])
                       for i in range(0, len(X), 4096)]).numpy()

    print(f"\nCalibrated threshold (diagnostic): {threshold:.4f}")
    print(f"Test bars: {len(probs)}\n")

    # Per-trade return and ranking score in each strategy's own direction.
    # The NN predicts long-bracket success, so shorts rank by (1 - prob).
    print("Per strategy (ATR-bracket outcomes, costs charged):")
    dir_rets, dir_scores, any_active = {}, {}, np.zeros(len(probs), bool)
    for i, name in enumerate(STRATEGY_NAMES):
        fired = setups[:, i] > 0
        rets = ret_short if is_short(name) else ret_long
        score = (1 - probs) if is_short(name) else probs
        dir_rets[name], dir_scores[name] = rets, score
        tag = name if name in config.ACTIVE_STRATEGIES else f"{name} (off)"
        report(tag, stats(rets, fired))
        if name in config.ACTIVE_STRATEGIES:
            any_active |= fired

    print("\nNN ranking diagnostic (per active strategy, top vs bottom half"
          " of its own fired bars):")
    for i, name in enumerate(STRATEGY_NAMES):
        if name not in config.ACTIVE_STRATEGIES:
            continue
        fired = setups[:, i] > 0
        if fired.sum() < 20:
            continue
        med = np.median(dir_scores[name][fired])
        top = fired & (dir_scores[name] >= med)
        bot = fired & (dir_scores[name] < med)
        t, b = stats(dir_rets[name], top), stats(dir_rets[name], bot)
        print(f"  {name:12s} top {t['avg'] * 1e4:+.1f} bps "
              f"vs bottom {b['avg'] * 1e4:+.1f} bps")


if __name__ == "__main__":
    main()
