"""Conviction-based position sizing: does it beat flat risk?

Currently every trade risks a flat RISK_PER_TRADE. This tests whether
scaling each trade's size by a conviction score improves the risk-adjusted
return (Sharpe-like: mean/std of per-trade P&L) without changing which
trades are taken.

Conviction = the NN ensemble's directional confidence for the trade
(prob for longs, 1-prob for shorts), which the ranking diagnostic showed
sorts winners from losers within setups. We compare, on the test set:

  flat      : every trade weight 1.0
  conviction: weight scales linearly with NN conviction percentile
  binary    : weight 1.5 for top-half conviction, 0.5 for bottom-half

All are normalized to the same average weight, so total risk is unchanged —
only the ALLOCATION across trades differs. Better = higher mean/std.

Usage: python sizing_test.py
"""
from __future__ import annotations

import numpy as np
import torch

import config
from data import load_dataset
from model import Scaler, ensemble_probs, load_ensemble
from strategies import STRATEGY_NAMES, is_short


def stats(label, rets, weights):
    w = weights / weights.mean()  # normalize to avg weight 1
    pnl = rets * w
    sharpe = pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(252 * 7)
    print(f"  {label:12s} avg={pnl.mean()*1e4:+.1f} bps  "
          f"std={pnl.std()*1e4:.0f} bps  sharpe={sharpe:+.2f}")


def main():
    X, y, ret_long, ret_short, setups = load_dataset()["test"]
    scaler = Scaler.load()
    models = load_ensemble()
    Xt = torch.from_numpy(scaler.transform(X))
    probs = torch.cat([ensemble_probs(models, Xt[i:i + 4096])
                       for i in range(0, len(Xt), 4096)]).numpy()

    rets, conv = [], []
    for i, name in enumerate(STRATEGY_NAMES):
        if name not in config.ACTIVE_STRATEGIES:
            continue
        m = setups[:, i] > 0
        if not m.any():
            continue
        r = (ret_short if is_short(name) else ret_long)[m] - config.COST_PER_TRADE
        c = (1 - probs[m]) if is_short(name) else probs[m]
        rets.append(r); conv.append(c)
    rets = np.concatenate(rets)
    conv = np.concatenate(conv)
    pct = conv.argsort().argsort() / len(conv)  # conviction percentile 0..1

    print(f"{len(rets)} test trades\n")
    stats("flat", rets, np.ones_like(rets))
    # linear: weight from 0.5x (low conviction) to 1.5x (high)
    stats("conviction", rets, 0.5 + pct)
    stats("binary", rets, np.where(pct >= 0.5, 1.5, 0.5))


if __name__ == "__main__":
    main()
