"""Head-to-head: can the neural network BE the whole toolbox?

Two ways to decide trades on the same held-out test data:

  RULES:   the 16 hand-written setups fire entries; the NN only ranks.
  PURE-ML: no setups at all. Enter long on every bar where the ensemble's
           up-probability is in the top band, short where it's in the
           bottom band. The network alone chooses everything.

Both are scored with the identical ATR-bracket returns and trading costs,
so the comparison is apples-to-apples. This answers whether the model
carries the edge on its own or needs the rules.

Usage: python pure_ml.py
"""
from __future__ import annotations

import numpy as np
import torch

import config
from data import load_dataset
from model import Scaler, ensemble_probs, load_ensemble
from strategies import STRATEGY_NAMES, is_short


def summarize(label, rets):
    rets = np.asarray(rets)
    if len(rets) == 0:
        print(f"  {label:32s} no trades")
        return
    print(f"  {label:32s} trades={len(rets):6d}  "
          f"avg={rets.mean() * 1e4:+.1f} bps  hit={(rets > 0).mean():.3f}  "
          f"sum={rets.sum():+.2f}")


def main():
    ds = load_dataset()
    X, y, ret_long, ret_short, setups = ds["test"]
    scaler = Scaler.load()
    models = load_ensemble()

    def predict(arr):
        t = torch.from_numpy(scaler.transform(arr))
        return torch.cat([ensemble_probs(models, t[i:i + 4096])
                          for i in range(0, len(t), 4096)]).numpy()
    probs = predict(X)
    # Band cut-offs come from VALIDATION predictions - percentiles of the
    # test set's own predictions would peek at the test distribution.
    val_probs = predict(ds["val"][0])
    cost = config.COST_PER_TRADE

    print(f"Test bars: {len(probs)}\n")

    print("=== RULES: hand-written setups fire entries ===")
    rule_rets = []
    for i, name in enumerate(STRATEGY_NAMES):
        if name not in config.ACTIVE_STRATEGIES:
            continue
        fired = setups[:, i] > 0
        r = (ret_short if is_short(name) else ret_long)[fired] - cost
        rule_rets.append(r)
    rule_rets = np.concatenate(rule_rets)
    summarize("all active setups", rule_rets)

    print("\n=== PURE-ML: the network alone picks entries ===")
    # Sweep symmetric probability bands. Longs where prob is highest,
    # shorts where it is lowest; the width controls selectivity.
    for pct in (30, 20, 10, 5):
        hi = np.percentile(val_probs, 100 - pct)
        lo = np.percentile(val_probs, pct)
        long_r = ret_long[probs >= hi] - cost
        short_r = ret_short[probs <= lo] - cost
        both = np.concatenate([long_r, short_r])
        summarize(f"top/bottom {pct}%  (long+short)", both)

    print("\n=== PURE-ML restricted to its most confident longs only ===")
    for pct in (10, 5, 2):
        hi = np.percentile(val_probs, 100 - pct)
        summarize(f"top {pct}% prob -> long", ret_long[probs >= hi] - cost)

    # Reference: how often would each even be tradable per day?
    n_days = len(probs) / (40 * 7)  # ~40 tickers x 7 bars/day
    print(f"\n(~{n_days:.0f} trading days in test; "
          f"rules fired {len(rule_rets)} trades, "
          f"~{len(rule_rets) / n_days:.0f}/day)")


if __name__ == "__main__":
    main()
