"""Retrain the swing strategy's settings (its "weights") and audit them honestly.

    python retrain.py            # show what retraining would pick + walk-forward history
    python retrain.py --apply    # also write the new settings to swing_params.json
    python retrain.py --quality  # also retrain + print the trade-quality model's weights

How it retrains: every candidate setting is simulated on all coins; the winner is
the one with the best MEDIAN per-coin profit (in R) on trades that have already
closed - so one lucky coin can't drive the choice. The walk-forward history shows
what this procedure would have picked at the start of each past year, and how that
pick did in the year after (data it had not seen).

Findings when this was written (Nov 2026):
  - re-picking every year and switching blindly was WORSE than fixed settings
    (2019-now: $100 -> $262 vs $419) - the yearly winner jumps around = noise
  - but the 2024, 2025 and 2026 retrains all chose the same settings
    (stop 2.0 ATR, trail 4.0 ATR, stall 10 bars), and the Jan-2024 pick beat the
    old settings on the unseen 2024-now data ($175 vs $156, drawdown 37% vs 32%)
    -> adopted. Only --apply when the pick is stable across several retrains.
  - the trade-quality model (logistic regression on entry features) finds real but
    weak structure (AUC ~0.56) and did NOT improve returns when used to skip or
    shrink trades, so it is reported but not used live.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

import backtest_swing as bt
import config
import swing_live
from history import load
from swing import FEATURES

GRID = dict(breakout=[10, 20, 30, 50], slow=[50, 100, 200], init_stop_atr=[2.0, 2.5, 3.0],
            trail_atr=[2.5, 3.0, 4.0], stall_bars=[10, 15, 30])


def score(trades: list[dict]) -> float:
    if len(trades) < 30:
        return -1e9
    return float(pd.DataFrame(trades).groupby("symbol")["r"].sum().median())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--quality", action="store_true")
    args = ap.parse_args()

    frames = {s: load(s, "1d", 3650) for s in bt.SYMBOLS}
    lead = frames.get("BTC-USD")
    fee = config.FUTURES_FEE_PER_SIDE if config.USE_LEVERAGE else config.FEE_PER_SIDE["crypto"]
    combos = list(itertools.product(*GRID.values()))
    print(f"simulating {len(combos)} candidate settings on {len(frames)} coins...")
    trades = {}
    for v in combos:
        p = replace(swing_live.PARAMS, **dict(zip(GRID, v)))
        trades[v] = [dict(t, symbol=s) for s, df in frames.items()
                     for t in bt.simulate(df, p, fee=fee, leader=lead)]

    current = tuple(getattr(swing_live.PARAMS, k) for k in GRID)
    print("\nwalk-forward: what retraining picked at the start of each year -> next-year result")
    print(f"{'year':<6}{'picked (breakout, slow, stop, trail, stall)':<46}{'next-yr R':>10}{'current R':>11}")
    for year in range(2019, pd.Timestamp.now().year + 1):
        y0, y1 = pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        best = max(combos, key=lambda v: score([t for t in trades[v] if t["close_time"] < y0]))
        nxt = sum(t["r"] for t in trades[best] if y0 <= t["open_time"] < y1)
        cur = sum(t["r"] for t in trades[current] if y0 <= t["open_time"] < y1)
        print(f"{year:<6}{str(best):<46}{nxt:>+10.1f}{cur:>+11.1f}")

    best = max(combos, key=lambda v: score(trades[v]))
    print(f"\nretrained on ALL data: {dict(zip(GRID, best))}")
    print(f"current live settings:  {dict(zip(GRID, current))}")
    # only trust a pick the walk-forward retrains have been converging on: it must
    # match (ignoring stall_bars) the picks made at the start of the last 2 years
    recent = []
    for year in (pd.Timestamp.now().year - 1, pd.Timestamp.now().year):
        y0 = pd.Timestamp(f"{year}-01-01", tz="UTC")
        recent.append(max(combos, key=lambda v: score([t for t in trades[v] if t["close_time"] < y0])))
    stable = all(r[:4] == best[:4] for r in recent)
    print(f"stable across the last 2 yearly retrains: {'yes' if stable else 'NO'} {recent}")
    if args.apply and not stable:
        print("not applying: an unstable pick is usually noise (see the walk-forward table)")
    if args.apply and stable and best != current:
        path = swing_live.PARAMS_FILE
        blob = json.load(open(path)) if os.path.exists(path) else {"tf": "1d", "params": {}}
        blob["params"].update(dict(zip(GRID, best)))
        blob["note"] = f"retrained {pd.Timestamp.now():%Y-%m-%d} by retrain.py"
        with open(path, "w") as f:
            json.dump(blob, f, indent=1)
        print(f"wrote {os.path.basename(path)}")

    if args.quality:
        quality_report(frames, lead, fee)


def quality_report(frames: dict, lead, fee: float) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    warnings.filterwarnings("ignore")
    T = [t for s, df in frames.items() for t in bt.simulate(df, swing_live.PARAMS, fee=fee, leader=lead)]
    X = pd.DataFrame([t["features"] for t in T])[FEATURES]
    y = np.array([t["r"] > 0 for t in T])
    ot = pd.Series([t["open_time"] for t in T])
    ct = pd.Series([t["close_time"] for t in T])
    p = np.full(len(T), np.nan)
    for year in range(2019, pd.Timestamp.now().year + 1):
        y0, y1 = pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        tr, te = (ct < y0).values, ((ot >= y0) & (ot < y1)).values
        if tr.sum() >= 40 and te.sum():
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=1000)).fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(p)
    print(f"\ntrade-quality model: walk-forward AUC {roc_auc_score(y[ok], p[ok]):.3f} "
          f"(0.5 = coin flip) on {ok.sum()} unseen trades")
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=1000)).fit(X, y)
    print("weights (standardized; + = more likely to win):")
    for f, w in sorted(zip(FEATURES, m[-1].coef_[0]), key=lambda z: -abs(z[1])):
        print(f"  {f:16}{w:+.3f}")


if __name__ == "__main__":
    main()
