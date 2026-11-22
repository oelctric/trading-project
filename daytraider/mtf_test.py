"""Multi-timeframe alignment: the core discretionary day-trading technique.

Does a trade do better when the DAILY, 4-HOUR, and 1-HOUR trends all agree
with its direction (top-down alignment) vs when they conflict? Tests it
out-of-sample on BOTH unseen sets. If aligned trades clearly outperform,
multi-timeframe confirmation is the missing ingredient and we add it as a
filter/feature.

Usage: python mtf_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from data import compute_features, download_bars
from diagnose import OOS_UNIVERSE, OOS2_UNIVERSE
from strategies import STRATEGY_NAMES, is_short

COST = 5 / 1e4


def trend_sign(close, window):
    return np.sign(close / close.rolling(window).mean() - 1)


def collect(universe):
    rows = []
    for t in universe:
        try:
            f = compute_features(download_bars(t))
        except Exception as e:
            print(f"  skip {t}: {e}"); continue
        c = f["close"]
        # Timeframe proxies via nested hourly SMAs (robust, no resampling):
        # short ~3 days, medium ~2 weeks (4H feel), long ~1 month (daily feel).
        t1 = trend_sign(c, 20)     # short / intraday
        t4 = trend_sign(c, 80)     # medium
        td = trend_sign(c, 200)    # long / higher timeframe
        aligned_up = (t1 > 0) & (t4 > 0) & (td > 0)
        aligned_dn = (t1 < 0) & (t4 < 0) & (td < 0)
        for name in STRATEGY_NAMES:
            if name not in config.ACTIVE_STRATEGIES:
                continue
            short = is_short(name)
            rcol = "trail_ret_short" if short else "trail_ret"
            fired = (f[f"setup_{name}"] > 0) & f[rcol].notna()
            algn = aligned_dn if short else aligned_up
            for al, r in zip(algn[fired], f.loc[fired, rcol]):
                rows.append((bool(al), r - COST))
    return pd.DataFrame(rows, columns=["aligned", "net"])


def main():
    for label, uni in [("OOS set #1", OOS_UNIVERSE),
                       ("OOS set #2", OOS2_UNIVERSE)]:
        print(f"\n=== {label} ({len(uni)} tickers) ===")
        d = collect(uni)
        for al in [True, False]:
            r = d.loc[d["aligned"] == al, "net"]
            tag = "3-TF ALIGNED" if al else "not aligned"
            print(f"  {tag:14s} n={len(r):7d}  avg={r.mean()*1e4:+6.1f} bps  "
                  f"hit={(r>0).mean():.3f}")
        a = d.loc[d["aligned"], "net"].mean() * 1e4
        n = d.loc[~d["aligned"], "net"].mean() * 1e4
        print(f"  --> alignment edge: {a - n:+.1f} bps "
              f"({'helps' if a > n else 'no help'})")


if __name__ == "__main__":
    main()
