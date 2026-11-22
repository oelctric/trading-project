"""INVENTION: match the exit style to the setup style.

We use a trailing stop (a trend-following, let-winners-run exit) on ALL
setups. But mean-reversion setups (reversal/pullback/gap-fade/dip-buyer)
don't run — they revert to a target and stall. Hypothesis: mean-reversion
setups do better with a FIXED TARGET exit (trade_ret), while trend setups
do better with TRAILING (trail_ret).

Tests per family, both exits, on BOTH unseen sets. If the pattern is
consistent, we route each setup to its best exit — a real, mechanism-driven
improvement.

Usage: python exit_by_setup.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from data import compute_features, download_bars
from diagnose import OOS_UNIVERSE, OOS2_UNIVERSE
from strategies import is_short

COST = 5 / 1e4
MEAN_REV = {"reversal", "pullback", "gap_fade", "disc_strongday_dip",
            "reversal_dn", "pullback_dn", "gap_fade_dn"}
TREND = {"momentum", "breakout", "orb", "gap_go", "disc_orb_wide",
         "momentum_dn", "breakdown", "orb_dn", "gap_go_dn"}


def collect(universe):
    rows = []
    for t in universe:
        try:
            f = compute_features(download_bars(t))
        except Exception as e:
            print(f"  skip {t}: {e}"); continue
        for name in config.ACTIVE_STRATEGIES:
            short = is_short(name)
            trail_col = "trail_ret_short" if short else "trail_ret"
            fixed_col = "trade_ret_short" if short else "trade_ret"
            m = (f[f"setup_{name}"] > 0) & f[trail_col].notna() \
                & f[fixed_col].notna()
            fam = "mean_rev" if name in MEAN_REV else "trend"
            for tr, fx in zip(f.loc[m, trail_col], f.loc[m, fixed_col]):
                rows.append((fam, tr - COST, fx - COST))
    return pd.DataFrame(rows, columns=["fam", "trail", "fixed"])


def main():
    for label, uni in [("OOS set #1", OOS_UNIVERSE),
                       ("OOS set #2", OOS2_UNIVERSE)]:
        print(f"\n=== {label} ({len(uni)} tickers) ===")
        d = collect(uni)
        print(f"  {'family':10s} {'n':>7s} {'TRAIL':>9s} {'FIXED':>9s}  best")
        for fam in ["trend", "mean_rev"]:
            g = d[d["fam"] == fam]
            tr = g["trail"].mean() * 1e4
            fx = g["fixed"].mean() * 1e4
            best = "TRAIL" if tr > fx else "FIXED"
            print(f"  {fam:10s} {len(g):7d} {tr:+8.1f} {fx:+8.1f}  "
                  f"{best} (+{abs(tr-fx):.1f})")


if __name__ == "__main__":
    main()
