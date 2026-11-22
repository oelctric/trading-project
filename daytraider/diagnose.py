"""Competence scorecard: which ASPECTS of day trading does the system
handle, and where does it bleed?

The NN doesn't "understand" anything — but the system's behavior can be
graded across the conditions a day trader faces. For every trade the bot
would take (active-strategy, trailing-exit outcome, net of cost), this tags
the market context and reports per-bucket edge. Negative buckets are blind
spots to act on (add a filter, disable the setup in that condition, etc.).

Dimensions:
  setup family | direction | volatility regime (VIX) | market regime (SPY
  trend) | trade-vs-market alignment | trend-vs-chop (efficiency ratio) |
  hour of day | gap state

Usage: python diagnose.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config
from data import compute_features, download_bars
from strategies import STRATEGY_NAMES, is_short

# ~50 liquid tickers NONE of which are in the trained 40-ticker universe.
# The setups' active-list, thresholds, discovered strategies, and NN were
# all fit on config.TICKERS, so these are genuinely out-of-sample for the
# WHOLE system — the honest generalization test.
OOS_UNIVERSE = [
    "GOOG", "TXN", "AMAT", "LRCX", "ADI", "NOW", "INTU", "PANW", "SNPS",
    "KLAC", "MRVL", "MS", "USB", "PNC", "SCHW", "AXP", "BLK", "SPGI",
    "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY", "AMGN", "GILD",
    "TGT", "LOW", "SBUX", "MCD", "NKE", "PG", "CL", "MO", "PM",
    "COP", "SLB", "EOG", "PSX", "GE", "HON", "MMM", "UPS", "FDX", "RTX",
    "LMT", "DE", "NEE", "DUK", "SO", "T", "VZ", "AMT", "PLD", "O",
]

# A SECOND unseen set — none in the trained 40 OR in OOS_UNIVERSE. Used to
# confirm whether a risk-adjusted gate robustly beats raw-sum across TWO
# independent unseen universes (vs winning on one by chance).
OOS2_UNIVERSE = [
    "TFC", "COF", "BK", "MET", "PRU", "ALL", "TRV", "CB", "MMC", "AIG",
    "MSI", "ADSK", "ANET", "DELL", "CRWD", "FTNT",
    "CVS", "CI", "ELV", "ZTS", "MDT", "SYK", "BDX", "ISRG",
    "ITW", "PH", "CMI", "NSC", "CSX", "UNP", "EMR", "ETN",
    "EL", "KMB", "GIS", "HSY", "KR", "DG", "DLTR", "ROST", "TJX", "YUM",
    "MDLZ", "KMI", "OKE", "HAL", "DVN", "FANG", "WMB", "VLO",
]

FAMILY = {"momentum": "momentum", "momentum_dn": "momentum",
          "breakout": "breakout", "breakdown": "breakout",
          "reversal": "reversal", "reversal_dn": "reversal",
          "pullback": "pullback", "pullback_dn": "pullback",
          "orb": "orb", "orb_dn": "orb", "gap_go": "gap", "gap_go_dn": "gap",
          "gap_fade": "gap", "gap_fade_dn": "gap",
          "disc_strongday_dip": "discovered", "disc_orb_wide": "discovered"}


def spy_context():
    spy = download_bars("SPY")
    c = spy["close"]
    trend = np.sign(c / c.rolling(20).mean() - 1)      # +1 up / -1 down
    ret5 = c.pct_change(5)
    return pd.DataFrame({"spy_trend": trend, "spy_ret5": ret5})


def efficiency_ratio(close, n=10):
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return (net / path).fillna(0)                       # 1=trend, 0=chop


def main():
    oos = "--oos" in sys.argv
    universe = OOS_UNIVERSE if oos else config.TICKERS
    print(f"{'OUT-OF-SAMPLE (unseen tickers)' if oos else 'IN-SAMPLE'} "
          f"universe: {len(universe)} tickers")
    spy = spy_context()
    rows = []
    for ticker in universe:
        try:
            f = compute_features(download_bars(ticker))
        except Exception as e:
            print(f"  skip {ticker}: {e}"); continue
        f = f.join(spy.reindex(f.index, method="ffill"))
        f["er"] = efficiency_ratio(f["close"])
        for name in STRATEGY_NAMES:
            if name not in config.ACTIVE_STRATEGIES:
                continue
            rcol = "trail_ret_short" if is_short(name) else "trail_ret"
            fired = f[(f[f"setup_{name}"] > 0) & f[rcol].notna()]
            short = is_short(name)
            for ts, r in fired.iterrows():
                net = r[rcol] - config.COST_PER_TRADE
                # trade direction: long=+1, short=-1
                tdir = -1 if short else 1
                aligned = "with" if tdir == r["spy_trend"] else "against"
                rows.append({
                    "ret": net,
                    "family": FAMILY.get(name, name),
                    "dir": "short" if short else "long",
                    "vol": ("hi_vol" if r["vix_z"] > 0.5 else
                            "lo_vol" if r["vix_z"] < -0.5 else "mid_vol"),
                    "mkt": "spy_up" if r["spy_trend"] > 0 else "spy_down",
                    "align": aligned,
                    "regime": "trend" if r["er"] > 0.4 else "chop",
                    "hour": f"{ts.hour:02d}h",
                    "gap": ("gap_up" if r["gap"] > 0.005 else
                            "gap_dn" if r["gap"] < -0.005 else "no_gap"),
                })
    d = pd.DataFrame(rows)
    print(f"\n{len(d)} active-strategy trades (trailing exit, net of cost)")
    print(f"OVERALL: avg {d['ret'].mean()*1e4:+.1f} bps  "
          f"hit {(d['ret']>0).mean():.3f}\n")

    for dim in ["family", "dir", "vol", "mkt", "align", "regime", "hour",
                "gap"]:
        print(f"=== by {dim} ===")
        g = d.groupby(dim)["ret"]
        tbl = pd.DataFrame({"n": g.count(), "avg_bps": g.mean() * 1e4,
                            "hit": (g.apply(lambda x: (x > 0).mean()))})
        tbl = tbl.sort_values("avg_bps", ascending=False)
        for idx, row in tbl.iterrows():
            flag = "  <-- BLIND SPOT" if row["avg_bps"] < -1 and row["n"] > 100 else ""
            print(f"  {str(idx):10s} n={row['n']:6.0f}  "
                  f"avg={row['avg_bps']:+6.1f} bps  hit={row['hit']:.3f}{flag}")
        print()


if __name__ == "__main__":
    main()
