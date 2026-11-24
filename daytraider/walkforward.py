"""Walk-forward evaluation of the regime gate — reconciled + risk-adjusted.

Two upgrades over the original:
  1. EXIT RECONCILIATION: the gate now scores TRAILING-STOP returns
     (trail_ret / trail_ret_short) — the exact exit the live bot trades —
     instead of the fixed-bracket returns it used before.
  2. RISK-ADJUSTED METRICS: races the raw-sum gate against profit-factor,
     mean/std, and expectancy+hit-rate gates, all on the trailing returns,
     reporting month-robustness and single-month concentration (the metric
     that exposes lumpy, outlier-driven "edges").

All gate decisions use only past data (trailing 90d for the metric, plus a
30d net>0 fast cut). Cost is a CLI arg so the winner can be stress-tested.

Usage: python walkforward.py [cost_bps]     (default 5; try 7 or 8)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config
from data import compute_features, download_bars
from strategies import STRATEGY_NAMES, is_short

MIN_TRADES = 20   # a strategy needs this many trailing trades to be judged


def collect_trades() -> pd.DataFrame:
    """Every fired setup as (ts, strategy, GROSS trailing return). Cost is
    applied later so it can be swept without re-downloading."""
    from diagnose import OOS_UNIVERSE, OOS2_UNIVERSE
    if "--oos2" in sys.argv:
        universe, label = OOS2_UNIVERSE, "OUT-OF-SAMPLE set #2"
    elif "--oos" in sys.argv:
        universe, label = OOS_UNIVERSE, "OUT-OF-SAMPLE set #1"
    else:
        universe, label = config.TICKERS, "in-sample"
    print(f"  universe: {label} ({len(universe)} tickers)")
    rows = []
    # Evaluate the strategies the live bot can actually trade (it trades
    # ACTIVE_STRATEGIES & gated-in); --all includes the disabled ones too.
    names = STRATEGY_NAMES if "--all" in sys.argv else \
        [s for s in STRATEGY_NAMES if s in config.ACTIVE_STRATEGIES]
    for ticker in universe:
        try:
            feats = compute_features(download_bars(ticker))
        except Exception as e:
            print(f"  skipping {ticker}: {e}")
            continue
        print(f"  {ticker}: {len(feats)} bars")
        for name in names:
            rcol = "trail_ret_short" if is_short(name) else "trail_ret"
            fired = feats[(feats[f"setup_{name}"] > 0) & feats[rcol].notna()]
            for ts, row in fired.iterrows():
                rows.append((ts, name, row[rcol]))
    return pd.DataFrame(rows, columns=["ts", "strategy", "gross"])


def passes(rets: np.ndarray, kind: str) -> bool:
    if len(rets) < MIN_TRADES:
        return False
    if kind == "sum":
        return rets.sum() > 0
    if kind == "pf":
        gp = rets[rets > 0].sum()
        gl = -rets[rets < 0].sum()
        return gp > gl * 1.2 if gl > 0 else gp > 0
    if kind == "musig":
        return rets.std() > 0 and rets.mean() / rets.std() > 0.1
    if kind == "exphit":
        return rets.sum() > 0 and (rets > 0).mean() > 0.40
    raise ValueError(kind)


def main():
    cost = (float(sys.argv[1]) if len(sys.argv) > 1 else 5.0) / 1e4
    print(f"Building trailing-return trade history (cost {cost*1e4:.0f} bps)...")
    trades = collect_trades()
    trades["net"] = trades["gross"] - cost
    trades["month"] = trades["ts"].dt.to_period("M")
    months = sorted(trades["month"].unique())
    lb90 = pd.Timedelta(days=config.REGIME_LOOKBACK_DAYS)
    lb30 = pd.Timedelta(days=config.REGIME_LOOKBACK_FAST_DAYS)
    print(f"{len(trades)} trailing-exit trades over {len(months)} months\n")

    # each gate = (90d metric) AND (30d net sum > 0 fast cut)
    GATES = {
        "raw-sum": "sum",
        "profit-factor>1.2": "pf",
        "mean/std>0.1 (live)": "musig",
        "expectancy+hit>40%": "exphit",
    }
    frames = {g: [] for g in GATES}
    month_ret = {g: [] for g in GATES}
    for m in months[3:]:
        start = m.to_timestamp().tz_localize("America/New_York")
        p90 = trades[(trades["ts"] < start) & (trades["ts"] >= start - lb90)]
        p30 = trades[(trades["ts"] < start) & (trades["ts"] >= start - lb30)]
        g90 = {s: g["net"].to_numpy() for s, g in p90.groupby("strategy")}
        s30 = p30.groupby("strategy")["net"].sum()
        cur = trades[trades["month"] == m]
        for gname, kind in GATES.items():
            keep = {s for s, r in g90.items()
                    if passes(r, kind) and s30.get(s, 0.0) > 0}
            gt = cur[cur["strategy"].isin(keep)]
            frames[gname].append(gt)
            month_ret[gname].append(gt["net"].sum())

    print(f"{'gate':22s} {'trades':>7s} {'avg bps':>8s} {'months+':>8s} "
          f"{'top-mo':>7s} {'total':>8s}")
    for gname in GATES:
        allg = pd.concat(frames[gname])
        mr = pd.Series(month_ret[gname])
        conc = mr[mr > 0].max() / mr[mr > 0].sum() if (mr > 0).any() else np.nan
        print(f"{gname:22s} {len(allg):7d} {allg['net'].mean()*1e4:+8.1f} "
              f"{(mr>0).mean():7.0%} {conc:6.0%} {mr.sum():+8.2f}")

    print("\nPer-strategy trailing-return edge (full period, net):")
    for name in sorted(trades["strategy"].unique(), key=STRATEGY_NAMES.index):
        g = trades[trades["strategy"] == name]["net"]
        print(f"  {name:14s} n={len(g):6d}  avg={g.mean()*1e4:+6.1f} bps  "
              f"hit={(g>0).mean():.3f}  sum={g.sum():+.2f}")


if __name__ == "__main__":
    main()
