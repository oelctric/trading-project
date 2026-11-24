"""Optimize the trailing-stop multiplier — validated on TWO unseen sets.

Trailing exits were the biggest single improvement, but the 1.0-ATR trail
was never tuned. This sweeps the trail multiplier and reports the sharpe-
gated edge on both out-of-sample universes. Only a value that generalizes
on BOTH is trustworthy (the expectancy+hit fluke taught us one set lies).

Usage: python trail_sweep.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from data import _trailing_returns, compute_features, download_bars
from diagnose import OOS_UNIVERSE, OOS2_UNIVERSE
from strategies import STRATEGY_NAMES, is_short

MULTS = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
H = config.LABEL_HORIZON
lb90 = pd.Timedelta(days=config.REGIME_LOOKBACK_DAYS)
lb30 = pd.Timedelta(days=config.REGIME_LOOKBACK_FAST_DAYS)


def sharpe_gate_edge(trades):
    """Sharpe-gated walk-forward avg bps (mean/std>0.10 + 30d sum>0)."""
    trades = trades.sort_values("ts")
    months = sorted(trades["ts"].dt.to_period("M").unique())
    trades["m"] = trades["ts"].dt.to_period("M")
    kept = []
    for mo in months[3:]:
        start = mo.to_timestamp().tz_localize("America/New_York")
        p90 = trades[(trades["ts"] < start) & (trades["ts"] >= start - lb90)]
        p30 = trades[(trades["ts"] < start) & (trades["ts"] >= start - lb30)]
        s30 = p30.groupby("strategy")["net"].sum()
        keep = set()
        for s, g in p90.groupby("strategy"):
            r = g["net"].to_numpy()
            if len(r) < 20 or s30.get(s, 0) <= 0:
                continue
            if r.std() > 0 and r.mean() / r.std() > 0.10:
                keep.add(s)
        cur = trades[trades["m"] == mo]
        kept.append(cur[cur["strategy"].isin(keep)])
    g = pd.concat(kept)
    return g["net"].mean() * 1e4, len(g)


def collect(universe, cost=5/1e4):
    """Per-ticker cached arrays -> trades per trail multiplier."""
    per_mult = {m: [] for m in MULTS}
    for t in universe:
        try:
            f = compute_features(download_bars(t))
        except Exception as e:
            print(f"  skip {t}: {e}"); continue
        hi, lo = f["high"].to_numpy(), f["low"].to_numpy()
        cl, atr = f["close"].to_numpy(), f["atr_pct"].to_numpy()
        setups = {s: f[f"setup_{s}"].to_numpy() for s in STRATEGY_NAMES
                  if s in config.ACTIVE_STRATEGIES}
        idx = f.index
        for m in MULTS:
            # session-capped like the live bot (it flattens before every close)
            rl, rs = _trailing_returns(hi, lo, cl, atr, H, m,
                                       config.SL_ATR_MULT, config.TP_BACKSTOP_ATR,
                                       session=np.asarray(f.index.date))
            for name, col in setups.items():
                r = rs if is_short(name) else rl
                mask = (col > 0) & ~np.isnan(r)
                for k in np.where(mask)[0]:
                    per_mult[m].append((idx[k], name, r[k] - cost))
    return {m: pd.DataFrame(rows, columns=["ts", "strategy", "net"])
            for m, rows in per_mult.items()}


def main():
    for label, uni in [("OOS set #1", OOS_UNIVERSE),
                       ("OOS set #2", OOS2_UNIVERSE)]:
        print(f"\n=== {label} ({len(uni)} tickers) ===")
        data = collect(uni)
        print(f"  {'trail x ATR':12s} {'gated avg':>10s} {'trades':>8s}")
        for m in MULTS:
            avg, n = sharpe_gate_edge(data[m])
            star = "  <- current" if m == config.TRAIL_ATR_MULT else ""
            print(f"  {m:<12.2f} {avg:+8.1f} bps {n:8d}{star}")


if __name__ == "__main__":
    main()
