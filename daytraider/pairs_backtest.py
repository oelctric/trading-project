"""Cointegration pairs-trading backtest (market-neutral stat-arb).

The honest test of the "stop trading single stocks" idea. Market-neutral, so
it sidesteps the microstructure/latency wall — you capture a multi-sigma
spread convergence, not a micro-reversion, so there's room to absorb costs.

Discipline:
  1. SELECT pairs on the first 60% of history (Engle-Granger cointegration
     test, p < 0.05) and fit the hedge ratio there.
  2. TRADE the spread on the held-out last 40% ONLY (out-of-sample): enter
     when |z| > ENTRY_Z, exit when |z| < EXIT_Z or stop at STOP_Z.
  3. Charge realistic two-leg round-trip costs.

Candidate pairs are same-sector names (most likely to be cointegrated).

Usage: python pairs_backtest.py [cost_bps_per_leg_roundtrip]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from data import download_bars

# same-sector candidate pairs (plus the classic KO/PEP)
CANDIDATES = [
    ("JPM", "BAC"), ("GS", "JPM"), ("WFC", "BAC"), ("C", "WFC"),
    ("XOM", "CVX"), ("KO", "PEP"), ("HD", "COST"), ("WMT", "COST"),
    ("GOOGL", "META"), ("MSFT", "GOOGL"), ("AMD", "NVDA"), ("QCOM", "MU"),
    ("V", "MA"), ("CRM", "ORCL"), ("AVGO", "QCOM"), ("UNH", "LLY"),
]
ENTRY_Z, EXIT_Z, STOP_Z = 2.0, 0.5, 4.0
Z_WIN = 100   # rolling window (bars) for the spread z-score


def backtest_pair(a, b, cost):
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(df) < 500:
        return None
    n = len(df)
    i_split = int(n * 0.6)
    train = df.iloc[:i_split]

    # cointegration on TRAIN only
    _, pval, _ = coint(train["a"], train["b"])
    if pval > 0.05:
        return {"pval": pval, "trades": 0, "reason": "not cointegrated"}
    beta = np.polyfit(train["b"], train["a"], 1)[0]
    if beta <= 0:   # the live bot skips negative-beta pairs - measure what is traded
        return {"pval": pval, "trades": 0, "reason": "negative beta (not traded live)"}

    spread = df["a"] - beta * df["b"]
    mu = spread.rolling(Z_WIN).mean()
    sd = spread.rolling(Z_WIN).std()
    z = (spread - mu) / sd
    z_test = z.iloc[i_split:]
    sp_test = spread.iloc[i_split:]

    # trade the spread out-of-sample
    pos, entry_sp, notional, rets = 0, 0.0, 1.0, []
    for t in range(1, len(z_test)):
        zt = z_test.iloc[t]
        if pos == 0 and np.isfinite(zt):
            if abs(zt) > ENTRY_Z:
                pos, entry_sp = (-1 if zt > 0 else 1), sp_test.iloc[t]   # short / long spread
                # capital committed, measured at ENTRY; abs(beta) so a negative
                # hedge ratio can't make the notional negative and flip P&L signs
                notional = abs(df["a"].iloc[i_split + t]) + \
                    abs(beta) * abs(df["b"].iloc[i_split + t])
        elif pos != 0:
            exit_now = abs(zt) < EXIT_Z or abs(zt) > STOP_Z
            if exit_now:
                # normalize by the entry notional; charge 2-leg cost
                gross = pos * (sp_test.iloc[t] - entry_sp)
                rets.append(gross / notional - cost)
                pos = 0
    rets = np.array(rets)
    if len(rets) == 0:
        return {"pval": pval, "trades": 0, "reason": "no signals"}
    return {"pval": pval, "beta": beta, "trades": len(rets),
            "avg_bps": rets.mean() * 1e4, "hit": (rets > 0).mean(),
            "total": rets.sum()}


def main():
    cost = (float(sys.argv[1]) if len(sys.argv) > 1 else 10.0) / 1e4
    print(f"Cointegration pairs backtest (2-leg round-trip cost "
          f"{cost*1e4:.0f} bps, out-of-sample)\n")
    cache = {}

    def bars(t):
        if t not in cache:
            try:
                cache[t] = download_bars(t)["close"]
            except Exception as e:
                print(f"  {t}: download failed ({e})")
                cache[t] = None
        return cache[t]

    results = []
    for a_sym, b_sym in CANDIDATES:
        a, b = bars(a_sym), bars(b_sym)
        if a is None or b is None:
            continue
        r = backtest_pair(a, b, cost)
        if r is None:
            continue
        if r["trades"] == 0:
            print(f"  {a_sym}-{b_sym:5s} p={r['pval']:.3f}  {r['reason']}")
        else:
            print(f"  {a_sym}-{b_sym:5s} p={r['pval']:.3f}  "
                  f"trades={r['trades']:3d}  avg={r['avg_bps']:+6.1f} bps  "
                  f"hit={r['hit']:.2f}  total={r['total']:+.3f}")
            results.append((f"{a_sym}-{b_sym}", r))

    if results:
        allavg = np.mean([r["avg_bps"] for _, r in results])
        alltot = np.sum([r["total"] for _, r in results])
        pos = np.mean([r["avg_bps"] > 0 for _, r in results])
        print(f"\nCointegrated pairs traded: {len(results)}")
        print(f"  mean per-trade edge: {allavg:+.1f} bps")
        print(f"  pairs profitable:    {pos:.0%}")
        print(f"  summed total return: {alltot:+.3f}")


if __name__ == "__main__":
    main()
