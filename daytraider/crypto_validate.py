"""Validate the crypto mean-reversion strategy before building the live bot.

Feasibility showed only the reversal setup (oversold-RSI bounce) has a raw
edge on crypto. This checks whether it holds up properly:
  1. trailing-stop exit vs fixed bracket (trailing was our big stock win)
  2. month-by-month robustness (not just a pooled average)
  3. at MAKER cost (limit orders, ~15 bps) since taker (~25 bps) kills it

Long-only (shorting crypto was a bloodbath in the feasibility test).

Usage: python crypto_validate.py
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

COINS = ["BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD", "LINK/USD",
         "AVAX/USD", "DOGE/USD", "UNI/USD", "AAVE/USD", "DOT/USD",
         "BCH/USD", "XRP/USD", "MKR/USD", "GRT/USD", "SUSHI/USD"]
MAKER = 15 / 1e4       # limit-order fee assumption
TP_A, SL_A, TRAIL_A, H = 1.5, 1.0, 1.0, 8


def rsi(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def prep(df):
    o = df.copy()
    c, h, lo = o["close"], o["high"], o["low"]
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()],
                   axis=1).max(axis=1)
    o["atr"] = tr.ewm(alpha=1/14, adjust=False).mean() / c
    r = rsi(c)
    o["reversal"] = ((r.shift(1) < 30) & (r > r.shift(1))).fillna(False)
    return o


def exits(entry, atr, hi, lo, cl):
    """fixed-bracket and trailing outcomes for a long over forward bars."""
    tp, sl = TP_A * atr, SL_A * atr
    fixed = cl[-1] / entry - 1
    for k in range(len(hi)):
        if (lo[k] - entry) / entry <= -sl:
            fixed = -sl; break
        if (hi[k] - entry) / entry >= tp:
            fixed = tp; break
    trail, best = cl[-1] / entry - 1, 0.0
    for k in range(len(hi)):
        favlo = (lo[k] - entry) / entry
        favhi = (hi[k] - entry) / entry
        stop = max(-sl, best - TRAIL_A * atr)
        if favlo <= stop:
            trail = stop; break
        best = max(best, favhi)
    return fixed, trail


def main():
    client = CryptoHistoricalDataClient()
    end = datetime.now(timezone.utc) - timedelta(minutes=30)
    req = CryptoBarsRequest(symbol_or_symbols=COINS,
                            timeframe=TimeFrame(1, TimeFrameUnit.Hour),
                            start=end - timedelta(days=1000), end=end)
    df = client.get_crypto_bars(req).df
    coins = df.index.get_level_values(0).unique().tolist()
    print(f"Loaded {len(coins)} coins\n")

    rows = []
    for coin in coins:
        o = prep(df.xs(coin))
        H_a, L_a = o["high"].to_numpy(), o["low"].to_numpy()
        C_a, A_a = o["close"].to_numpy(), o["atr"].to_numpy()
        rev = o["reversal"].to_numpy()
        idx = o.index
        for i in range(len(o) - H):
            if not rev[i] or np.isnan(A_a[i]):
                continue
            fx, tr = exits(C_a[i], A_a[i], H_a[i+1:i+1+H], L_a[i+1:i+1+H],
                           C_a[i+1:i+1+H])
            rows.append((idx[i], coin, fx - MAKER, tr - MAKER))
    d = pd.DataFrame(rows, columns=["ts", "coin", "fixed", "trail"])
    print(f"{len(d)} reversal trades (maker cost {MAKER*1e4:.0f} bps)\n")

    for col in ("fixed", "trail"):
        r = d[col].values
        print(f"  {col:6s} avg={r.mean()*1e4:+.1f} bps  hit={(r>0).mean():.3f}"
              f"  total_sum={r.sum():+.2f}")

    # monthly robustness on the better exit
    best = "trail" if d["trail"].mean() > d["fixed"].mean() else "fixed"
    d["month"] = d["ts"].dt.to_period("M")
    m = d.groupby("month")[best].agg(["count", "mean", "sum"])
    m["mean_bps"] = m["mean"] * 1e4
    print(f"\nMonthly ({best} exit): {(m['sum']>0).mean():.0%} of months positive")
    print(m[["count", "mean_bps", "sum"]].tail(15).to_string())


if __name__ == "__main__":
    main()
