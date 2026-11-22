"""1-minute bot backtest with HFT-aware features (the honest attempt).

The 15-minute attempt lost because costs/noise/false-positives ate the edge.
This tests a purpose-built 1-minute design that attacks each of those:

  1. PASSIVE FILL (adverse selection): a long posts a limit at the close;
     it fills ONLY if the next bar's low trades STRICTLY below the limit
     (exact touch = no fill = worst-case queue position). This is the
     make-or-break test — a naive "fill on touch" backtest is a fantasy.
  2. RVOL VETO: a setup fires only if the current minute's volume is
     > 200% of that same clock-minute's trailing 10-day average.
  3. TIME-OF-DAY breakdown: edge is reported per 30-min bucket, so a
     ToD gate can bench setups outside their profitable window.
  4. MULTI-TIMEFRAME EXIT: entries are 1-minute, but the trailing stop
     uses the 15-minute ATR so normal 1-min chop doesn't hunt it.

Compares PASSIVE vs NAIVE fills so the adverse-selection cost is explicit.

Usage: python minute_bot_backtest.py [months] [cost_bps]
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config
from data import _client

SUBSET = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD"]
RVOL_MIN = 2.0
TRAIL_TF_MIN = 15          # trailing stop anchored to 15-min ATR
TRAIL_MULT = 1.0
HOLD_CAP_MIN = 60          # exit at market after this long


def minute_bars(ticker, days):
    end = datetime.now(timezone.utc) - timedelta(minutes=16)
    req = StockBarsRequest(symbol_or_symbols=ticker,
                           timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                           start=end - timedelta(days=days), end=end,
                           adjustment="all")
    df = _client().get_stock_bars(req).df
    if df.empty:
        return df
    df = df.droplevel(0)
    df.index = df.index.tz_convert("America/New_York")
    return df.between_time("09:30", "16:00")


def rsi(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def prep(df):
    o = df.copy()
    c = o["close"]
    o["rsi"] = rsi(c)
    o["reversal"] = (o["rsi"].shift(1) < 30) & (o["rsi"] > o["rsi"].shift(1))
    # RVOL vs same clock-minute trailing 10-day mean
    o["hm"] = o.index.strftime("%H:%M")
    o["rvol"] = o.groupby("hm")["volume"].transform(
        lambda v: v / v.rolling(10, min_periods=3).mean().shift(1))
    # 15-min ATR mapped onto 1-min bars (fractional)
    b15 = df.resample("15min").agg(high=("high", "max"), low=("low", "min"),
                                   close=("close", "last")).dropna()
    tr = pd.concat([b15["high"] - b15["low"],
                    (b15["high"] - b15["close"].shift()).abs(),
                    (b15["low"] - b15["close"].shift()).abs()], axis=1).max(axis=1)
    atr15 = (tr.ewm(alpha=1/14, adjust=False).mean() / b15["close"])
    o["atr15"] = atr15.reindex(o.index, method="ffill")
    o["bucket"] = (o.index.hour.astype(str).str.zfill(2) + ":" +
                   np.where(o.index.minute < 30, "00", "30"))
    return o


def trail_exit(entry, atr, hi, lo, cl, cap):
    """Trailing stop (multi-TF ATR) over forward 1-min bars, long side."""
    sl0 = TRAIL_MULT * atr
    best = 0.0
    for k in range(min(cap, len(hi))):
        favhi = (hi[k] - entry) / entry
        favlo = (lo[k] - entry) / entry
        stop = max(-sl0, best - TRAIL_MULT * atr)
        if favlo <= stop:
            return stop
        best = max(best, favhi)
    j = min(cap, len(cl)) - 1
    return cl[j] / entry - 1 if j >= 0 else 0.0


def main():
    months = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cost = (float(sys.argv[2]) if len(sys.argv) > 2 else 3.0) / 1e4
    days = months * 31

    naive, passive, buckets = [], [], {}
    for t in SUBSET:
        try:
            o = prep(minute_bars(t, days))
        except Exception as e:
            print(f"  skip {t}: {e}"); continue
        H = o["high"].to_numpy(); L = o["low"].to_numpy()
        C = o["close"].to_numpy(); A = o["atr15"].to_numpy()
        rev = o["reversal"].fillna(False).to_numpy()
        rv = o["rvol"].to_numpy()
        bkt = o["bucket"]
        n = len(o)
        fired = 0
        for i in range(n - 2):
            if not rev[i] or np.isnan(A[i]) or A[i] <= 0:
                continue
            if not (rv[i] > RVOL_MIN):          # RVOL veto
                continue
            fired += 1
            fwd_h, fwd_l, fwd_c = H[i+1:], L[i+1:], C[i+1:]
            # NAIVE: enter at close[i]
            naive.append(trail_exit(C[i], A[i], fwd_h, fwd_l, fwd_c,
                                    HOLD_CAP_MIN) - cost)
            # PASSIVE: limit at close[i]; fill only if next low STRICTLY below
            limit = C[i]
            if L[i+1] < limit:                  # strict — exact touch no fill
                pv = trail_exit(limit, A[i], H[i+2:], L[i+2:], C[i+2:],
                                HOLD_CAP_MIN) - cost
                passive.append(pv)
                buckets.setdefault(bkt.iloc[i], []).append(pv)
        print(f"  {t}: {fired} RVOL-passed reversal signals")

    naive = np.array(naive); passive = np.array(passive)
    print(f"\ncost {cost*1e4:.0f} bps, RVOL>{RVOL_MIN}x, 15-min-ATR trail\n")
    print(f"NAIVE fill (enter at close):   n={len(naive):5d}  "
          f"avg={naive.mean()*1e4:+.1f} bps  hit={(naive>0).mean():.3f}")
    if len(passive):
        fillrate = len(passive) / len(naive)
        print(f"PASSIVE fill (strict low<lim): n={len(passive):5d}  "
              f"avg={passive.mean()*1e4:+.1f} bps  hit={(passive>0).mean():.3f}"
              f"  (fill rate {fillrate:.0%})")
    print("\n=== PASSIVE edge by time-of-day bucket ===")
    for b in sorted(buckets):
        r = np.array(buckets[b])
        if len(r) >= 20:
            print(f"  {b}  n={len(r):4d}  avg={r.mean()*1e4:+6.1f} bps")


if __name__ == "__main__":
    main()
