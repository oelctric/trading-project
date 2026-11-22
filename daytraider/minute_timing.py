"""Does minute-level ENTRY TIMING improve the hourly system?

The hourly system decides WHAT to trade. This tests using minute bars to
time the ENTRY: instead of entering at the signal bar's close, place a limit
a small pullback away and enter only if the next hour's minute bars trade
there. Better fills raise per-trade edge — but waiting can MISS trades that
run without pulling back, and some of those would have won.

Net verdict = fill improvement (bps saved on entries that fill) minus the
cost of missed winners. Positive => build the timing layer; negative =>
minutes don't help even for timing.

Scoped to a subset of tickers / recent months so the minute pull is fast.

Usage: python minute_timing.py [months] [pullback_bps]
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config
from data import _client, compute_features, download_bars
from strategies import STRATEGY_NAMES, is_short

SUBSET = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD", "META", "AMZN"]


def minute_bars(ticker: str, days: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc) - timedelta(minutes=16)
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=end - timedelta(days=days), end=end, adjustment="all")
    df = _client().get_stock_bars(req).df
    if df.empty:
        return df
    df = df.droplevel(0)
    df.index = df.index.tz_convert("America/New_York")
    return df.between_time("09:30", "16:00")[["high", "low", "close"]]


def main():
    months = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    pull = (float(sys.argv[2]) if len(sys.argv) > 2 else 8.0) / 1e4
    days = months * 31
    window_min = 30  # minutes to wait for the pullback fill

    imm_rets, timed_rets, saved_bps, missed = [], [], [], 0
    for ticker in SUBSET:
        try:
            feats = compute_features(download_bars(ticker, days=days + 120))
            mins = minute_bars(ticker, days)
        except Exception as e:
            print(f"  skip {ticker}: {e}"); continue
        if mins.empty:
            print(f"  skip {ticker}: no minute data"); continue
        cutoff = feats.index[-1] - pd.Timedelta(days=days)
        sig = feats[feats.index >= cutoff]
        n = 0
        for ts, row in sig.iterrows():
            fired = [s for s in STRATEGY_NAMES
                     if s in config.ACTIVE_STRATEGIES and row[f"setup_{s}"] > 0]
            if not fired:
                continue
            short = is_short(fired[0])
            base_ret = row["trade_ret_short"] if short else row["trade_ret"]
            if np.isnan(base_ret):
                continue
            close = row["close"]
            # minute bars in the hour AFTER the signal
            win = mins[(mins.index > ts)
                       & (mins.index <= ts + pd.Timedelta(minutes=window_min))]
            if win.empty:
                continue
            n += 1
            imm_rets.append(base_ret - config.COST_PER_TRADE)
            # timed entry: limit a pullback away; fill if touched
            if not short:
                limit = close * (1 - pull)
                filled = (win["low"] <= limit).any()
            else:
                limit = close * (1 + pull)
                filled = (win["high"] >= limit).any()
            if filled:
                # entered `pull` better -> add that to the trade return
                timed_rets.append(base_ret + pull - config.COST_PER_TRADE)
                saved_bps.append(pull * 1e4)
            else:
                missed += 1  # no fill: trade skipped entirely
        print(f"  {ticker}: {n} timed signals")

    imm = np.array(imm_rets)
    timed = np.array(timed_rets)
    total = len(imm)
    fill_rate = len(timed) / total if total else 0
    print(f"\nSignals: {total}   fill rate: {fill_rate:.1%}   "
          f"missed (no pullback): {missed}")
    print(f"Immediate entry (baseline):  avg {imm.mean()*1e4:+.1f} bps  "
          f"hit {(imm>0).mean():.3f}  sum {imm.sum():+.3f}")
    if len(timed):
        print(f"Timed entry (filled only):   avg {timed.mean()*1e4:+.1f} bps  "
              f"hit {(timed>0).mean():.3f}  sum {timed.sum():+.3f}")
    # Fair comparison: total P&L per SIGNAL (missed = 0), both scored the same
    print(f"\nP&L per signal (missed counts as 0):")
    print(f"  immediate: {imm.sum()/total*1e4:+.1f} bps/signal")
    print(f"  timed:     {timed.sum()/total*1e4:+.1f} bps/signal")
    print(f"\nVerdict: timed {'BEATS' if timed.sum() > imm.sum() else 'LOSES to'}"
          f" immediate by {(timed.sum()-imm.sum())*1e4:+.1f} bps total "
          f"over {total} signals")


if __name__ == "__main__":
    main()
