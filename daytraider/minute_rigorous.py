"""Rigorous minute-level entry-timing test (no circular shortcut).

The earlier minute_timing.py was an artifact: it just added the pullback
buffer to the hourly outcome. This version simulates the ACTUAL trade
outcome at minute resolution from each entry point:

  immediate : enter at the hourly signal close; simulate the bracket forward
              over real minute bars until TP / SL / end-of-day.
  timed     : place a limit a pullback below (long); if a minute bar trades
              there within the entry window, enter at that price and
              simulate the bracket forward from there. If it never fills,
              the trade is MISSED (counts as 0).

Both use the identical minute-level bracket simulation, so the comparison
is honest — it captures pullbacks that keep going and stop you out, and the
winners you miss by waiting. P&L per signal (missed = 0) is the verdict.

Scoped to a subset of tickers / recent months so the minute pull is fast.

Usage: python minute_rigorous.py [months] [pullback_bps]
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
ENTRY_WINDOW_MIN = 30


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


def bracket_outcome(entry, atr, fwd, short):
    """Simulate a fixed bracket from `entry` over forward minute bars `fwd`
    (until TP / SL / end of that day's bars). Returns fractional P&L."""
    tp_d, sl_d = config.TP_ATR_MULT * atr, config.SL_ATR_MULT * atr
    if not short:
        tp, sl = entry * (1 + tp_d), entry * (1 - sl_d)
        for hi, lo in zip(fwd["high"].values, fwd["low"].values):
            if lo <= sl:
                return -sl_d
            if hi >= tp:
                return tp_d
        return fwd["close"].values[-1] / entry - 1
    else:
        tp, sl = entry * (1 - tp_d), entry * (1 + sl_d)
        for hi, lo in zip(fwd["high"].values, fwd["low"].values):
            if hi >= sl:
                return -sl_d
            if lo <= tp:
                return tp_d
        return 1 - fwd["close"].values[-1] / entry


def main():
    months = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    pull = (float(sys.argv[2]) if len(sys.argv) > 2 else 8.0) / 1e4
    days = months * 31
    cost = config.COST_PER_TRADE

    imm, timed, missed, total = [], [], 0, 0
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
            atr = row["atr_pct"]
            close = row["close"]
            day_end = ts.normalize() + pd.Timedelta(hours=16)
            # hourly bars are labelled by their START: the signal only exists
            # once the bar CLOSES, so trading may begin at ts + bar length
            # (starting at ts used minutes from inside the signal bar itself)
            t0 = ts + pd.Timedelta(minutes=config.BAR_MINUTES)
            entry_win = mins[(mins.index >= t0)
                             & (mins.index < t0 + pd.Timedelta(minutes=ENTRY_WINDOW_MIN))]
            fwd_all = mins[(mins.index >= t0) & (mins.index <= day_end)]
            if entry_win.empty or fwd_all.empty:
                continue
            total += 1; n += 1
            # immediate: enter at hourly close, simulate forward
            imm.append(bracket_outcome(close, atr, fwd_all, short) - cost)
            # timed: limit a pullback away within the entry window
            if not short:
                limit = close * (1 - pull)
                hit = entry_win.index[entry_win["low"] <= limit]
            else:
                limit = close * (1 + pull)
                hit = entry_win.index[entry_win["high"] >= limit]
            if len(hit):
                fill_ts = hit[0]
                fwd = mins[(mins.index > fill_ts) & (mins.index <= day_end)]
                if len(fwd):
                    timed.append(bracket_outcome(limit, atr, fwd, short) - cost)
                else:
                    timed.append(0.0)
            else:
                missed += 1  # never filled -> trade skipped
        print(f"  {ticker}: {n} signals")

    imm = np.array(imm); timed = np.array(timed)
    print(f"\nSignals: {total}   fills: {len(timed)} ({len(timed)/total:.0%})"
          f"   missed: {missed}")
    print(f"Immediate:  avg {imm.mean()*1e4:+.1f} bps  hit {(imm>0).mean():.3f}"
          f"  sum {imm.sum():+.3f}")
    print(f"Timed(fill):avg {timed.mean()*1e4:+.1f} bps  hit {(timed>0).mean():.3f}"
          f"  sum {timed.sum():+.3f}")
    print("\nP&L per signal (missed = 0):")
    print(f"  immediate: {imm.sum()/total*1e4:+.1f} bps/signal")
    print(f"  timed:     {timed.sum()/total*1e4:+.1f} bps/signal")
    d = (timed.sum() - imm.sum()) / total * 1e4
    print(f"\nVerdict: timed entry {'BEATS' if d > 0 else 'LOSES to'} immediate "
          f"by {d:+.1f} bps/signal")


if __name__ == "__main__":
    main()
