"""Exit management: does anything beat our static ATR bracket?

Re-simulates every setup-fired trade bar-by-bar over the forward hourly path
under four exit styles, direction-aware, with costs:

  fixed    : current — TP +1.5 ATR, SL -1.0 ATR, horizon 8 bars
  trail    : SL -1.0 ATR, then trail 1.0 ATR behind the best price reached
  partial  : take half at +0.75 ATR, move stop to breakeven, rest runs to
             +1.5 ATR / breakeven / horizon
  wide_run : SL -1.0 ATR, TP +2.5 ATR (let winners run further)

Reports avg bps/trade per style across all active-strategy trades, and on
the profitable strategies the gate keeps (the ones that matter live).

Usage: python exit_sim.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config
from data import compute_features, download_bars
from strategies import STRATEGY_NAMES, is_short

H = config.LABEL_HORIZON
PROFITABLE = ["momentum", "orb", "disc_orb_wide", "gap_fade", "breakout",
              "gap_go", "pullback"]


def sim_trade(entry, atr, highs, lows, closes, short):
    """Return net fractional P&L for each exit style for one trade.
    highs/lows/closes are the forward H bars after entry."""
    tp_d, sl_d = config.TP_ATR_MULT * atr, config.SL_ATR_MULT * atr
    sgn = -1.0 if short else 1.0

    def favor(px):   # signed move in trade's favor
        return sgn * (px - entry) / entry

    # fixed
    fixed = closes[-1] / entry - 1.0
    fixed *= sgn
    for k in range(len(highs)):
        hi_fav = favor(highs[k] if not short else lows[k])
        lo_fav = favor(lows[k] if not short else highs[k])
        if lo_fav <= -sl_d:
            fixed = -sl_d; break
        if hi_fav >= tp_d:
            fixed = tp_d; break

    # wide_run (TP 2.5 ATR)
    wide = closes[-1] / entry - 1.0
    wide *= sgn
    for k in range(len(highs)):
        hi_fav = favor(highs[k] if not short else lows[k])
        lo_fav = favor(lows[k] if not short else highs[k])
        if lo_fav <= -sl_d:
            wide = -sl_d; break
        if hi_fav >= 2.5 * atr:
            wide = 2.5 * atr; break

    # trail (1.0 ATR behind best)
    trail = closes[-1] / entry - 1.0
    trail *= sgn
    best = 0.0
    for k in range(len(highs)):
        hi_fav = favor(highs[k] if not short else lows[k])
        lo_fav = favor(lows[k] if not short else highs[k])
        stop = max(-sl_d, best - atr)  # trailing stop level (in favor units)
        if lo_fav <= stop:
            trail = stop; break
        best = max(best, hi_fav)
    else:
        trail = favor(closes[-1])

    # partial: half at +0.75 ATR then breakeven on rest
    part_tp = 0.75 * atr
    took_half = False
    partial = closes[-1] / entry - 1.0
    partial *= sgn
    for k in range(len(highs)):
        hi_fav = favor(highs[k] if not short else lows[k])
        lo_fav = favor(lows[k] if not short else highs[k])
        if not took_half:
            if lo_fav <= -sl_d:
                partial = -sl_d; break
            if hi_fav >= part_tp:
                took_half = True  # bank half, stop to breakeven
        else:
            if lo_fav <= 0.0:  # breakeven stop on remainder
                partial = 0.5 * part_tp + 0.5 * 0.0; break
            if hi_fav >= tp_d:
                partial = 0.5 * part_tp + 0.5 * tp_d; break
    else:
        if took_half:
            partial = 0.5 * part_tp + 0.5 * favor(closes[-1])
    return {"fixed": fixed, "trail": trail, "partial": partial,
            "wide_run": wide}


def main():
    styles = ["fixed", "trail", "partial", "wide_run"]
    acc = {s: [] for s in styles}
    acc_prof = {s: [] for s in styles}
    test_only = "test" in sys.argv
    for ticker in config.TICKERS:
        try:
            f = compute_features(download_bars(ticker))
        except Exception as e:
            print(f"  skip {ticker}: {e}"); continue
        if test_only:  # out-of-sample: last 15% of each ticker's history
            f = f.iloc[int(len(f) * 0.85):]
        H_arr = f["high"].to_numpy(); L = f["low"].to_numpy()
        C = f["close"].to_numpy(); A = f["atr_pct"].to_numpy()
        fired_cols = {s: f[f"setup_{s}"].to_numpy() for s in STRATEGY_NAMES
                      if s in config.ACTIVE_STRATEGIES}
        print(f"  {ticker}: {len(f)} bars")
        for i in range(len(f) - H):
            hit = [s for s, col in fired_cols.items() if col[i] > 0]
            if not hit:
                continue
            short = is_short(hit[0])
            res = sim_trade(C[i], A[i], H_arr[i+1:i+1+H], L[i+1:i+1+H],
                            C[i+1:i+1+H], short)
            for s in styles:
                acc[s].append(res[s] - config.COST_PER_TRADE)
                if any(h in PROFITABLE for h in hit):
                    acc_prof[s].append(res[s] - config.COST_PER_TRADE)

    print("\n=== Exit style comparison (all active-strategy trades) ===")
    for s in styles:
        r = np.array(acc[s])
        print(f"  {s:9s} trades={len(r):6d}  avg={r.mean()*1e4:+.1f} bps  "
              f"hit={(r>0).mean():.3f}  sum={r.sum():+.2f}")
    print("\n=== On profitable strategies only (what the gate keeps) ===")
    for s in styles:
        r = np.array(acc_prof[s])
        print(f"  {s:9s} trades={len(r):6d}  avg={r.mean()*1e4:+.1f} bps  "
              f"hit={(r>0).mean():.3f}  sum={r.sum():+.2f}")


if __name__ == "__main__":
    main()
