"""Live market-neutral cointegration pairs bot (Alpaca paper).

Standalone strategy — does NOT share logic with the hourly directional bot.
Validated by pairs_walkforward.py: +86.8 bps/trade out-of-sample, 93% of
periods positive, 22% top-period concentration on the focused universe, with
Bonferroni-corrected pair selection (no false-positive harvesting).

HONEST STATUS: positive out-of-sample but LOW-FREQUENCY (~26-42 trades/yr)
and small-sample, so magnitude is uncertain. It is market-neutral (long one
leg, short beta x the other), so it sidesteps the microstructure/latency
wall. Paper only until it builds a live record.

Mechanics:
  * every REFRESH_HOURS, re-select cointegrated pairs from a trailing window
    (correlation pre-filter + Bonferroni-corrected Engle-Granger). Pairs that
    stop being cointegrated simply aren't reselected = the decoupling monitor.
  * each hourly scan: enter a pair's spread at |z|>2 (long the underperformer,
    short the outperformer), exit at |z|<0.5 or stop at |z|>4.
  * dollar-neutral sizing, max MAX_PAIRS concurrent, daily loss halt.

Usage: python pairs_trade.py            (loop)
       python pairs_trade.py --once     (single pass)
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

import config
from data import download_bars
from pairs_walkforward import (ENTRY_Z, EXIT_Z, STOP_Z, UNIVERSE, Z_WIN,
                               select_pairs, SELECT_BARS)

MAX_PAIRS = 4
PAIR_NOTIONAL_FRAC = 0.10     # ~10% of equity per leg (both legs ~ market-neutral)
REFRESH_HOURS = 24
DAILY_LOSS_LIMIT = 0.02
SCAN_SECONDS = 3600


def get_client() -> TradingClient:
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        sys.exit("Missing Alpaca keys in .env")
    return TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
                         paper=True)


def load_prices(tickers, days=90) -> pd.DataFrame:
    cols = {}
    for t in tickers:
        try:
            cols[t] = download_bars(t, days=days)["close"]
        except Exception as e:
            print(f"  {t}: price load failed ({e})")
    return pd.DataFrame(cols).dropna()


def zscore(spread: pd.Series) -> float:
    m = spread.rolling(Z_WIN, min_periods=20).mean().iloc[-1]
    s = spread.rolling(Z_WIN, min_periods=20).std().iloc[-1]
    return float((spread.iloc[-1] - m) / s) if s and np.isfinite(s) else np.nan


def pair_key(a, b):
    return f"{a}|{b}"


def main():
    once = "--once" in sys.argv
    client = get_client()
    print(f"Pairs bot connected -- equity "
          f"${float(client.get_account().equity):,.2f}  "
          f"(market-neutral, low-frequency, paper experiment)")

    selected: list[tuple] = []      # (a, b, beta)
    open_pairs: dict[str, dict] = {}
    last_refresh = 0.0

    while True:
      try:
        now = time.time()
        # --- periodic Bonferroni-corrected re-selection (decoupling monitor) ---
        if now - last_refresh > REFRESH_HOURS * 3600 or not selected:
            px = load_prices(UNIVERSE, days=160)  # need > SELECT_BARS(500) bars
            if len(px) > SELECT_BARS:
                selected = select_pairs(px.iloc[-SELECT_BARS:])
                last_refresh = now
                print(f"\n[{datetime.now(timezone.utc):%Y-%m-%d %H:%M}] "
                      f"re-selected {len(selected)} cointegrated pairs: "
                      f"{[(a, b) for a, b, _ in selected]}")
            else:
                print(f"  only {len(px)} aligned bars (<{SELECT_BARS}); "
                      f"cannot select yet")

        # daily loss halt
        acct = client.get_account()
        eq, last = float(acct.equity), float(acct.last_equity)
        halted = last > 0 and eq < last * (1 - DAILY_LOSS_LIMIT)
        if halted:
            print("  daily loss limit hit -- no new pair entries today")

        needed = {t for a, b, _ in selected for t in (a, b)} \
            | {t for k in open_pairs for t in k.split("|")}
        prices = load_prices(sorted(needed), days=30)
        held_syms = {p.symbol for p in client.get_all_positions()}

        # --- manage exits ---
        for key in list(open_pairs):
            a, b = key.split("|")
            st = open_pairs[key]
            if a not in prices or b not in prices:
                continue
            spread = prices[a] - st["beta"] * prices[b]
            z = zscore(spread)
            if not np.isfinite(z):
                continue
            if abs(z) < EXIT_Z or abs(z) > STOP_Z:
                reason = "converged" if abs(z) < EXIT_Z else "stop"
                print(f"  CLOSE {key} (z={z:+.2f}, {reason})")
                try:
                    client.close_position(a)
                    client.close_position(b)
                except Exception as e:
                    print(f"    close error: {e}")
                open_pairs.pop(key, None)

        # --- entries ---
        if not halted:
            for a, b, beta in selected:
                key = pair_key(a, b)
                if key in open_pairs or len(open_pairs) >= MAX_PAIRS:
                    continue
                if a in held_syms or b in held_syms:
                    continue  # leg already tied up in another pair
                if a not in prices or b not in prices:
                    continue
                spread = prices[a] - beta * prices[b]
                z = zscore(spread)
                if not np.isfinite(z) or abs(z) < ENTRY_Z:
                    continue
                leg = eq * PAIR_NOTIONAL_FRAC
                pa, pb = prices[a].iloc[-1], prices[b].iloc[-1]
                qa, qb = round(leg / pa, 4), round(leg / pb, 4)
                # z>0: spread rich -> short A, long B ; z<0: mirror
                if z > 0:
                    sa, sb = OrderSide.SELL, OrderSide.BUY
                else:
                    sa, sb = OrderSide.BUY, OrderSide.SELL
                try:
                    client.submit_order(MarketOrderRequest(
                        symbol=a, qty=qa, side=sa,
                        time_in_force=TimeInForce.DAY))
                    client.submit_order(MarketOrderRequest(
                        symbol=b, qty=qb, side=sb,
                        time_in_force=TimeInForce.DAY))
                    open_pairs[key] = {"beta": beta, "entry_z": z}
                    print(f"  OPEN {key}  z={z:+.2f}  "
                          f"{sa.value} {qa} {a} / {sb.value} {qb} {b}")
                except Exception as e:
                    print(f"    entry error {key}: {e}")

        if once:
            print(f"  open pairs: {list(open_pairs)}")
            return
        time.sleep(SCAN_SECONDS)
      except Exception as e:
        print(f"  loop error (retry 60s): {e}")
        time.sleep(60)


if __name__ == "__main__":
    main()
