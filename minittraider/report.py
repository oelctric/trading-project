"""Is live paper trading behaving like the backtest said it would?

    python report.py

Compares closed swing trades in trades.csv with what the backtest produced on
unseen data (Jul 2023 -> now) with the same settings, and only calls a
difference real when it's statistically meaningful - a handful of trades proves
nothing either way.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import backtest_swing as bt
import config
import swing_live
from history import load
from portfolio import STATE_FILE, TRADES_FILE

MIN_TRADES = 20


def backtest_expectation() -> dict:
    frames = {s: load(s, "1d", 3650, max_age_hours=24) for s in bt.SYMBOLS}
    fee = config.FUTURES_FEE_PER_SIDE if config.USE_LEVERAGE else config.FEE_PER_SIDE["crypto"]
    cut = pd.Timestamp("2023-07-24", tz="UTC")
    r = np.array([t["r"] for s, df in frames.items()
                  for t in bt.simulate(df, swing_live.PARAMS, fee=fee, leader=frames.get("BTC-USD"))
                  if t["open_time"] >= cut])
    days = (pd.Timestamp.now(tz="UTC") - cut).days
    return {"avg_r": r.mean(), "std_r": r.std(ddof=1), "win": (r > 0).mean(),
            "per_month": len(r) / days * 30.4, "n": len(r)}


def main() -> None:
    exp = backtest_expectation()
    print(f"backtest expectation (unseen Jul 2023 -> now, {exp['n']} trades): "
          f"avg {exp['avg_r']:+.2f}R per trade, win {exp['win']:.0%}, "
          f"~{exp['per_month']:.1f} trades/month")

    if os.path.exists(STATE_FILE):
        st = json.load(open(STATE_FILE))
        print(f"paper account: cash ${st['cash']:,.2f}, realized P&L {st['realized_pnl']:+,.2f}, "
              f"{len(st['positions'])} open position(s)")
        for sym, p in st["positions"].items():
            es = p.get("exit_state") or {}
            if es:
                print(f"  open {p['side']} {sym}: entry {p['entry']:.6g}, stop {es['stop']:.6g} "
                      f"({es['stage']}), best {es['mfe_r']:+.2f}R, {es['bars']} days")

    if not os.path.exists(TRADES_FILE):
        print("\nno closed trades yet")
        return
    df = pd.read_csv(TRADES_FILE)
    if "strategy" not in df.columns:
        print("\ntrades.csv is from the old minute strategy - no swing trades logged yet")
        return
    sw = df[df["strategy"] == "swing"]
    n = len(sw)
    print(f"\nlive swing trades closed: {n}")
    if n == 0:
        return
    r = sw["r_multiple"].astype(float)
    print(f"  avg {r.mean():+.2f}R, win {(r > 0).mean():.0%}, total {r.sum():+.1f}R, "
          f"P&L {sw['pnl'].astype(float).sum():+,.2f}, avg hold {sw['hold_minutes'].astype(float).mean() / 1440:.1f} days")
    print("  exits:", sw["exit_reason"].str.replace(r" \(.*", "", regex=True).value_counts().to_dict())
    if n < MIN_TRADES:
        print(f"  verdict: too early to judge - need ~{MIN_TRADES} trades "
              f"(expect ~{MIN_TRADES / max(exp['per_month'], 0.1):.0f} months)")
        return
    z = (r.mean() - exp["avg_r"]) / (exp["std_r"] / np.sqrt(n))
    if z < -2:
        verdict = "WORSE than the backtest beyond normal luck - investigate before trusting it"
    elif z > 2:
        verdict = "better than the backtest (enjoy it, but expect it to cool off)"
    else:
        verdict = "consistent with the backtest (difference is within normal luck)"
    print(f"  verdict: {verdict} (z = {z:+.1f})")


if __name__ == "__main__":
    main()
