"""Train the MetaAgent planner: Q-learning over macro states.

    python train_cognitive.py

For every historical minute, the planner chooses TREND / REVERT / STAND_ASIDE.
If a sub-agent then finds a clean setup, the trade is simulated (entry next
bar, stop-first intrabar, taker fees) and the reward is the P&L in R-multiples
(profit / amount risked). Standing aside earns exactly 0 - so a strategy mode
only accumulates positive value in a macro state if its trades actually beat
fees there. Learned values are judged on the held-out final third of history.

Perception features are precomputed per bar and cached (the slow pass).
"""

from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import pandas as pd

from backtest import CACHE_DIR, FEES, get_history
from cognitive import (ACTIONS, CONFIDENCE_BAR, MetaAgent, RiskManager,
                       macro_state_key, run_sub_agent)
from perception import State, perceive

DAYS_CRYPTO = 45
DAYS_STOCK = 28
WINDOW = 260
TRAIN_FRACTION = 2 / 3
EQUITY = 10_000.0
RISK_FRACTION = 0.01
EPOCHS = 4
ALPHA = 0.08           # Q-value learning rate
EPSILON = 0.25         # exploration during training

SYMBOLS = [("BTC-USD", True), ("ETH-USD", True), ("SOL-USD", True),
           ("SPY", False), ("TSLA", False), ("NVDA", False)]

rng = random.Random(11)

STATE_FIELDS = ["price", "atr", "hour", "regime", "trend_score", "boll_z",
                "rsi", "ema21_dist", "support", "resistance"]


# ---------------------------------------------------------------- precompute

def precompute(symbol: str, df: pd.DataFrame, days: int) -> pd.DataFrame:
    cache = os.path.join(CACHE_DIR, f"{symbol}_{days}d_cog.csv")
    src = os.path.join(CACHE_DIR, f"{symbol}_{days}d.csv")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(src):
        return pd.read_csv(cache)
    t0 = time.time()
    rows = []
    for i in range(WINDOW, len(df) - 1):
        s = perceive(df.iloc[i - WINDOW:i])   # no book/tape in history - neutral
        row = {"i": i,
               "gap": (df.index[i] - df.index[i - 1]).total_seconds() > 180,
               "fuzz": json.dumps({k: round(v, 3) for k, v in s.fuzz.items()})}
        for fld in STATE_FIELDS:
            row[fld] = getattr(s, fld)
        rows.append(row)
    pre = pd.DataFrame(rows)
    pre.to_csv(cache, index=False)
    print(f"  {symbol}: perceived {len(pre):,} bars in {time.time() - t0:.0f}s", flush=True)
    return pre


def row_to_state(row) -> State:
    s = State()
    for fld in STATE_FIELDS:
        v = row[fld]
        if fld in ("support", "resistance"):
            v = None if pd.isna(v) else float(v)
        setattr(s, fld, v)
    s.fuzz = json.loads(row["fuzz"])
    return s


class SymbolData:
    def __init__(self, symbol: str, is_crypto: bool, df: pd.DataFrame, pre: pd.DataFrame):
        self.symbol = symbol
        self.fee = FEES["crypto" if is_crypto else "stock"]
        self.opens = df["open"].to_numpy()
        self.highs = df["high"].to_numpy()
        self.lows = df["low"].to_numpy()
        self.closes = df["close"].to_numpy()
        self.times = df.index.asi8
        self.n = len(df)
        self.pre = pre.reset_index(drop=True)
        self.states = [row_to_state(r) for _, r in self.pre.iterrows()]
        self.keys = [macro_state_key(s) for s in self.states]
        self.idx = self.pre["i"].to_numpy(dtype=int)
        self.gap = self.pre["gap"].to_numpy(dtype=bool)
        self.split = int(len(self.pre) * TRAIN_FRACTION)


# ---------------------------------------------------------------- trade sim

def simulate_trade(sd: SymbolData, k: int, side: str, stop: float, target: float,
                   hold: int = 30) -> tuple[float, int]:
    """Net P&L (after fees) and exit row for one plan. Pessimistic fills."""
    i = int(sd.idx[k])
    sign = 1 if side == "long" else -1
    entry = sd.opens[i]
    if sign * (target - entry) <= 0 or sign * (entry - stop) <= 0:
        return 0.0, k  # gapped past the setup - trade never happens
    qty = min(EQUITY * RISK_FRACTION / (sign * (entry - stop)), EQUITY * 0.25 / entry)

    exit_price, j = None, i
    for j in range(i, min(i + hold, sd.n)):
        if j > i and (sd.times[j] - sd.times[j - 1]) > 180e9:
            exit_price = sd.closes[j - 1]
            break
        if (sd.lows[j] <= stop if sign == 1 else sd.highs[j] >= stop):
            exit_price = stop
            break
        if (sd.highs[j] >= target if sign == 1 else sd.lows[j] <= target):
            exit_price = target
            break
    if exit_price is None:
        j = min(i + hold, sd.n) - 1
        exit_price = sd.closes[j]
    net = sign * (exit_price - entry) * qty - sd.fee * qty * (entry + exit_price)
    return net, max(k, j - int(sd.idx[0]))


def bar_time(sd: SymbolData, k: int):
    from datetime import datetime, timezone
    k = min(k, len(sd.idx) - 1)   # a trade can exit on the final bar
    return datetime.fromtimestamp(sd.times[int(sd.idx[k])] / 1e9, tz=timezone.utc)


def attempt(sd: SymbolData, k: int, mode: str, risk: RiskManager | None) -> tuple[float, int]:
    """Let a sub-agent act under `mode` at row k. Returns (net P&L, exit row)."""
    s = sd.states[k]
    confidence, _, plan = run_sub_agent(mode, s)
    if plan is None or confidence < CONFIDENCE_BAR:
        return 0.0, k
    side, stop, target, _ = plan
    checker = risk or RiskManager()
    # historical bar time, not the wall clock - otherwise one loss streak
    # freezes the cooldown for the entire simulation
    if checker.veto(side, s.price, stop, target, EQUITY, now=bar_time(sd, k)) is not None:
        return 0.0, k
    return simulate_trade(sd, k, side, stop, target)


# ---------------------------------------------------------------- training

def train(data: list[SymbolData]) -> MetaAgent:
    meta = MetaAgent(path="__none__")   # start blank
    meta.trained = True
    for epoch in range(EPOCHS):
        t0 = time.time()
        pnl, trades = 0.0, 0
        for sd in data:
            k = 0
            while k < sd.split:
                if sd.gap[k] or sd.states[k].regime in ("wild", "unknown"):
                    k += 1
                    continue
                key = sd.keys[k]
                vals = meta.values(key)
                a = rng.randrange(3) if rng.random() < EPSILON \
                    else max(range(3), key=lambda x: vals[x])
                if ACTIONS[a] == "STAND_ASIDE":
                    reward, k_exit = 0.0, k
                else:
                    net, k_exit = attempt(sd, k, ACTIONS[a], None)
                    reward = net / (EQUITY * RISK_FRACTION)   # R-multiples
                    if k_exit > k:
                        pnl += net
                        trades += 1
                vals[a] += ALPHA * (reward - vals[a])
                k = k_exit + 1
        print(f"epoch {epoch + 1}/{EPOCHS}: explored {trades} trades, "
              f"raw P&L {pnl:+,.2f} | {time.time() - t0:.0f}s", flush=True)
    return meta


def evaluate(data: list[SymbolData], meta: MetaAgent | None, part: str,
             forced_mode: str | None = None) -> tuple[float, int, int]:
    """Greedy policy (or a forced always-on mode) on train or test rows."""
    total, trades, wins = 0.0, 0, 0
    for sd in data:
        lo, hi = (0, sd.split) if part == "train" else (sd.split, len(sd.pre))
        risk = RiskManager()
        k = lo
        while k < hi:
            if sd.gap[k] or sd.states[k].regime in ("wild", "unknown"):
                k += 1
                continue
            if forced_mode:
                mode = forced_mode
            else:
                mode, _ = meta.choose(sd.states[k])
            if mode == "STAND_ASIDE":
                k += 1
                continue
            net, k_exit = attempt(sd, k, mode, risk)
            if k_exit > k:
                total += net
                trades += 1
                wins += net > 0
                risk.record(net, now=bar_time(sd, k_exit))
            k = k_exit + 1
    return total, trades, wins


def main() -> None:
    print("loading data and perceiving history (slow only the first time)...")
    data = []
    for symbol, is_crypto in SYMBOLS:
        days = DAYS_CRYPTO if is_crypto else DAYS_STOCK
        df = get_history(symbol, is_crypto, days)
        if df.empty or len(df) < WINDOW + 100:
            print(f"  {symbol}: not enough data, skipped")
            continue
        data.append(SymbolData(symbol, is_crypto, df, precompute(symbol, df, days)))

    print(f"\ntraining planner: {EPOCHS} epochs, eps={EPSILON}, alpha={ALPHA}")
    meta = train(data)

    print(f"\n{'=' * 78}")
    print("JUDGMENT - net P&L after taker fees ($10k/symbol, 1% risk per trade)")
    print(f"{'=' * 78}")
    print(f"{'policy':<26}{'period':>8}{'P&L':>12}{'trades':>8}{'win%':>7}")
    print("-" * 61)
    for name, forced in [("learned planner", None),
                         ("always TREND mode", "TREND"),
                         ("always REVERT mode", "REVERT")]:
        for part in ("train", "test"):
            total, trades, wins = evaluate(data, meta, part, forced)
            wr = f"{wins / trades:.0%}" if trades else "-"
            print(f"{name:<26}{part:>8}{total:>+12.2f}{trades:>8}{wr:>7}")
    print("-" * 61)
    print("the only number that matters: 'learned planner' on 'test'.")

    n_states = len(meta.q)
    meta.save(meta=dict(trained=time.strftime("%Y-%m-%d"), states=n_states,
                        epochs=EPOCHS, days=DAYS_CRYPTO))
    print(f"\nplanner saved to meta_q.json ({n_states} macro states learned) - "
          f"the live app now uses it automatically.")


if __name__ == "__main__":
    main()
