"""Live/backtest parity: replaying history through swing_live.manage() (the code the
live bot runs) must produce the same exits as backtest_swing.simulate().

    python -m pytest tests -q        (or: python tests/test_swing_parity.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

import backtest_swing as bt  # noqa: E402
import swing_live  # noqa: E402
from history import load  # noqa: E402


def replay(df, trade):
    """Drive manage() bar by bar, exactly as the live loop would see the market."""
    e = df.index.get_loc(trade["open_time"])
    sign = 1 if trade["side"] == "long" else -1
    closed = df.iloc[:e]                               # bars closed at entry time
    atr_v = float(swing_live.indicators(closed, swing_live.PARAMS)["atr"].iloc[-1])
    state = swing_live.new_exit_state("BTC-USD", trade["side"], trade["entry"], atr_v, closed,
                                      opened_at=trade["open_time"])   # backtest enters at the bar open
    for j in range(e, len(df)):
        # during bar j the live bot sees prices down to the low (long) / up to the high (short)
        worst = df["low"].iloc[j] if sign > 0 else df["high"].iloc[j]
        why, state, _ = swing_live.manage(state, df.iloc[:j], float(worst))
        if why:
            # a signal exit discovered at bar j's first scan fills at bar j's open
            return df.index[j], why
    return df.index[-1], "end of data"


def check(symbol, n_trades=15):
    df = load(symbol, swing_live.TIMEFRAME, 3650)
    trades = bt.simulate(df, swing_live.PARAMS, leader=None)[-n_trades:]
    bad = 0
    for t in trades:
        when, why = replay(df, t)
        same_reason = why.split(" (")[0] == t["reason"].split(" (")[0]
        if when != t["close_time"] or not same_reason:
            bad += 1
            print(f"MISMATCH {symbol} {t['open_time']:%Y-%m-%d}: backtest {t['close_time']:%Y-%m-%d} "
                  f"{t['reason']!r} vs live {when:%Y-%m-%d} {why!r}")
    return len(trades), bad


def test_offline_stop_is_not_missed():
    """Bot offline for the whole trade: on restart, manage() must find the exit the
    backtest found, on the same bar, and (for stops) at the same fill price."""
    df = load("BTC-USD", swing_live.TIMEFRAME, 3650)
    trades = bt.simulate(df, swing_live.PARAMS, leader=None)[-15:]
    checked = 0
    for t in trades:
        if t["reason"] == "end of data":
            continue
        e = df.index.get_loc(t["open_time"])
        closed = df.iloc[:e]
        atr_v = float(swing_live.indicators(closed, swing_live.PARAMS)["atr"].iloc[-1])
        state = swing_live.new_exit_state("BTC-USD", t["side"], t["entry"], atr_v, closed,
                                          opened_at=t["open_time"])
        k = df.index.get_loc(t["close_time"])
        # one call after the exit bar has closed, with a harmless current price
        why, state, fill = swing_live.manage(state, df.iloc[:k + 1], t["entry"])
        assert why is not None, t
        assert why.split(" (")[0] == t["reason"].split(" (")[0], (why, t["reason"])
        if fill is not None:
            sign = 1 if t["side"] == "long" else -1
            assert abs(fill * (1 - sign * bt.SLIP) - t["exit"]) < 1e-6 * t["exit"], (fill, t["exit"])
        checked += 1
    assert checked >= 5


def test_pre_entry_prices_are_ignored():
    """Entering hours into a daily bar: that bar's earlier low must not stop us out."""
    df = load("BTC-USD", "1d", 3650).iloc[-400:].copy()
    i = len(df) - 2
    closed = df.iloc[:i]
    entry = float(df["open"].iloc[i])
    atr_v = float(swing_live.indicators(closed, swing_live.PARAMS)["atr"].iloc[-1])
    st = swing_live.new_exit_state("BTC-USD", "long", entry, atr_v, closed,
                                   opened_at=df.index[i] + pd.Timedelta(hours=3))
    df.iloc[i, df.columns.get_loc("low")] = st["stop"] - 1000        # dip BEFORE entry
    df.iloc[i, df.columns.get_loc("close")] = entry
    why, st2, fill = swing_live.manage(st, df.iloc[:i + 1], entry)
    assert why is None or "stop" not in why, why
    # a later bar dipping through the stop DOES stop us out
    df.iloc[i + 1, df.columns.get_loc("low")] = st2["stop"] - 1
    why, _, fill = swing_live.manage(st2, df.iloc[:i + 2], entry)
    assert why and "stop" in why, why


def test_parity():
    total = bad = 0
    for s in ("BTC-USD", "ETH-USD"):
        n, b = check(s)
        total, bad = total + n, bad + b
    assert bad == 0, f"{bad}/{total} trades exited differently live vs backtest"


if __name__ == "__main__":
    test_parity()
    print("parity OK")
    test_offline_stop_is_not_missed()
    print("offline stop OK")
    test_pre_entry_prices_are_ignored()
    print("pre-entry prices ignored OK")
