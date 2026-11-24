"""Indicator filters: no lookahead, and filters actually gate entries.

    python tests/test_indicators.py
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import swing_live  # noqa: E402
from history import load  # noqa: E402
from swing import FILTERS, entry_signal, indicators  # noqa: E402

COLS = ["adx", "macd_hist", "squeeze", "weekly_trend", "obv_trend", "atr_rank", "close_loc", "rsi"]


def test_no_lookahead():
    df = load("BTC-USD", "1d", 3650)
    full = indicators(df, swing_live.PARAMS)
    for n in (900, 2000, 3000):
        part = indicators(df.iloc[:n], swing_live.PARAMS)
        d = (full.loc[part.index, COLS] - part[COLS]).abs().max().max()
        assert d < 1e-9, (n, d)


def test_calm_vol_blocks_wild_markets():
    df = load("BTC-USD", "1d", 3650)
    p = replace(swing_live.PARAMS, filters=())
    ind = indicators(df, p)
    pf = replace(p, filters=("calm_vol",))
    blocked = passed = 0
    for i in range(400, len(df)):
        row = ind.iloc[i]
        side, _ = entry_signal(df["close"].iloc[i], row, p, 0.001)
        if side == "none":
            continue
        side_f, why = entry_signal(df["close"].iloc[i], row, pf, 0.001)
        if row["atr_rank"] > 0.8:
            assert side_f == "none" and "calm_vol" in why
            blocked += 1
        else:
            assert side_f == side
            passed += 1
    assert blocked > 0 and passed > 0


def test_every_filter_runs():
    df = load("ETH-USD", "1d", 3650)
    for name in FILTERS:
        p = replace(swing_live.PARAMS, filters=(name,))
        ind = indicators(df, p).iloc[-1]
        entry_signal(df["close"].iloc[-1], ind, p, 0.001)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
