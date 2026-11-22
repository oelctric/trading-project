"""Config sweep: which settings (if any) make the strategy clear the fee hurdle?

The slow part - computing the six signal scores and volatility regime for every
bar - depends only on the data, not the trade rules. So we compute those once
per symbol (cached to disk), then replay many rule configurations over the
precomputed scores in seconds each.

The confluence math below is copied verbatim from brain.decide() so results
match the live app; only the thresholds/targets are parameterized.

Sizing is a fixed $10,000 account per symbol with no compounding, so configs
are compared on equal footing. Trades are entered at the next bar's open,
stops assumed to fill before targets in the same bar (pessimistic).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import config
from backtest import CACHE_DIR, WINDOW, get_history
from signals import ALL_SIGNALS, atr, volatility_regime

EQUITY = 10_000.0
DAYS = 14
SYMBOLS = [("BTC-USD", True), ("ETH-USD", True), ("SOL-USD", True),
           ("SPY", False), ("TSLA", False), ("NVDA", False)]

# who leads whom: BTC leads the alts, SPY leads single stocks
LEADERS = {"ETH-USD": "BTC-USD", "SOL-USD": "BTC-USD", "TSLA": "SPY", "NVDA": "SPY"}

# (name, min_conf, min_agree, tp_atr, sl_atr, hold_minutes)
CONFIGS = [
    ("baseline (live)",   0.60, 3, 1.5, 1.0,  5),
    ("strict",            0.70, 4, 1.5, 1.0,  5),
    ("very strict",       0.75, 5, 1.5, 1.0,  5),
    ("wider targets",     0.60, 3, 3.0, 1.5, 15),
    ("strict + wide",     0.70, 4, 3.0, 1.5, 15),
    ("ultra selective",   0.80, 5, 4.0, 2.0, 30),
]

# fee scenarios, per side of notional
FEE_TAKER = {"crypto": 0.0025, "stock": 0.0005}   # market orders (what the live sim assumes)
FEE_MAKER = {"crypto": 0.0010, "stock": 0.0005}   # patient limit orders on a maker-fee tier


# ---------------------------------------------------------------- precompute

def precompute(symbol: str, df: pd.DataFrame,
               leader_df: pd.DataFrame | None) -> pd.DataFrame:
    """One row per bar: signal scores, regime, ATR. Cached - this is the slow pass."""
    cache = os.path.join(CACHE_DIR, f"{symbol}_{DAYS}d_scores_v2.csv")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(
            os.path.join(CACHE_DIR, f"{symbol}_{DAYS}d.csv")):
        return pd.read_csv(cache)

    # leader's 3-bar move in its own ATRs, aligned to this symbol's clock;
    # scored exactly like brain.micro_signals does: clip(move / 1.5)
    leader_score = None
    if leader_df is not None:
        lm = (leader_df["close"] - leader_df["close"].shift(3)) / atr(leader_df)
        leader_score = (lm / 1.5).clip(-1, 1).reindex(df.index, method="ffill")

    t0 = time.time()
    rows = []
    for i in range(WINDOW, len(df) - 1):
        window = df.iloc[i - WINDOW:i]
        regime, mult, _ = volatility_regime(window)
        row = {
            "i": i,
            "gap": (df.index[i] - df.index[i - 1]).total_seconds() > 180,
            "atr": float(atr(window).iloc[-1]),
            "regime": regime,
            "mult": mult,
            **{name: fn(window)[0] for name, fn in ALL_SIGNALS.items()},
        }
        if leader_score is not None:
            # value as of the last CLOSED bar - no lookahead
            v = leader_score.iloc[i - 1]
            row["leader"] = 0.0 if pd.isna(v) else float(v)
        rows.append(row)
    pre = pd.DataFrame(rows)
    pre.to_csv(cache, index=False)
    print(f"  {symbol}: scored {len(pre):,} bars in {time.time() - t0:.0f}s")
    return pre


def confluence(scores: np.ndarray, regime: str, mult: float,
               min_conf: float, min_agree: int) -> str:
    """Verbatim port of the gating math in brain.decide()."""
    bulls = int((scores > 0.25).sum())
    bears = int((scores < -0.25).sum())
    avg = float(scores.mean())
    direction = "long" if avg > 0 else "short"
    agreeing = bulls if direction == "long" else bears
    opposing = bears if direction == "long" else bulls
    alignment = agreeing / len(scores)
    conf = max(0.0, abs(avg) * 0.5 + alignment * 0.5 - 0.20 * opposing) * mult
    conf = min(conf, 0.99)
    if regime in ("dead", "wild", "unknown"):
        return "none"
    if agreeing < min_agree or conf < min_conf:
        return "none"
    return direction


# ---------------------------------------------------------------- simulation

def run_config(df: pd.DataFrame, pre: pd.DataFrame, is_crypto: bool,
               min_conf: float, min_agree: int, tp_atr: float, sl_atr: float,
               hold: int) -> list[dict]:
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    times = df.index.asi8  # ns
    n = len(df)

    sig_names = list(ALL_SIGNALS) + (["leader"] if "leader" in pre.columns else [])
    score_mat = pre[sig_names].to_numpy()
    regimes = pre["regime"].to_numpy()
    mults = pre["mult"].to_numpy()
    atrs = pre["atr"].to_numpy()
    gaps = pre["gap"].to_numpy()

    kind = "crypto" if is_crypto else "stock"
    trades = []
    k = 0
    while k < len(pre):
        if gaps[k]:
            k += 1
            continue
        action = confluence(score_mat[k], regimes[k], mults[k], min_conf, min_agree)
        if action == "none":
            k += 1
            continue

        i = int(pre["i"].iloc[k])
        sign = 1 if action == "long" else -1
        entry = opens[i]
        a = atrs[k]
        if a <= 0 or entry <= 0:
            k += 1
            continue
        tp = entry + sign * a * tp_atr
        sl = entry - sign * a * sl_atr
        qty = min(EQUITY * config.RISK_PER_TRADE / (a * sl_atr),
                  EQUITY * config.MAX_POSITION_PCT / entry)

        exit_price, held_end = None, i
        for j in range(i, min(i + hold, n)):
            held_end = j
            if j > i and (times[j] - times[j - 1]) > 180e9:
                exit_price = closes[j - 1]
                break
            hit_sl = lows[j] <= sl if sign == 1 else highs[j] >= sl
            hit_tp = highs[j] >= tp if sign == 1 else lows[j] <= tp
            if hit_sl:
                exit_price = sl
                break
            if hit_tp:
                exit_price = tp
                break
        if exit_price is None:
            held_end = min(i + hold, n) - 1
            exit_price = closes[held_end]

        gross = (exit_price - entry) * qty * sign
        notional = qty * (entry + exit_price)
        trades.append({
            "gross": gross,
            "net_taker": gross - FEE_TAKER[kind] * notional,
            "net_maker": gross - FEE_MAKER[kind] * notional,
        })
        k = held_end + 1 - WINDOW  # pre row k corresponds to bar i = WINDOW + k
    return trades


# ---------------------------------------------------------------- main

def main() -> None:
    print("loading cached data and scoring bars (slow only the first time)...")
    frames: dict[str, pd.DataFrame] = {}
    for symbol, is_crypto in SYMBOLS:
        df = get_history(symbol, is_crypto, DAYS)
        if df.empty or len(df) < WINDOW + 10:
            print(f"  {symbol}: no data, skipped")
            continue
        frames[symbol] = df
    data = []
    for symbol, is_crypto in SYMBOLS:
        if symbol not in frames:
            continue
        leader_df = frames.get(LEADERS.get(symbol, ""))
        data.append((symbol, is_crypto, frames[symbol],
                     precompute(symbol, frames[symbol], leader_df)))

    print(f"\n{'=' * 96}")
    print(f"CONFIG SWEEP - last {DAYS} days, all six symbols combined, $10k/symbol, no compounding")
    print(f"taker fees: crypto {FEE_TAKER['crypto']:.2%}/side | "
          f"maker scenario: crypto {FEE_MAKER['crypto']:.2%}/side | stocks {FEE_TAKER['stock']:.2%}/side")
    print(f"{'=' * 96}")
    head = (f"{'config':<18}{'conf':>6}{'agree':>6}{'tp/sl':>9}{'hold':>6}"
            f"{'trades':>8}{'win%':>6}{'no-fee P&L':>12}{'maker P&L':>11}{'taker P&L':>11}")
    print(head)
    print("-" * len(head))
    for name, min_conf, min_agree, tp_atr, sl_atr, hold in CONFIGS:
        all_trades = []
        for symbol, is_crypto, df, pre in data:
            all_trades += run_config(df, pre, is_crypto, min_conf, min_agree,
                                     tp_atr, sl_atr, hold)
        nt = len(all_trades)
        wins = sum(1 for t in all_trades if t["net_taker"] > 0)
        gross = sum(t["gross"] for t in all_trades)
        maker = sum(t["net_maker"] for t in all_trades)
        taker = sum(t["net_taker"] for t in all_trades)
        print(f"{name:<18}{min_conf:>6.0%}{min_agree:>6}{f'{tp_atr}/{sl_atr}':>9}"
              f"{f'{hold}m':>6}{nt:>8}{(wins / nt if nt else 0):>6.0%}"
              f"{gross:>+12.2f}{maker:>+11.2f}{taker:>+11.2f}")
    print("-" * len(head))
    print("win% is counted after taker fees. hold = max minutes before forced exit.")


if __name__ == "__main__":
    main()
