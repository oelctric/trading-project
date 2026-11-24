"""Backtester: replay weeks of historical 1-minute candles through the EXACT same
brain the live app uses, and find out honestly whether the strategy makes money.

Run:
    python backtest.py                 # default: ~14 days, BTC/ETH/SOL + SPY/TSLA/NVDA
    python backtest.py --days 7
    python backtest.py --symbols BTC-USD ETH-USD

Realism rules (all chosen to be pessimistic rather than flattering):
  - decisions use only candles that were closed at the time (no lookahead)
  - entry fills at the NEXT candle's open, not the close we just watched
  - if one candle touches both stop-loss and take-profit, assume the stop hit first
  - fees charged on both sides (crypto 0.25%/side ~ Coinbase Advanced mid-tier;
    stocks 0.05%/side as a slippage+spread proxy); zero-fee results shown too
  - stock positions are force-closed at the end of each session (no overnight holds)

Downloaded history is cached in backtest_data/ so re-runs are instant.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf

import config
from bounce import decide_bounce
from brain import decide
from cognitive import decide_cognitive
from signals import atr

# who leads whom: BTC leads the alts, SPY leads single stocks
LEADERS = {"ETH-USD": "BTC-USD", "SOL-USD": "BTC-USD", "TSLA": "SPY", "NVDA": "SPY"}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "backtest_data")
WINDOW = 220              # candles of history the brain sees at each step
SLIPPAGE_ATR = config.BACKTEST_SLIPPAGE_ATR
FEE_MULT = config.BACKTEST_FEE_MULTIPLIER
FEES = {"crypto": 0.0025, "stock": 0.0005}   # per side, of notional
EQUITY_START = 10_000.0   # each symbol simulated on its own $10k account

_session = requests.Session()
_session.headers["User-Agent"] = "minittraider-backtest/1.0"


# ---------------------------------------------------------------- data download

def fetch_crypto_history(product: str, days: int) -> pd.DataFrame:
    """Paginate Coinbase 1-minute candles (300 per request)."""
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)
    frames = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(minutes=300), end)
        r = _session.get(
            f"https://api.exchange.coinbase.com/products/{product}/candles",
            params={"granularity": 60, "start": cursor.isoformat(), "end": chunk_end.isoformat()},
            timeout=15,
        )
        r.raise_for_status()
        rows = r.json()
        if rows:
            frames.append(pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"]))
        cursor = chunk_end
        time.sleep(0.15)  # stay well under the public rate limit
        print(f"\r  {product}: {min(100, int(100 * (cursor - start) / (end - start)))}% downloaded", end="")
    print()
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop_duplicates("time").set_index("time").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def fetch_stock_history(symbol: str, days: int) -> pd.DataFrame:
    """Yahoo only serves 1-minute data for the last ~30 days, max ~8 days per request."""
    days = min(days, 28)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    frames = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=7), end)
        df = yf.Ticker(symbol).history(start=cursor, end=chunk_end, interval="1m")
        if not df.empty:
            frames.append(df)
        cursor = chunk_end
    print(f"  {symbol}: downloaded")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = df.index.tz_convert("UTC")
    return df[~df.index.duplicated()].sort_index().astype(float)


def get_history(symbol: str, is_crypto: bool, days: int) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{symbol}_{days}d.csv")
    if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < 6 * 3600:
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    df = fetch_crypto_history(symbol, days) if is_crypto else fetch_stock_history(symbol, days)
    if not df.empty:
        df.to_csv(cache)
    return df


# ---------------------------------------------------------------- simulation

def simulate(symbol: str, df: pd.DataFrame, is_crypto: bool,
             leader_df: pd.DataFrame | None = None, leader_name: str = "",
             strategy: str = "confluence") -> dict:
    fee = FEES["crypto" if is_crypto else "stock"]
    max_hold = {"bounce": config.BOUNCE_MAX_HOLD_MINUTES,
                "cognitive": config.COGNITIVE_MAX_HOLD_MINUTES}.get(
        strategy, config.MAX_HOLD_MINUTES)

    # leader's 3-bar move in its own ATRs, aligned to this symbol's clock
    leader_mom = None
    if leader_df is not None:
        leader_mom = ((leader_df["close"] - leader_df["close"].shift(3))
                      / atr(leader_df)).reindex(df.index, method="ffill")
    equity = EQUITY_START
    trades: list[dict] = []
    peak = equity
    max_dd = 0.0
    bars_considered = 0

    i = WINDOW
    n = len(df)
    while i < n - 1:
        window = df.iloc[i - WINDOW:i]
        # a session gap (stocks overnight / data hole) breaks 1-minute continuity
        gap_ahead = (df.index[i] - df.index[i - 1]).total_seconds() > 180
        if gap_ahead:
            i += 1
            continue
        bars_considered += 1

        if strategy == "cognitive":
            # no historical book/tape - the live app is better informed
            decision = decide_cognitive(symbol, window)
        elif strategy == "bounce":
            # zones inferred from price only - the live app also sees book walls
            decision = decide_bounce(symbol, window)
        else:
            micro = {}
            if leader_mom is not None:
                v = leader_mom.iloc[i - 1]  # value as of the last closed bar
                if not pd.isna(v):
                    micro = {"leader": float(v), "leader_name": leader_name}
            decision = decide(symbol, window, micro)
        if decision.action == "none":
            i += 1
            continue

        # ---- enter at the next bar's open with slippage
        side = decision.action
        sign = 1 if side == "long" else -1
        entry_base = float(df["open"].iloc[i])
        a = decision.atr
        # Apply slippage: for long, slippage adds to entry cost; for short, slippage reduces entry price
        entry_slippage = SLIPPAGE_ATR * a * sign
        entry = entry_base + entry_slippage
        tp = decision.tp or entry + sign * a * config.TAKE_PROFIT_ATR
        sl = decision.sl or entry - sign * a * config.STOP_LOSS_ATR
        stop_dist = abs(entry - sl)
        if stop_dist <= 0 or sign * (tp - entry) <= 0 or sign * (entry - sl) <= 0:
            i += 1  # the overnight open gapped past the zone; setup no longer valid
            continue

        qty = equity * config.RISK_PER_TRADE / stop_dist
        qty = min(qty, equity * config.MAX_POSITION_PCT / entry)

        # ---- walk forward bar by bar until an exit triggers
        exit_price, exit_reason, held = None, None, 0
        for j in range(i, min(i + max_hold, n)):
            bar = df.iloc[j]
            held = j - i + 1
            # session gap while holding -> close at last available price
            if j > i and (df.index[j] - df.index[j - 1]).total_seconds() > 180:
                exit_price, exit_reason = float(df["close"].iloc[j - 1]), "session end"
                held -= 1
                break
            hit_sl = bar["low"] <= sl if side == "long" else bar["high"] >= sl
            hit_tp = bar["high"] >= tp if side == "long" else bar["low"] <= tp
            if hit_sl:                       # pessimistic: stop checked first
                exit_price, exit_reason = sl, "stop loss"
                break
            if hit_tp:
                exit_price, exit_reason = tp, "take profit"
                break
        if exit_price is None:
            j = min(i + max_hold, n) - 1
            exit_price, exit_reason = float(df["close"].iloc[j]), "time limit"

        gross = (exit_price - entry) * qty * sign
        cost = FEE_MULT * fee * qty * (entry + exit_price)
        net = gross - cost
        equity += net
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
        trades.append({
            "time": df.index[i], "side": side, "entry": entry, "exit": exit_price,
            "gross": gross, "fees": cost, "net": net, "reason": exit_reason,
            "held": held, "confidence": decision.confidence,
        })
        i = j + 1  # move past the exit bar; one position at a time, like live

    return summarize(symbol, trades, equity, max_dd, bars_considered)


def summarize(symbol: str, trades: list[dict], equity: float, max_dd: float, bars: int) -> dict:
    wins = [t for t in trades if t["net"] > 0]
    losses = [t for t in trades if t["net"] <= 0]
    gross_win = sum(t["net"] for t in wins)
    gross_loss = -sum(t["net"] for t in losses)
    return {
        "symbol": symbol,
        "bars": bars,
        "trades": len(trades),
        "wins": len(wins),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "net": sum(t["net"] for t in trades),
        "net_no_fees": sum(t["gross"] for t in trades),
        "fees": sum(t["fees"] for t in trades),
        "avg_win": gross_win / len(wins) if wins else 0.0,
        "avg_loss": gross_loss / len(losses) if losses else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss else float("inf"),
        "max_dd": max_dd,
        "return_pct": (equity - EQUITY_START) / EQUITY_START,
        "avg_hold": sum(t["held"] for t in trades) / len(trades) if trades else 0.0,
        "trade_list": trades,
    }


# ---------------------------------------------------------------- report

def report(results: list[dict], days: int, strategy: str = "confluence") -> None:
    print(f"\n{'=' * 78}")
    print(f"BACKTEST RESULTS ({strategy}) - last {days} days, 1-minute bars, same brain as live app")
    print(f"fees: crypto {FEES['crypto']:.2%}/side, stocks {FEES['stock']:.2%}/side | "
          f"$10,000 per symbol")
    print(f"{'=' * 78}")
    header = (f"{'symbol':<10}{'bars':>7}{'trades':>7}{'win%':>7}{'net P&L':>11}"
              f"{'no-fee P&L':>12}{'fees paid':>11}{'PF':>6}{'maxDD':>7}{'ret%':>7}")
    print(header)
    print("-" * len(header))
    for r in results:
        pf = f"{r['profit_factor']:.2f}" if r["trades"] else "-"
        print(f"{r['symbol']:<10}{r['bars']:>7}{r['trades']:>7}"
              f"{r['win_rate']:>7.0%}{r['net']:>+11.2f}{r['net_no_fees']:>+12.2f}"
              f"{r['fees']:>11.2f}{pf:>6}{r['max_dd']:>7.1%}{r['return_pct']:>+7.2%}")
    total_net = sum(r["net"] for r in results)
    total_gross = sum(r["net_no_fees"] for r in results)
    total_fees = sum(r["fees"] for r in results)
    total_trades = sum(r["trades"] for r in results)
    total_wins = sum(r["wins"] for r in results)
    print("-" * len(header))
    print(f"{'TOTAL':<10}{'':>7}{total_trades:>7}"
          f"{(total_wins / total_trades if total_trades else 0):>7.0%}"
          f"{total_net:>+11.2f}{total_gross:>+12.2f}{total_fees:>11.2f}")
    print()
    verdict = "PROFITABLE" if total_net > 0 else "NOT PROFITABLE"
    print(f"verdict with fees:    {verdict} ({total_net:+,.2f} over {days} days)")
    print(f"verdict before fees:  {'profitable' if total_gross > 0 else 'not profitable'} "
          f"({total_gross:+,.2f}) - fees ate {total_fees:,.2f}")

    # save every simulated trade for inspection
    all_trades = [dict(t, symbol=r["symbol"]) for r in results for t in r["trade_list"]]
    if all_trades:
        out = os.path.join(os.path.dirname(__file__), "backtest_trades.csv")
        pd.DataFrame(all_trades).to_csv(out, index=False)
        print(f"\nevery simulated trade saved to {os.path.basename(out)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--symbols", nargs="*", default=None)
    # default to the minute strategies this script documents; the swing
    # strategy (the live default) has its own backtester - backtest_swing.py,
    # also reachable here with --strategy swing
    ap.add_argument("--strategy", choices=["confluence", "bounce", "cognitive", "swing"],
                    default="cognitive" if config.STRATEGY == config.Strategy.SWING
                    else config.STRATEGY.value)
    args, unknown = ap.parse_known_args()
    if unknown and args.strategy != "swing":
        ap.error(f"unrecognized arguments: {' '.join(unknown)}")
    if args.strategy == "swing":
        import backtest_swing
        # forward the options backtest_swing understands; --strategy is ours
        fwd, skip = [], False
        for a in sys.argv[1:]:
            if skip:
                skip = False
                continue
            if a == "--strategy":
                skip = True
                continue
            if a.startswith("--strategy="):
                continue
            fwd.append(a)
        sys.argv = [sys.argv[0]] + fwd
        backtest_swing.main()
        return

    crypto = ["BTC-USD", "ETH-USD", "SOL-USD"]
    stocks = ["SPY", "TSLA", "NVDA"]
    watch = ([(s, s in crypto or s.endswith("-USD")) for s in args.symbols]
             if args.symbols else [(s, True) for s in crypto] + [(s, False) for s in stocks])

    print(f"downloading {args.days} days of 1-minute history...")
    frames: dict[str, pd.DataFrame] = {}
    for symbol, is_crypto in watch:
        try:
            df = get_history(symbol, is_crypto, args.days)
        except Exception as e:
            print(f"  {symbol}: download failed ({e})")
            continue
        if df.empty or len(df) < WINDOW + 10:
            print(f"  {symbol}: not enough data, skipped")
            continue
        frames[symbol] = df

    results = []
    for symbol, is_crypto in watch:
        if symbol not in frames:
            continue
        df = frames[symbol]
        leader_name = LEADERS.get(symbol, "")
        t0 = time.time()
        print(f"  simulating {symbol} over {len(df):,} bars...", flush=True)
        results.append(simulate(symbol, df, is_crypto,
                                frames.get(leader_name), leader_name, args.strategy))
        print(f"    done in {time.time() - t0:.0f}s: {results[-1]['trades']} trades, "
              f"net {results[-1]['net']:+,.2f}")
    if results:
        report(results, args.days, args.strategy)


if __name__ == "__main__":
    main()
