"""Main loop: scan the market every minute, manage open scalps, explain every decision.

Run it:            python trader.py
Verbose reasoning: python trader.py -v      (prints the full thinking for every symbol)
One scan and exit: python trader.py --once
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone

import pandas as pd

import config
from bounce import decide_bounce
from brain import decide, fmt_price
from cognitive import decide_cognitive, record_result
from data import get_candles, fetch_micro, order_book_walls, tape_imbalance
from logger import get_logger, setup_logger, get_daily_log_file
from portfolio import Portfolio
import swing_live

# Setup logging
if config.LOG_TO_FILE:
    setup_logger(level=config.LOG_LEVEL, log_file=get_daily_log_file())
else:
    setup_logger(level=config.LOG_LEVEL)
log = get_logger("trader")

VERBOSE = "-v" in sys.argv
ONCE = "--once" in sys.argv

WATCHLIST: list[tuple[str, bool]] = (
    [(s, True) for s in config.CRYPTO_SYMBOLS] + [(s, False) for s in config.STOCK_SYMBOLS]
)


def fresh(df: pd.DataFrame) -> bool:
    """Is the latest candle recent enough to trade on? (Stocks go stale off-hours.)"""
    if df is None or df.empty:
        return False
    age = (pd.Timestamp.now(tz="UTC") - df.index[-1]).total_seconds() / 60
    return age <= config.STALE_DATA_MINUTES


def scan(portfolio: Portfolio) -> None:
    """Run one market scan cycle - check all symbols and manage positions."""
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    prices: dict[str, float] = {}
    dfs: dict[str, pd.DataFrame] = {}
    log.info("\n=== scan %s ===", now)
    print(f"\n=== scan {now} ===")
    swing_mode = config.STRATEGY == config.Strategy.SWING
    if swing_mode:
        swing_live.prefetch([s for s, c in WATCHLIST if c or not config.SWING_CRYPTO_ONLY])

    for symbol, is_crypto in WATCHLIST:
        if swing_mode and config.SWING_CRYPTO_ONLY and not is_crypto \
                and symbol not in portfolio.positions:
            continue   # swing trades crypto only - don't download stock data every minute
        try:
            df = get_candles(symbol, is_crypto)
        except Exception as e:
            log.error("Data error for %s: %s", symbol, e)
            print(f"  {symbol}: data error ({e})")
            continue
        dfs[symbol] = df

        if not fresh(df):
            pos = portfolio.positions.get(symbol)
            if pos is not None and pos.exit_state:
                log.warning("Data stale for %s - holding swing position, will re-check next scan", symbol)
                print(f"  {symbol}: no fresh data - holding {pos.side} (multi-day trade), re-checking next scan")
                continue
            if symbol in portfolio.positions:
                log.warning("Data stale for %s with open position - closing", symbol)
                print(f"  {symbol}: data went stale with a position open - closing at last price")
                price = float(df["close"].iloc[-1]) if df is not None and not df.empty else portfolio.positions[symbol].entry
                pnl = portfolio.close(symbol, price, "stale data / market closed")
                print(f"    closed for {pnl:+.2f}")
            elif VERBOSE:
                print(f"  {symbol}: skipped (no fresh data - market probably closed)")
            continue

        price = float(df["close"].iloc[-1])
        prices[symbol] = price

        # 1a. Swing positions: dynamic exit manager, re-evaluated every scan.
        if symbol in portfolio.positions and portfolio.positions[symbol].exit_state:
            pos = portfolio.positions[symbol]
            try:
                bars = swing_live.closed_bars(symbol, is_crypto)
            except Exception as e:
                log.error("bar fetch failed for %s: %s", symbol, e)
                bars = None
            why, pos.exit_state, fill = swing_live.manage(pos.exit_state, bars, price)
            if fill is not None and portfolio.check_liquidation(symbol, fill):
                why = "LIQUIDATED (gapped through the stop and the liquidation level)"
            elif fill is None and portfolio.check_liquidation(symbol, price):  # an earlier stop fill wins
                why, fill = "LIQUIDATED (price hit the liquidation level)", None
            portfolio.save()
            if why:
                exit_px = fill if fill is not None else price
                pnl = portfolio.close(symbol, exit_px, why)
                record_result(pnl)
                print(f"  CLOSE {pos.side} {symbol} @ {fmt_price(exit_px)} -> {pnl:+.2f} ({why})")
            else:
                print(f"  hold {pos.side} {symbol}: {pos.pnl(price):+.2f} unrealized, "
                      f"{swing_live.describe(pos.exit_state, price)}")
            continue

        # 1b. Manage an open position first.
        if symbol in portfolio.positions:
            pos = portfolio.positions[symbol]
            why = portfolio.check_exit(symbol, price)
            if why:
                pnl = portfolio.close(symbol, price, why)
                record_result(pnl)
                log.info("CLOSE %s %s @ %s -> %+.2f (%s)", pos.side, symbol, fmt_price(price), pnl, why)
                print(f"  CLOSE {pos.side} {symbol} @ {fmt_price(price)} -> {pnl:+.2f} ({why})")
            else:
                print(f"  hold {pos.side} {symbol}: {pos.pnl(price):+.2f} unrealized, "
                      f"{pos.age_minutes():.1f}/{pos.max_hold or config.MAX_HOLD_MINUTES:.0f} min")
            continue

        # 2. Otherwise, think about a new entry.
        exit_state = None
        if config.STRATEGY == config.Strategy.SWING:
            if config.SWING_CRYPTO_ONLY and not is_crypto:
                continue
            try:
                bars = swing_live.closed_bars(symbol, is_crypto)
            except Exception as e:
                log.error("bar fetch failed for %s: %s", symbol, e)
                print(f"  {symbol}: daily bars unavailable ({e}) - skipping entry check")
                continue
            decision = swing_live.decide_swing(symbol, bars)
            decision.price = price          # we fill at the live price, not the bar close
            if decision.action != "none":
                exit_state = swing_live.new_exit_state(symbol, decision.action, price,
                                                       decision.atr, bars)
                decision.sl = exit_state["stop"]
            max_hold = 0
        elif config.STRATEGY == config.Strategy.COGNITIVE:
            walls = order_book_walls(symbol) if is_crypto else None
            tape = tape_imbalance(symbol) if is_crypto else None
            decision = decide_cognitive(symbol, df, walls, tape,
                                        equity=portfolio.equity(prices))
            max_hold = config.COGNITIVE_MAX_HOLD_MINUTES
        elif config.STRATEGY == config.Strategy.BOUNCE:
            walls = order_book_walls(symbol) if is_crypto else None
            tape = tape_imbalance(symbol) if is_crypto else None
            decision = decide_bounce(symbol, df, walls, tape)
            max_hold = config.BOUNCE_MAX_HOLD_MINUTES
        else:
            leader = "BTC-USD" if is_crypto else "SPY"
            leader_df = dfs.get(leader) if symbol != leader else None
            micro = fetch_micro(symbol, is_crypto, leader_df, leader)
            decision = decide(symbol, df, micro)
            max_hold = 0
        if decision.action != "none" and portfolio.can_open(symbol):
            summary = decision.reasons[-1]
            pos = portfolio.open(symbol, decision.action, decision.price,
                                 decision.atr, decision.confidence, summary,
                                 tp=decision.tp, sl=decision.sl, max_hold=max_hold,
                                 exit_state=exit_state)
            if pos:
                if exit_state is not None:
                    swing_live.mark_acted(symbol, bars)   # only now that it really opened
                log.info("OPEN %s %s: qty %.6f, tp %s, sl %s", pos.side, symbol, pos.qty,
                         "dynamic" if pos.exit_state else fmt_price(pos.take_profit),
                         fmt_price(pos.stop_loss))
                print("  " + decision.explain().replace("\n", "\n  "))
                tp_txt = "dynamic" if pos.exit_state else fmt_price(pos.take_profit)
                print(f"  OPEN {pos.side} {symbol}: qty {pos.qty:.6f}, "
                      f"tp {tp_txt}, sl {fmt_price(pos.stop_loss)}")
        elif VERBOSE:
            print("  " + decision.explain().replace("\n", "\n  "))
        else:
            print(f"  {symbol}: no trade ({decision.reasons[-1].removeprefix('=> no trade: ')})")

    equity = portfolio.equity(prices)
    total = portfolio.wins + portfolio.losses
    win_rate = f"{portfolio.wins / total:.0%}" if total else "n/a"
    summary = (f"--- equity ${equity:,.2f} | cash ${portfolio.cash:,.2f} | "
               f"realized P&L {portfolio.realized_pnl:+,.2f} | "
               f"trades {total} (win rate {win_rate}) | open {len(portfolio.positions)} ---")
    log.info(summary)
    print(summary)


def main() -> None:
    """Start the trading loop."""
    log.info("Starting minittraider")
    log.info("Strategy: %s", config.STRATEGY)
    print("minittraider - paper trading with live prices (no real money at risk)")
    print(f"watching: {', '.join(s for s, _ in WATCHLIST)}")
    if config.STRATEGY == config.Strategy.SWING:
        print(f"strategy: swing - {swing_live.TIMEFRAME} trend trades, fee-aware entries, "
              f"dynamic exits (break-even, trailing stop, trend-broken, stall)")
    elif config.STRATEGY == config.Strategy.BOUNCE:
        print(f"strategy: bounce - trade rebounds off large-order zones only, "
              f"reward/risk {config.BOUNCE_MIN_RR}x+, exit at zone target/stop or "
              f"{config.BOUNCE_MAX_HOLD_MINUTES} min")
    elif config.STRATEGY == config.Strategy.COGNITIVE:
        print(f"strategy: cognitive - layered perception/planning/execution architecture, "
              f"max hold {config.COGNITIVE_MAX_HOLD_MINUTES} min")
    else:
        print(f"strategy: confluence - enter only when {config.MIN_AGREEING_SIGNALS}+ signals "
              f"align at {config.MIN_CONFIDENCE:.0%}+ confidence; exit at TP/SL or "
              f"{config.MAX_HOLD_MINUTES} min")
    portfolio = Portfolio()
    try:
        while True:
            try:
                scan(portfolio)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log.exception("Scan failed")
                print(f"scan failed: {e}")
            if ONCE:
                break
            time.sleep(config.SCAN_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        log.info("Stopped by user")
        print("\nstopped. state saved in paper_state.json, trade log in trades.csv")
    except Exception:
        log.exception("Fatal error")
        raise


if __name__ == "__main__":
    main()
