"""24/7 crypto mean-reversion paper bot (Alpaca crypto).

HONEST STATUS: this is a speculative forward experiment, not a validated
edge. Backtesting (crypto_validate.py) showed the reversal strategy's
apparent profit is dominated by two outlier months; even with the regime
gate it is fragile and outlier-dependent. It is built to run on PAPER so
forward results can tell us whether the edge is real. Do not point it at a
funded account.

Strategy: long-only oversold-RSI bounce (the one crypto setup with raw
edge), on liquid coins, entered with maker limit orders, exited with a
1-ATR trailing stop managed in-loop (Alpaca crypto has no resting brackets).
Traded only while the strategy's trailing-30-day pooled performance is
positive (regime gate). Per-trade risk sizing and a daily loss limit apply.

Usage: python crypto_trade.py            (loop)
       python crypto_trade.py --once     (single pass)
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

import config

COINS = ["BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD", "LINK/USD",
         "AVAX/USD", "DOGE/USD", "UNI/USD", "AAVE/USD", "DOT/USD",
         "BCH/USD", "XRP/USD"]
MAX_POSITIONS = 4
RISK_PER_TRADE = 0.0025
MAX_NOTIONAL_FRACTION = 0.10   # cap any one coin at 10% of equity (fragile
                               # single-strategy book — was 0.25, too large)
STALE_ORDER_MIN = 30           # cancel unfilled maker limits after 30 min
TRAIL_ATR = 1.0
SL_ATR = 1.0
MAKER_BUFFER = 0.0010        # limit this far below price for a maker fill
DAILY_LOSS_LIMIT = 0.03      # crypto is volatile; wider than the 1% stock cap
SCAN_SECONDS = 900           # manage exits every 15 min; enter hourly
REGIME_LOOKBACK_DAYS = 30

_data = CryptoHistoricalDataClient()
_trail: dict[str, dict] = {}   # symbol -> {dist, hwm, atr}


def norm(sym: str) -> str:     # position symbol "BTCUSD" -> "BTC/USD"
    return sym if "/" in sym else sym[:-3] + "/" + sym[-3:]


# The account is SHARED with the stock bot. These filters ensure the crypto
# bot only ever touches its OWN positions/orders (crypto symbols), never the
# stock bot's — a hard lesson: it was cancelling the stock bot's stop orders.
CRYPTO_POS_SYMS = {x.replace("/", "") for x in COINS}   # e.g. "BTCUSD"


def is_crypto_position(sym: str) -> bool:
    return sym in CRYPTO_POS_SYMS


def is_crypto_order(sym: str) -> bool:
    return "/" in sym   # crypto orders are submitted as "BTC/USD"


def get_client() -> TradingClient:
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        sys.exit("Missing Alpaca keys in .env")
    return TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
                         paper=True)


def _rsi(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def load_bars(days=75) -> dict[str, pd.DataFrame]:  # need 30d for vol median
    end = datetime.now(timezone.utc) - timedelta(minutes=5)
    req = CryptoBarsRequest(symbol_or_symbols=COINS,
                            timeframe=TimeFrame(1, TimeFrameUnit.Hour),
                            start=end - timedelta(days=days), end=end)
    df = _data.get_crypto_bars(req).df
    out = {}
    for coin in df.index.get_level_values(0).unique():
        o = df.xs(coin).copy()
        c, h, lo = o["close"], o["high"], o["low"]
        tr = pd.concat([h - lo, (h - c.shift()).abs(),
                        (lo - c.shift()).abs()], axis=1).max(axis=1)
        o["atr"] = tr.ewm(alpha=1/14, adjust=False).mean() / c
        r = _rsi(c)
        o["reversal"] = ((r.shift(1) < 30) & (r > r.shift(1))).fillna(False)
        # Volatility-regime filter: reversal only pays in elevated vol
        # (validated: doubles per-trade edge, less month-concentration).
        vol = c.pct_change().rolling(48).std()
        o["vol_hi"] = (vol > vol.rolling(24 * 30).median()).fillna(False)
        out[coin] = o
    return out


def regime_ok(bars) -> bool:
    """Gate: pooled trailing-30d reversal (trailing-exit proxy) net positive."""
    pnl, n = 0.0, 0
    for o in bars.values():
        recent = o.tail(REGIME_LOOKBACK_DAYS * 24)
        fired = recent[recent["reversal"]]
        # proxy outcome: next-8-bar return minus maker cost
        for ts in fired.index:
            loc = o.index.get_loc(ts)
            if loc + 8 < len(o):
                r = o["close"].iloc[loc + 8] / o["close"].iloc[loc] - 1
                pnl += r - 0.0015
                n += 1
    print(f"  regime pnl(30d): {pnl:+.3f} over {n} signals -> "
          f"{'TRADE' if pnl > 0 else 'SIT OUT'}")
    return pnl > 0


def manage_exits(client, bars):
    for p in client.get_all_positions():
        if not is_crypto_position(p.symbol):
            continue   # NEVER touch the stock bot's positions
        sym = norm(p.symbol)
        price = float(p.current_price)
        st = _trail.get(sym)
        if st is None:
            atr = float(bars[sym]["atr"].iloc[-1]) if sym in bars else 0.02
            st = {"dist": TRAIL_ATR * atr * price, "hwm": price}
            _trail[sym] = st
        st["hwm"] = max(st["hwm"], price)
        stop = st["hwm"] - st["dist"]
        if price <= stop:
            print(f"  trailing stop hit {sym} @ {price:.4f} (stop {stop:.4f})"
                  f" -> selling")
            try:
                client.submit_order(MarketOrderRequest(
                    symbol=sym, qty=abs(float(p.qty)), side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC))
                _trail.pop(sym, None)
            except Exception as e:
                print(f"  sell failed {sym}: {e}")


def cancel_stale_orders(client):
    """Cancel unfilled maker limits older than STALE_ORDER_MIN so they don't
    pile up (crypto has no DAY expiry; GTC orders sit forever otherwise)."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    now = datetime.now(timezone.utc)
    try:
        for o in client.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.OPEN, limit=50)):
            if not is_crypto_order(o.symbol):
                continue   # NEVER cancel the stock bot's orders
            age = (now - o.submitted_at).total_seconds() / 60
            if age > STALE_ORDER_MIN:
                client.cancel_order_by_id(o.id)
                print(f"  cancelled stale order {o.symbol} (age {age:.0f}m)")
    except Exception as e:
        print(f"  stale-order cleanup failed: {e}")


def open_order_symbols(client) -> set:
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    try:
        return {norm(o.symbol) for o in client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=50))
            if is_crypto_order(o.symbol)}
    except Exception:
        return set()


def enter(client, bars):
    account = client.get_account()
    equity = float(account.equity)
    # Count BOTH filled positions AND pending orders toward the limit so
    # unfilled maker limits don't let the bot re-order and over-commit.
    held = {norm(p.symbol) for p in client.get_all_positions()
            if is_crypto_position(p.symbol)}
    held |= open_order_symbols(client)
    slots = MAX_POSITIONS - len(held)
    if slots <= 0:
        print("  no free slots (positions + pending orders)"); return
    cands = []
    for coin, o in bars.items():
        if coin in held or not bool(o["reversal"].iloc[-1]):
            continue
        if not bool(o["vol_hi"].iloc[-1]):
            continue  # volatility-regime filter
        cands.append((coin, float(o["close"].iloc[-1]),
                      float(o["atr"].iloc[-1])))
    print(f"  reversal signals (high-vol): {[c for c, _, _ in cands] or 'none'}")
    for coin, price, atr in cands[:slots]:
        stop_dist = SL_ATR * atr * price
        if stop_dist <= 0:
            continue
        notional = min(equity * RISK_PER_TRADE / (SL_ATR * atr),
                       equity * MAX_NOTIONAL_FRACTION)
        qty = round(notional / price, 6)
        if qty * price < 1:
            continue
        limit = round(price * (1 - MAKER_BUFFER), 2)
        try:
            client.submit_order(LimitOrderRequest(
                symbol=coin, qty=qty, side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC, limit_price=limit))
            _trail[coin] = {"dist": TRAIL_ATR * atr * price, "hwm": price}
            print(f"  BUY {qty} {coin} lmt={limit} (~${notional:,.0f})")
        except Exception as e:
            print(f"  buy failed {coin}: {e}")


def daily_loss_hit(client) -> bool:
    a = client.get_account()
    eq, last = float(a.equity), float(a.last_equity)
    if last > 0 and eq < last * (1 - DAILY_LOSS_LIMIT):
        print(f"  DAILY LOSS LIMIT: {eq:.0f} vs {last:.0f} -- pausing entries")
        return True
    return False


def main():
    once = "--once" in sys.argv
    client = get_client()
    print(f"Crypto paper bot connected -- equity "
          f"${float(client.get_account().equity):,.2f}  "
          f"(SPECULATIVE: backtest edge is fragile; paper only)")
    last_hour = None
    while True:
        now = datetime.now(timezone.utc)
        try:
            bars = load_bars()
            cancel_stale_orders(client)
            manage_exits(client, bars)
            new_hour = now.replace(minute=0, second=0, microsecond=0)
            if new_hour != last_hour:
                print(f"\n[{now:%Y-%m-%d %H:%M UTC}] hourly scan")
                if not daily_loss_hit(client) and regime_ok(bars):
                    enter(client, bars)
                last_hour = new_hour
        except Exception as e:
            print(f"  loop error: {e}")
        if once:
            return
        time.sleep(SCAN_SECONDS)


if __name__ == "__main__":
    main()
