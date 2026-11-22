"""Live 1-minute market data.

Crypto: Coinbase Exchange public API (no account or key needed).
Stocks: Yahoo Finance via yfinance (free, ~1 minute delayed - fine for paper trading).

Both are normalized to the same DataFrame shape:
    index: UTC timestamps, columns: open, high, low, close, volume
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

import config
from logger import get_logger
from signals import atr

log = get_logger("data")

@dataclass
class ApiResponse:
    """Response with data and metadata."""
    data: Optional[pd.DataFrame] = None
    success: bool = False
    error: Optional[str] = None
    age_seconds: int = 0


_session = requests.Session()
_session.headers["User-Agent"] = "minittraider/1.0"

# Track API failures to alert users
_api_failure_count: dict[str, int] = {}
_last_alert_time: dict[str, float] = {}


def _track_failure(source: str, error: str, max_failures: int = 5, alert_interval: float = 300) -> None:
    """Track API failures and log alerts."""
    _api_failure_count[source] = _api_failure_count.get(source, 0) + 1
    failures = _api_failure_count[source]

    log.debug("API error from %s: %s", source, error)

    # Alert user periodically on repeated failures
    now = time.time()
    if failures % max_failures == 0 or failures == 1:
        log.warning("API failure from %s (failure #%d): %s", source, failures, error)
        _last_alert_time[source] = now
    elif now - _last_alert_time.get(source, 0) > alert_interval:
        log.warning("API %s has had %d total failures (last: %s)", source, failures, error)
        _last_alert_time[source] = now


def _reset_failure_tracker(source: str) -> None:
    """Reset failure count on success."""
    _api_failure_count[source] = 0


def crypto_ticker(product: str) -> float | None:
    """Live last-trade price for a Coinbase product.

    Returns None if API fails (logged appropriately).
    """
    url = f"https://api.exchange.coinbase.com/products/{product}/ticker"
    try:
        r = _session.get(url, timeout=10)
        r.raise_for_status()
        _reset_failure_tracker("coinbase")
        return float(r.json()["price"])
    except Exception as e:
        _track_failure("coinbase", str(e))
        return None


def crypto_candles(product: str) -> pd.DataFrame:
    """Latest ~300 one-minute candles for a Coinbase product like 'BTC-USD'.

    Coinbase omits minutes with no trades, so the newest candle can lag a few
    minutes on quiet pairs. We patch in the live ticker price as the current
    candle so the latest price is always real-time and freshness checks pass.

    Raises:
        requests.RequestException: If API fails after tracking the failure.
    """
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    try:
        r = _session.get(url, params={"granularity": 60}, timeout=15)
        r.raise_for_status()
        _reset_failure_tracker("coinbase")
        rows = r.json()  # [[time, low, high, open, close, volume], ...] newest first
        if not rows:
            log.warning("No candles returned for %s", product)
        df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        price = crypto_ticker(product)
        if price and not df.empty:
            now = pd.Timestamp.now(tz="UTC").floor("min")
            last = df.index[-1]
            if now > last:
                prev_close = df["close"].iloc[-1]
                df.loc[now] = [prev_close, max(price, prev_close),
                               min(price, prev_close), price, 0.0]
            else:
                df.loc[last, "close"] = price
                df.loc[last, "high"] = max(df.loc[last, "high"], price)
                df.loc[last, "low"] = min(df.loc[last, "low"], price)
        log.debug("Fetched %d candles for %s (age: %.1f min)", len(df), product,
                  (pd.Timestamp.now(tz="UTC") - df.index[-1]).total_seconds() / 60)
        return df
    except Exception as e:
        _track_failure("coinbase", f"candles fetch for {product}: {e}")
        raise


def stock_candles(symbol: str) -> pd.DataFrame:
    """Today's one-minute candles for a stock ticker.

    Raises:
        Exception: If API fails after tracking the failure.
    """
    try:
        df = yf.Ticker(symbol).history(period="1d", interval="1m")
        if df.empty:
            log.debug("No data for stock %s (market may be closed)", symbol)
            return df
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index = df.index.tz_convert("UTC")
        _reset_failure_tracker("yfinance")
        log.debug("Fetched %d stock candles for %s", len(df), symbol)
        return df.astype(float)
    except Exception as e:
        _track_failure("yfinance", f"stock candles for {symbol}: {e}")
        raise


def get_candles(symbol: str, is_crypto: bool) -> pd.DataFrame:
    return crypto_candles(symbol) if is_crypto else stock_candles(symbol)


# ---------------------------------------------------------------- microstructure
# Live information that is NOT derived from past prices: what the order book and
# the trade tape look like right now. Only available for crypto, only live
# (nobody archives this for free, so it can't be backtested - it is evaluated
# by paper trading forward).

def order_book_imbalance(product: str, band: float = 0.003) -> float | None:
    """Bid vs ask depth within `band` of the mid price, in -1..+1.

    +0.4 means 40% more size waiting to buy than to sell near the price.
    """
    try:
        r = _session.get(f"https://api.exchange.coinbase.com/products/{product}/book",
                         params={"level": 2}, timeout=10)
        r.raise_for_status()
        _reset_failure_tracker("coinbase")
        book = r.json()
        bids, asks = book["bids"], book["asks"]
        mid = (float(bids[0][0]) + float(asks[0][0])) / 2
        bid_depth = sum(float(s) for p, s, *_ in bids if float(p) >= mid * (1 - band))
        ask_depth = sum(float(s) for p, s, *_ in asks if float(p) <= mid * (1 + band))
        total = bid_depth + ask_depth
        return (bid_depth - ask_depth) / total if total else None
    except Exception as e:
        _track_failure("coinbase", f"order book for {product}: {e}")
        return None


def tape_imbalance(product: str, limit: int = 200) -> float | None:
    """Aggressive buying vs selling in the most recent trades, in -1..+1.

    Coinbase reports the MAKER side: a 'sell'-side print means the taker bought.
    """
    try:
        r = _session.get(f"https://api.exchange.coinbase.com/products/{product}/trades",
                         params={"limit": limit}, timeout=10)
        r.raise_for_status()
        _reset_failure_tracker("coinbase")
        trades = r.json()
        taker_buys = sum(float(t["size"]) for t in trades if t["side"] == "sell")
        taker_sells = sum(float(t["size"]) for t in trades if t["side"] == "buy")
        total = taker_buys + taker_sells
        return (taker_buys - taker_sells) / total if total else None
    except Exception as e:
        _track_failure("coinbase", f"tape imbalance for {product}: {e}")
        return None


def order_book_walls(product: str, ratio: float | None = None,
                     span: float = 0.02, bins: int = 40) -> dict | None:
    """Large resting orders ('walls') in the live book - the bounce zones.

    Levels within `span` (2%) of the mid are grouped into `bins` price buckets;
    a wall is a bucket holding at least `ratio` times the median bucket size.
    Returns {'supports': [(price, strength)], 'resists': [(price, strength)]}
    sorted nearest-to-price first.
    """
    if ratio is None:
        ratio = config.BOUNCE_WALL_RATIO
    try:
        r = _session.get(f"https://api.exchange.coinbase.com/products/{product}/book",
                         params={"level": 2}, timeout=10)
        r.raise_for_status()
        _reset_failure_tracker("coinbase")
        book = r.json()
        mid = (float(book["bids"][0][0]) + float(book["asks"][0][0])) / 2
        width = mid * span / bins

        def walls(levels, reverse):
            buckets: dict[int, float] = {}
            for p, s, *_ in levels:
                p = float(p)
                if abs(p - mid) > mid * span:
                    continue
                buckets[int(p / width)] = buckets.get(int(p / width), 0.0) + float(s)
            if len(buckets) < 5:
                return []
            sizes = sorted(buckets.values())
            med = sizes[len(sizes) // 2] or 1e-12
            found = [((b + 0.5) * width, size / med)
                     for b, size in buckets.items() if size / med >= ratio]
            return sorted(found, reverse=reverse)

        result = {"supports": walls(book["bids"], reverse=True),
                  "resists": walls(book["asks"], reverse=False)}
        log.debug("Order book walls for %s: %d supports, %d resistances",
                  product, len(result["supports"]), len(result["resists"]))
        return result
    except Exception as e:
        _track_failure("coinbase", f"order book walls for {product}: {e}")
        return None


def fetch_micro(symbol: str, is_crypto: bool,
                leader_df: pd.DataFrame | None = None, leader_name: str = "") -> dict:
    """Everything the brain gets to see beyond this symbol's own candles."""
    micro: dict = {}
    if is_crypto:
        micro["book"] = order_book_imbalance(symbol)
        micro["tape"] = tape_imbalance(symbol)
    if leader_df is not None and len(leader_df) > 20:
        a = float(atr(leader_df).iloc[-1])
        if a > 0:
            micro["leader"] = float(
                (leader_df["close"].iloc[-1] - leader_df["close"].iloc[-4]) / a)
            micro["leader_name"] = leader_name
    return micro
