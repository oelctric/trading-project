"""Historical candle downloader + cache for any bar size.

Coinbase public API serves 300 candles per request at granularities of
1m, 5m, 15m, 1h, 6h, 1d. Yahoo serves hourly stock bars for ~730 days.

    from history import load
    df = load("BTC-USD", "1h", days=1095)

Cached as backtest_data/<symbol>_<tf>_<days>d.csv (refreshed after 12h).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(__file__), "backtest_data")
GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}

_session = requests.Session()
_session.headers["User-Agent"] = "minittraider-history/1.0"


def is_crypto(symbol: str) -> bool:
    return symbol.endswith("-USD")


def fetch_coinbase(product: str, tf: str, days: int) -> pd.DataFrame:
    gran = GRANULARITY[tf]
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)
    step = timedelta(seconds=gran * 300)
    frames, cursor = [], start
    while cursor < end:
        chunk_end = min(cursor + step, end)
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                r = _session.get(
                    f"https://api.exchange.coinbase.com/products/{product}/candles",
                    params={"granularity": gran, "start": cursor.isoformat(),
                            "end": chunk_end.isoformat()},
                    timeout=20,
                )
                if r.status_code == 429:
                    last_err = RuntimeError("rate limited (429)")
                    time.sleep(1.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                rows = r.json()
                break
            except (requests.RequestException, ValueError) as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        else:
            # a missing chunk would leave a silent hole in the history (broken
            # indicators, fake gaps) - fail loudly instead of returning bad data
            raise RuntimeError(f"{product} {tf}: chunk {cursor:%Y-%m-%d %H:%M} failed after "
                               f"5 attempts ({last_err})")
        if rows:
            frames.append(pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"]))
        cursor = chunk_end
        time.sleep(0.12)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop_duplicates("time").set_index("time").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def fetch_yahoo(symbol: str, tf: str, days: int) -> pd.DataFrame:
    import yfinance as yf
    interval = {"1h": "60m", "15m": "15m", "5m": "5m", "1d": "1d"}[tf]
    max_days = {"60m": 729, "15m": 59, "5m": 59, "1d": 3650}[interval]
    df = yf.Ticker(symbol).history(period=f"{min(days, max_days)}d", interval=interval)
    if df.empty:
        return df
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = df.index.tz_convert("UTC")
    return df[~df.index.duplicated()].sort_index().astype(float)


RESAMPLED = {"2h": ("1h", "2h"), "4h": ("1h", "4h"), "8h": ("1h", "8h"), "12h": ("1h", "12h")}


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule, origin="epoch").agg(agg).dropna()


def load(symbol: str, tf: str = "1h", days: int = 365, max_age_hours: float = 12) -> pd.DataFrame:
    if tf in RESAMPLED:
        base, rule = RESAMPLED[tf]
        return resample(load(symbol, base, days, max_age_hours), rule)
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{symbol}_{tf}_{days}d.csv")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_hours * 3600:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = fetch_coinbase(symbol, tf, days) if is_crypto(symbol) else fetch_yahoo(symbol, tf, days)
    if not df.empty:
        df.to_csv(path)
    return df


if __name__ == "__main__":
    import sys
    tf = sys.argv[1] if len(sys.argv) > 1 else "1h"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
    syms = sys.argv[3:] or ["BTC-USD", "ETH-USD", "SOL-USD"]
    for s in syms:
        t0 = time.time()
        d = load(s, tf, days)
        print(f"{s} {tf}: {len(d):,} bars {d.index[0] if len(d) else ''} -> "
              f"{d.index[-1] if len(d) else ''} ({time.time() - t0:.0f}s)", flush=True)
