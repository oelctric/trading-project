"""Feasibility: do our setups have an edge on 24/7 crypto, after fees?

Alpaca crypto never closes — appealing (no gaps, always-on). But our system
is stock/session-specific (VIX, session VWAP, ORB, gaps don't apply). This
tests only the PRICE-BASED setups (momentum, breakout, reversal, pullback +
short mirrors) on hourly crypto bars, with triple-barrier bracket labels,
and reports the per-setup edge at several ROUND-TRIP cost levels — Alpaca
crypto fees are ~15-25 bps PER SIDE, so 30-50 bps per trade.

Usage: python crypto_feasibility.py
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


COINS = ["BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD", "LINK/USD",
         "AVAX/USD", "DOGE/USD", "UNI/USD", "AAVE/USD", "DOT/USD"]
TP, SL, H = 1.5, 1.0, 8


def bars(client, symbols, days=720):
    end = datetime.now(timezone.utc) - timedelta(minutes=30)
    req = CryptoBarsRequest(symbol_or_symbols=symbols,
                            timeframe=TimeFrame(1, TimeFrameUnit.Hour),
                            start=end - timedelta(days=days), end=end)
    return client.get_crypto_bars(req).df


def rsi(close, n=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def features(df):
    o = df.copy()
    c, h, lo, v = o["close"], o["high"], o["low"], o["volume"]
    o["vol_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    o["rsi"] = rsi(c)
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()],
                   axis=1).max(axis=1)
    o["atr_pct"] = tr.ewm(alpha=1/14, adjust=False).mean() / c
    o["sma20"] = c.rolling(20).mean()
    o["sma50"] = c.rolling(50).mean()
    o["ret3"] = c.pct_change(3)
    # triple-barrier long/short returns
    tp_d, sl_d = TP * o["atr_pct"], SL * o["atr_pct"]
    ftp = pd.Series(np.inf, index=o.index); fsl = pd.Series(np.inf, index=o.index)
    ftp_s = pd.Series(np.inf, index=o.index); fsl_s = pd.Series(np.inf, index=o.index)
    for k in range(1, H + 1):
        ftp = ftp.mask((h.shift(-k) >= c*(1+tp_d)) & np.isinf(ftp), k)
        fsl = fsl.mask((lo.shift(-k) <= c*(1-sl_d)) & np.isinf(fsl), k)
        ftp_s = ftp_s.mask((lo.shift(-k) <= c*(1-tp_d)) & np.isinf(ftp_s), k)
        fsl_s = fsl_s.mask((h.shift(-k) >= c*(1+sl_d)) & np.isinf(fsl_s), k)
    hz = c.shift(-H) / c - 1
    o["rl"] = np.where(np.isfinite(fsl) & (fsl <= ftp), -sl_d,
                       np.where(np.isfinite(ftp), tp_d, hz))
    o["rs"] = np.where(np.isfinite(fsl_s) & (fsl_s <= ftp_s), -sl_d,
                       np.where(np.isfinite(ftp_s), tp_d, -hz))
    o.loc[o.index[-H:], ["rl", "rs"]] = np.nan
    return o


def setups(o):
    return {
        "momentum": (o["ret3"] > o["ret3"].rolling(200, min_periods=50).quantile(0.9)) & (o["vol_z"] > 0.5),
        "breakout": (o["close"] > o["high"].rolling(20).max().shift(1)) & (o["vol_z"] > 0),
        "reversal": (rsi(o["close"]).shift(1) < 30) & (rsi(o["close"]) > rsi(o["close"]).shift(1)),
        "pullback": (o["close"] > o["sma50"]) & (o["sma20"] > o["sma50"]) & (o["close"].div(o["sma20"]).sub(1).rolling(3).min() <= 0) & (o["close"] > o["open"]),
        "momentum_dn": (o["ret3"] < o["ret3"].rolling(200, min_periods=50).quantile(0.1)) & (o["vol_z"] > 0.5),
        "breakdown": (o["close"] < o["low"].rolling(20).min().shift(1)) & (o["vol_z"] > 0),
    }


def main():
    client = CryptoHistoricalDataClient()
    df = bars(client, COINS)
    have = df.index.get_level_values(0).unique().tolist()
    print(f"Loaded {len(have)} coins, {len(df)} hourly bars total\n")

    rows = []
    for coin in have:
        o = features(df.xs(coin))
        for name, mask in setups(o).items():
            short = name.endswith("_dn") or name == "breakdown"
            rcol = "rs" if short else "rl"
            m = mask.fillna(False) & o[rcol].notna()
            for r in o.loc[m, rcol].values:
                rows.append((name, r))
    d = pd.DataFrame(rows, columns=["setup", "ret"])
    print(f"{len(d)} setup-trades across crypto\n")

    # ROUND-TRIP costs (fees are charged on BOTH the entry and the exit):
    # Alpaca crypto ~0.15% maker / ~0.25% taker per side -> 30-50 bps per trade.
    labels = {5: "stock-like", 15: "one maker side only (too optimistic)",
              30: "maker in + maker out", 40: "maker in + market out (live bot)",
              50: "market in + market out"}
    for cost_bps in (5, 15, 30, 40, 50):
        cost = cost_bps / 1e4
        print(f"=== per-trade edge at {cost_bps} bps round-trip cost ({labels[cost_bps]}) ===")
        for name in d["setup"].unique():
            r = d.loc[d["setup"] == name, "ret"].values - cost
            print(f"  {name:12s} trades={len(r):6d}  avg={r.mean()*1e4:+6.1f} bps"
                  f"  hit={(r>0).mean():.3f}")
        allr = d["ret"].values - cost
        print(f"  {'ALL':12s} trades={len(allr):6d}  avg={allr.mean()*1e4:+6.1f} bps\n")


if __name__ == "__main__":
    main()
