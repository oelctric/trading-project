"""Independent signals computed from 1-minute candles.

Each signal returns a score from -1.0 (strongly bearish) to +1.0 (strongly bullish),
plus a short plain-English note. A score near 0 means "this signal sees nothing".

The signals are deliberately different in nature (trend, momentum, volume, value,
structure) so that when several of them agree, it genuinely means independent
evidence is lining up - which is exactly when short scalps are most predictable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clip(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ---------------------------------------------------------------- signals

def sig_trend(df: pd.DataFrame) -> tuple[float, str]:
    """EMA stack: are the fast, medium and slow averages lined up?"""
    close = df["close"]
    e9, e21, e50 = ema(close, 9).iloc[-1], ema(close, 21).iloc[-1], ema(close, 50).iloc[-1]
    if e9 > e21 > e50:
        return 1.0, "trend up (EMAs stacked bullish)"
    if e9 < e21 < e50:
        return -1.0, "trend down (EMAs stacked bearish)"
    score = _clip((e9 - e21) / max(atr(df).iloc[-1], 1e-9) * 0.5)
    return score, "trend mixed"


def sig_momentum(df: pd.DataFrame) -> tuple[float, str]:
    """Price change over the last 3 minutes, measured in units of normal candle range."""
    a = atr(df).iloc[-1]
    move = df["close"].iloc[-1] - df["close"].iloc[-4]
    score = _clip(move / max(a * 3, 1e-9))
    if abs(score) < 0.15:
        return score, "no momentum"
    return score, f"momentum {'up' if score > 0 else 'down'} ({move / max(a, 1e-9):+.1f} ATRs over 3 min)"


def sig_rsi(df: pd.DataFrame) -> tuple[float, str]:
    """RSI direction, but fades near exhaustion so we don't chase a spent move."""
    r = rsi(df["close"]).iloc[-1]
    if np.isnan(r):
        return 0.0, "rsi unavailable"
    score = _clip((r - 50) / 25)
    if r > 75:
        return _clip(score - 0.8), f"RSI {r:.0f}: overbought, upside likely spent"
    if r < 25:
        return _clip(score + 0.8), f"RSI {r:.0f}: oversold, downside likely spent"
    return score, f"RSI {r:.0f}"


def sig_volume(df: pd.DataFrame) -> tuple[float, str]:
    """Volume surge in the direction of the last candles = real participation."""
    vol = df["volume"]
    avg = vol.iloc[-21:-1].mean()
    if not avg or np.isnan(avg):
        return 0.0, "no volume data"
    surge = vol.iloc[-1] / avg
    direction = np.sign(df["close"].iloc[-1] - df["open"].iloc[-1])
    if surge < 1.3:
        return 0.0, "normal volume"
    score = _clip(direction * min((surge - 1.0), 2.0) / 2.0)
    return score, f"volume surge x{surge:.1f} on a {'green' if direction > 0 else 'red'} candle"


def sig_vwap(df: pd.DataFrame) -> tuple[float, str]:
    """Where price sits relative to the volume-weighted average price of the session."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, np.nan)
    vwap = ((typical * vol).cumsum() / vol.cumsum()).ffill()  # zero-volume bars carry the last value
    dist = (df["close"].iloc[-1] - vwap.iloc[-1]) / max(atr(df).iloc[-1], 1e-9)
    if np.isnan(dist):
        return 0.0, "vwap unavailable"
    score = _clip(dist / 2.0)
    side = "above" if dist > 0 else "below"
    return score, f"price {abs(dist):.1f} ATRs {side} VWAP"


def sig_structure(df: pd.DataFrame) -> tuple[float, str]:
    """Consecutive closes in one direction = orderly move, not noise."""
    closes = df["close"].iloc[-5:]
    diffs = np.sign(closes.diff().dropna())
    net = diffs.sum()
    score = _clip(net / 4)
    if abs(net) >= 3:
        return score, f"{int(abs(net))} of last 4 candles pushed {'up' if net > 0 else 'down'}"
    return score * 0.5, "choppy candle structure"


ALL_SIGNALS = {
    "trend": sig_trend,
    "momentum": sig_momentum,
    "rsi": sig_rsi,
    "volume": sig_volume,
    "vwap": sig_vwap,
    "structure": sig_structure,
}


# ---------------------------------------------------------------- regime

def volatility_regime(df: pd.DataFrame) -> tuple[str, float, str]:
    """Classify the current environment: is this even a market worth scalping?

    Returns (regime_name, confidence_multiplier, note).
    Compares the current ATR to its own recent history, so it adapts per symbol.
    """
    a = atr(df)
    recent = a.iloc[-1]
    history = a.iloc[-120:].dropna()
    if len(history) < 30 or recent <= 0:
        return "unknown", 0.0, "not enough history to judge the environment"
    pct = float((history < recent).mean())  # percentile of current volatility
    if pct < 0.20:
        return "dead", 0.3, f"volatility very low ({pct:.0%} pct): market is asleep, moves won't follow through"
    if pct > 0.92:
        return "wild", 0.4, f"volatility extreme ({pct:.0%} pct): too erratic, hard to predict anything"
    if pct > 0.60:
        return "active", 1.0, f"volatility healthy ({pct:.0%} pct): moves have energy"
    return "normal", 0.85, f"volatility normal ({pct:.0%} pct)"
