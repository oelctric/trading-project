"""Setup detectors for eight trading styles: four long, four short mirrors.

Each detector returns a boolean Series aligned to the feature frame: True on
bars where that setup fires. These become both model inputs and trade
triggers — the live bot enters when a setup fires, ranked by the neural
network and gated by recent regime performance.
"""
from __future__ import annotations

import pandas as pd

import config

LONG_STRATEGIES = ["momentum", "breakout", "reversal", "pullback", "orb",
                   "gap_go", "gap_fade",
                   "disc_strongday_dip", "disc_orb_wide"]
SHORT_STRATEGIES = ["momentum_dn", "breakdown", "reversal_dn",
                    "pullback_dn", "orb_dn", "gap_go_dn", "gap_fade_dn"]
STRATEGY_NAMES = LONG_STRATEGIES + SHORT_STRATEGIES
STRATEGY_COLUMNS = [f"setup_{s}" for s in STRATEGY_NAMES]


# ---- long side ----

def momentum(df: pd.DataFrame) -> pd.Series:
    """Strong directional push: 3-bar return in top decile with volume surge."""
    ret3 = df["close"].pct_change(3)
    strong = ret3 > ret3.rolling(200, min_periods=50).quantile(0.90)
    return (strong & (df["vol_z"] > 0.5)).fillna(False)


def breakout(df: pd.DataFrame) -> pd.Series:
    """Close above the prior 20-bar high on above-average volume."""
    prior_high = df["high"].rolling(20).max().shift(1)
    return ((df["close"] > prior_high) & (df["vol_z"] > 0)).fillna(False)


def reversal(df: pd.DataFrame) -> pd.Series:
    """Oversold bounce: RSI was under 30 and has turned up."""
    rsi = df["rsi"] * 100
    return ((rsi.shift(1) < 30) & (rsi > rsi.shift(1))).fillna(False)


def pullback(df: pd.DataFrame) -> pd.Series:
    """Dip-buy in an uptrend: price above SMA50, SMA20 above SMA50, price
    touched/undercut SMA20 within 3 bars, current bar green."""
    uptrend = (df["sma50_dist"] > 0) & (df["sma50_dist"] > df["sma20_dist"])
    touched = df["sma20_dist"].rolling(3).min() <= 0.0
    green = df["body_pct"] > 0
    return (uptrend & touched & green).fillna(False)


def orb(df: pd.DataFrame) -> pd.Series:
    """Opening range breakout: close crosses above the first bar's high
    on above-average volume."""
    above = df["orh_dist"] > 0
    return (above & ~above.shift(1, fill_value=False)
            & (df["vol_z"] > 0)).fillna(False)


def _early(df: pd.DataFrame) -> pd.Series:
    """Gap plays run in the first ~90 minutes: the 9:00 and 10:00 bars."""
    return pd.Series(df.index.hour <= 10, index=df.index)


def gap_go(df: pd.DataFrame) -> pd.Series:
    """Gap-and-go long: gapped up >=1% and holding above the open."""
    return ((df["gap"] >= 0.01) & (df["open_dist"] > 0)
            & _early(df)).fillna(False)


def gap_fade(df: pd.DataFrame) -> pd.Series:
    """Gap-fade long: gapped down >=1% but buyers pushed it back above
    the open (the gap is filling)."""
    return ((df["gap"] <= -0.01) & (df["open_dist"] > 0)
            & _early(df)).fillna(False)


# ---- machine-discovered (discover.py survivors; train +11.0/+10.8 bps,
# val +9.7/+8.7, test +17.1/+16.2 — the regime gate supervises them like
# every other strategy and benches them if they decay live) ----

def disc_strongday_dip(df: pd.DataFrame) -> pd.Series:
    """Buy-the-dip on a strong day: >1.3% above yesterday's close but
    sitting at/below the 20-bar SMA."""
    return ((df["pdc_dist"] > 0.0130)
            & (df["sma20_dist"] < 0.0010)).fillna(False)


def disc_orb_wide(df: pd.DataFrame) -> pd.Series:
    """Above the opening-range high while bars are unusually wide-ranged."""
    return ((df["orh_dist"] > 0.0010)
            & (df["range_pct"] > 0.0136)).fillna(False)


# ---- short side (mirrors) ----

def momentum_dn(df: pd.DataFrame) -> pd.Series:
    """Strong downside push: 3-bar return in bottom decile with volume surge."""
    ret3 = df["close"].pct_change(3)
    weak = ret3 < ret3.rolling(200, min_periods=50).quantile(0.10)
    return (weak & (df["vol_z"] > 0.5)).fillna(False)


def breakdown(df: pd.DataFrame) -> pd.Series:
    """Close below the prior 20-bar low on above-average volume."""
    prior_low = df["low"].rolling(20).min().shift(1)
    return ((df["close"] < prior_low) & (df["vol_z"] > 0)).fillna(False)


def reversal_dn(df: pd.DataFrame) -> pd.Series:
    """Overbought fade: RSI was above 70 and has turned down."""
    rsi = df["rsi"] * 100
    return ((rsi.shift(1) > 70) & (rsi < rsi.shift(1))).fillna(False)


def pullback_dn(df: pd.DataFrame) -> pd.Series:
    """Rally-sell in a downtrend: price below SMA50, SMA20 below SMA50,
    price touched/overshot SMA20 within 3 bars, current bar red."""
    downtrend = (df["sma50_dist"] < 0) & (df["sma50_dist"] < df["sma20_dist"])
    touched = df["sma20_dist"].rolling(3).max() >= 0.0
    red = df["body_pct"] < 0
    return (downtrend & touched & red).fillna(False)


def orb_dn(df: pd.DataFrame) -> pd.Series:
    """Opening range breakdown: close crosses below the first bar's low
    on above-average volume."""
    below = df["orl_dist"] < 0
    return (below & ~below.shift(1, fill_value=False)
            & (df["vol_z"] > 0)).fillna(False)


def gap_go_dn(df: pd.DataFrame) -> pd.Series:
    """Gap-and-go short: gapped down >=1% and still selling below the open."""
    return ((df["gap"] <= -0.01) & (df["open_dist"] < 0)
            & _early(df)).fillna(False)


def gap_fade_dn(df: pd.DataFrame) -> pd.Series:
    """Gap-fade short: gapped up >=1% but failing back below the open."""
    return ((df["gap"] >= 0.01) & (df["open_dist"] < 0)
            & _early(df)).fillna(False)


_DETECTORS = {
    "momentum": momentum, "breakout": breakout,
    "reversal": reversal, "pullback": pullback, "orb": orb,
    "gap_go": gap_go, "gap_fade": gap_fade,
    "disc_strongday_dip": disc_strongday_dip, "disc_orb_wide": disc_orb_wide,
    "momentum_dn": momentum_dn, "breakdown": breakdown,
    "reversal_dn": reversal_dn, "pullback_dn": pullback_dn,
    "orb_dn": orb_dn, "gap_go_dn": gap_go_dn, "gap_fade_dn": gap_fade_dn,
}


def add_setups(df: pd.DataFrame) -> pd.DataFrame:
    """Append the setup columns (as 0/1 floats) to a feature frame.

    With VWAP_FILTER on, the pro-desk rule applies: longs only above the
    session VWAP, shorts only below it.
    """
    out = df.copy()
    long_ok = short_ok = pd.Series(True, index=out.index)
    if config.VWAP_FILTER and "vwap_dist" in out.columns:
        long_ok = long_ok & (out["vwap_dist"] >= 0)
        short_ok = short_ok & (out["vwap_dist"] <= 0)
    if config.MARKET_FILTER and "mkt_above_vwap" in out.columns:
        # "Check the macro before trading micro": longs need SPY above its
        # own session VWAP, shorts need SPY below it.
        long_ok = long_ok & (out["mkt_above_vwap"] > 0)
        short_ok = short_ok & (out["mkt_above_vwap"] < 1)
    for name, fn in _DETECTORS.items():
        fired = fn(out)
        fired = fired & (short_ok if name in SHORT_STRATEGIES else long_ok)
        out[f"setup_{name}"] = fired.astype("float32")
    return out


def is_short(strategy: str) -> bool:
    return strategy in SHORT_STRATEGIES
