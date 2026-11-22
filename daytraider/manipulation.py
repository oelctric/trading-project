"""Detect likely market-manipulation / liquidity-games signatures.

Manipulators (spoofers, stop-hunters) prey on retail traders exactly at the
obvious levels where our breakout/ORB setups fire. This module flags the
fingerprints those tactics leave in the data we can see, so the live bot can
stand aside instead of being the liquidity that gets run.

Two layers:
  * bar-level (works on history and live): stop-hunt wicks, absorption,
    fake breakouts, volume/price divergence — all computable from OHLCV.
  * quote-level (live only): a resting quote wall that vanishes between
    checks with the imbalance flipping = classic spoof/pull.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bar_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar manipulation signatures from an OHLCV+feature frame.

    Expects the columns compute_features() produces (atr_pct, vol_z,
    pdh_dist, pdl_dist, orh_dist, orl_dist, upper_wick, lower_wick, ...).
    Returns a frame of 0/1 flags plus a combined 'manip_score'.
    """
    out = pd.DataFrame(index=df.index)
    close, high, low = df["close"], df["high"], df["low"]
    rng = (high - low).replace(0, np.nan)
    body = (close - df["open"]).abs()

    # Stop-hunt / liquidity grab: price pokes through yesterday's high (or
    # low) or the opening-range extreme, then closes back inside on a long
    # rejection wick. The stops beyond the level were the target.
    poked_high = ((df["pdh_dist"].shift(1) < 0) & (high > df["high"].shift(1))
                  & (close < df["high"].shift(1)))
    long_upper = df["upper_wick"] * close > 0.6 * rng
    out["stophunt_high"] = (poked_high & long_upper).fillna(False).astype(int)
    poked_low = ((df["pdl_dist"].shift(1) > 0) & (low < df["low"].shift(1))
                 & (close > df["low"].shift(1)))
    long_lower = df["lower_wick"] * close > 0.6 * rng
    out["stophunt_low"] = (poked_low & long_lower).fillna(False).astype(int)

    # Absorption: very high relative volume but tiny price result — a large
    # passive player is eating everything without letting price move.
    out["absorption"] = ((df["vol_z"] > 2.0)
                         & (body < 0.2 * df["atr_pct"] * close)
                         ).fillna(False).astype(int)

    # Fake breakout: new 20-bar high on WEAK volume that closes back below
    # the breakout level within two bars (trap for breakout buyers).
    hi20 = high.rolling(20).max().shift(1)
    broke = high > hi20
    failed = close < hi20
    out["fake_breakout"] = ((broke & (df["vol_z"] < 0)
                            & (failed | failed.shift(-1)))
                           .fillna(False).astype(int))

    # Effort/result divergence: big volume, price barely nets anywhere over
    # 3 bars — churn that often precedes a manipulated move.
    net3 = (close / close.shift(3) - 1).abs()
    out["churn"] = ((df["vol_z"] > 1.5)
                   & (net3 < 0.3 * df["atr_pct"])).fillna(False).astype(int)

    out["manip_score"] = out.sum(axis=1)
    return out


class QuoteSpoofDetector:
    """Live-only: remembers each symbol's last top-of-book and flags a
    spoof when a large resting size disappears and the imbalance flips.

    Feed it (bid_size, ask_size) each scan via `update`; it returns True
    when the previous quote looks like it held a pulled fake wall.
    """

    def __init__(self, size_drop: float = 0.5, flip: float = 0.25):
        self._last: dict[str, tuple[float, float]] = {}
        self.size_drop = size_drop   # wall shrank to < this fraction
        self.flip = flip             # imbalance moved at least this much

    def update(self, symbol: str, bid_size: float, ask_size: float) -> bool:
        prev = self._last.get(symbol)
        self._last[symbol] = (bid_size, ask_size)
        if prev is None:
            return False
        pb, pa = prev
        prev_imb = pb / (pb + pa) if (pb + pa) else 0.5
        cur_imb = bid_size / (bid_size + ask_size) if (bid_size + ask_size) else 0.5
        # Bid wall pulled: was bid-heavy, bid size collapsed, imbalance fell.
        bid_pull = (prev_imb > 0.65 and bid_size < pb * self.size_drop
                    and prev_imb - cur_imb > self.flip)
        ask_pull = (prev_imb < 0.35 and ask_size < pa * self.size_drop
                    and cur_imb - prev_imb > self.flip)
        return bool(bid_pull or ask_pull)


if __name__ == "__main__":
    # Quick smoke test on one ticker's history.
    import sys
    sys.path.insert(0, ".")
    from data import compute_features, download_bars
    sym = sys.argv[1] if len(sys.argv) > 1 else "TSLA"
    f = compute_features(download_bars(sym))
    flags = bar_flags(f)
    print(f"{sym}: {len(f)} bars")
    for col in flags.columns[:-1]:
        print(f"  {col:16s} fired on {flags[col].mean() * 100:.2f}% of bars")
    print(f"  bars with any manip signature: "
          f"{(flags['manip_score'] > 0).mean() * 100:.1f}%")
