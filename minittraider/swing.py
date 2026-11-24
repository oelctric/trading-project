"""Swing strategy: fee-aware trend entries + a dynamic exit manager.

Why this exists
---------------
The 1-minute scalpers could never win: a 1-minute candle moves ~0.05%, but a
crypto round trip costs ~0.5% in fees. Every trade started 10 candles in the
hole. This strategy trades hourly bars, only when the expected move is several
times the fees, and lets the *market* decide how long a trade lives:

Entry (all on closed bars, no lookahead)
  - trend filter: fast EMA above slow EMA (long) / below (short), price on the
    right side of the slow EMA
  - trend quality: Kaufman efficiency ratio above a floor (a clean, directional
    market rather than chop)
  - trigger: close breaks the prior N-bar high (long) / low (short)
  - fee filter: the initial risk (stop distance) must be at least
    MIN_RISK_FEES x the round-trip fee, so fees can never be more than a small
    fraction of what a trade is risking

Dynamic exit (re-evaluated every bar - there is no fixed holding time)
  1. hard stop at entry -/+ INIT_STOP_ATR * ATR
  2. break-even: once the trade has been +BE_R risk-units in profit, the stop is
     lifted to entry + fees, so a good trade can't turn into a loser
  3. chandelier trail: stop follows the best price since entry minus
     TRAIL_ATR * ATR, and tightens to TIGHT_ATR once the trade is +TIGHT_R
     in profit (lock in more of a big move)
  4. thesis broken: if the bar closes back through the slow EMA the trend that
     justified the trade is gone -> exit at the next open
  5. stagnation: if after STALL_BARS bars the trade never got +STALL_R in
     profit, the move didn't come -> exit instead of waiting
  6. MAX_HOLD_BARS is only a safety net

So winners are held for as long as the trend keeps paying, and losers / dead
trades are cut early.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SwingParams:
    fast: int = 20
    slow: int = 100
    breakout: int = 48           # bars for the breakout high/low
    er_len: int = 48             # efficiency-ratio lookback
    er_min: float = 0.25         # minimum trend quality
    atr_len: int = 14
    init_stop_atr: float = 2.5
    be_r: float = 1.0
    trail_atr: float = 3.0
    tight_r: float = 3.0
    tight_atr: float = 2.0
    stall_bars: int = 24
    stall_r: float = 0.5
    max_hold_bars: int = 24 * 14
    min_risk_fees: float = 3.0   # stop distance must be >= this x round-trip fee
    allow_short: bool = False
    regime_len: int = 0          # >0: only long when BTC > its EMA(regime_len), short when below
    mode: str = "breakout"       # "breakout" (new N-bar high) or "pullback" (dip to fast EMA in a trend)
    filters: tuple = ()          # extra indicator confirmations, names from FILTERS below

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- indicators

def indicators(df: pd.DataFrame, p: SwingParams,
               leader: pd.DataFrame | None = None,
               breadth: pd.Series | None = None) -> pd.DataFrame:
    """All columns are causal: row t only uses bars <= t.
    `leader` (BTC bars) feeds the market-regime filter when p.regime_len > 0."""
    out = pd.DataFrame(index=df.index)
    out["regime"] = 0.0
    if p.regime_len and leader is not None and len(leader):
        lc = leader["close"]
        reg = np.sign(lc - lc.ewm(span=p.regime_len, adjust=False).mean())
        out["regime"] = reg.reindex(df.index, method="ffill").fillna(0.0)
    c, h, l = df["close"], df["high"], df["low"]
    out["ema_f"] = c.ewm(span=p.fast, adjust=False).mean()
    out["ema_s"] = c.ewm(span=p.slow, adjust=False).mean()
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / p.atr_len, adjust=False).mean()
    change = (c - c.shift(p.er_len)).abs()
    path = c.diff().abs().rolling(p.er_len).sum()
    out["er"] = change / path.replace(0, np.nan)
    out["hh"] = h.shift(1).rolling(p.breakout).max()   # prior N-bar high (excl. this bar)
    out["ll"] = l.shift(1).rolling(p.breakout).min()
    out["open"], out["high"], out["low"] = df["open"], h, l
    # extra context for the trade-quality model (all causal)
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    out["ret20"] = c / c.shift(20) - 1
    out["vol_ratio"] = df["volume"] / df["volume"].rolling(20).median().shift(1)
    out["ema_spread"] = (out["ema_f"] - out["ema_s"]) / out["atr"]
    out["slow_slope"] = (out["ema_s"] - out["ema_s"].shift(10)) / out["atr"]
    out["lead_trend"] = 0.0
    out["lead_ret20"] = 0.0
    if leader is not None and len(leader):
        lc = leader["close"]
        lt = np.sign(lc - lc.ewm(span=100, adjust=False).mean())
        out["lead_trend"] = lt.reindex(df.index, method="ffill").fillna(0.0)
        out["lead_ret20"] = (lc / lc.shift(20) - 1).reindex(df.index, method="ffill").fillna(0.0)

    # ---- extra indicators for the optional entry filters (all causal)
    # ADX(14): trend strength regardless of direction
    up_mv, dn_mv = h.diff(), -l.diff()
    plus_dm = up_mv.where((up_mv > dn_mv) & (up_mv > 0), 0.0)
    minus_dm = dn_mv.where((dn_mv > up_mv) & (dn_mv > 0), 0.0)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr14
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    # MACD(12,26,9) histogram
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    # Bollinger bandwidth: was the market quiet (a squeeze) just before this bar?
    bw = 4 * c.rolling(20).std() / c.rolling(20).mean()
    bw_rank = bw.rolling(365, min_periods=120).rank(pct=True)
    out["squeeze"] = bw_rank.shift(1).rolling(10).min()      # lowest bandwidth rank in prior 10 bars
    # weekly trend: last COMPLETED week's close vs its 20-week EMA. Weekly bars are
    # labelled by their last day (Sunday), so a daily bar only "sees" a weekly value
    # once that whole week has closed - no extra shift needed (an extra shift made it
    # a week stale).
    wk = c.resample("W").last()
    wk_trend = np.sign(wk - wk.ewm(span=20, adjust=False).mean())
    out["weekly_trend"] = wk_trend.reindex(df.index, method="ffill").fillna(0.0)
    # on-balance volume vs its 20-bar EMA
    obv = (np.sign(c.diff()).fillna(0) * df["volume"]).cumsum()
    out["obv_trend"] = np.sign(obv - obv.ewm(span=20, adjust=False).mean())
    # volatility regime: today's ATR% as a percentile of the past year
    out["atr_rank"] = (out["atr"] / c).rolling(365, min_periods=120).rank(pct=True)
    # where the bar closed inside its range (1 = at the high)
    out["close_loc"] = (c - l) / (h - l).replace(0, np.nan)

    # ---- round 2 indicators
    # relative strength vs the leader (BTC): coin's 20-bar return minus BTC's
    out["rel_strength"] = 0.0
    if leader is not None and len(leader):
        lr = (leader["close"] / leader["close"].shift(20) - 1).reindex(df.index, method="ffill")
        out["rel_strength"] = (out["ret20"] - lr).fillna(0.0)
    # market breadth: share of the watchlist above its 100-bar EMA (-1..+1 scale)
    out["breadth"] = breadth.reindex(df.index, method="ffill").fillna(0.0) if breadth is not None else 0.0
    # Supertrend(10, 3): ATR band trend direction
    atr10 = tr.ewm(alpha=1 / 10, adjust=False).mean()
    mid = (h + l) / 2
    ub, lb = (mid + 3 * atr10).to_numpy(), (mid - 3 * atr10).to_numpy()
    cl = c.to_numpy()
    st_dir = np.zeros(len(df))
    fu, fl = ub.copy(), lb.copy()
    for k in range(1, len(df)):
        fu[k] = ub[k] if (ub[k] < fu[k - 1] or cl[k - 1] > fu[k - 1]) else fu[k - 1]
        fl[k] = lb[k] if (lb[k] > fl[k - 1] or cl[k - 1] < fl[k - 1]) else fl[k - 1]
        prev = st_dir[k - 1] or 1
        st_dir[k] = 1 if (prev < 0 and cl[k] > fu[k - 1]) or (prev > 0 and cl[k] >= fl[k - 1]) else -1
    out["supertrend"] = st_dir
    # Ichimoku: price vs the cloud (spans shifted 26 bars forward = computed 26 bars ago)
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    top, bot = np.maximum(span_a, span_b), np.minimum(span_a, span_b)
    out["ichimoku"] = np.where(c > top, 1.0, np.where(c < bot, -1.0, 0.0))
    out.loc[span_b.isna(), "ichimoku"] = np.nan
    # Aroon(25): how recently the high / low was made
    n = 25
    out["aroon_up"] = h.rolling(n + 1).apply(lambda x: x.argmax(), raw=True) / n * 100
    out["aroon_dn"] = l.rolling(n + 1).apply(lambda x: x.argmin(), raw=True) / n * 100
    # Choppiness index(14): 100 = pure chop, low = clean trend
    out["chop"] = 100 * np.log10(tr.rolling(14).sum() / (h.rolling(14).max() - l.rolling(14).min())) / np.log10(14)
    # stretch: distance from the 200-bar EMA in ATRs
    out["stretch200"] = (c - c.ewm(span=200, adjust=False).mean()) / out["atr"]
    # breakout-bar strength: today's range vs ATR (previous bar's ATR to avoid self-reference)
    out["range_x"] = (h - l) / out["atr"].shift(1)
    # stochastic %K(14)
    out["stoch"] = 100 * (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min()).replace(0, np.nan)
    out["weekday"] = df.index.dayofweek
    return out


def market_breadth(frames: dict[str, pd.DataFrame], span: int = 100) -> pd.Series:
    """Share of coins above their EMA(span), scaled to -1..+1 (all below .. all above)."""
    cols = {s: np.sign(d["close"] - d["close"].ewm(span=span, adjust=False).mean())
            for s, d in frames.items() if len(d)}
    return pd.DataFrame(cols).mean(axis=1)


# ---------------------------------------------------------------- optional indicator filters
# Each takes (sign, close, ind) with sign +1 for a long signal / -1 for a short and
# returns True if the indicator CONFIRMS the trade. Missing data never blocks.

def _ok(x) -> bool:
    return x is not None and not pd.isna(x)


FILTERS = {
    "adx": ("ADX > 20 (a real trend, not drift)",
            lambda s, c, i: not _ok(i.get("adx")) or i["adx"] > 20),
    "macd": ("MACD histogram agrees with the side",
             lambda s, c, i: not _ok(i.get("macd_hist")) or s * i["macd_hist"] > 0),
    "volume": ("volume >= 1.2x its 20-bar median (participation)",
               lambda s, c, i: not _ok(i.get("vol_ratio")) or i["vol_ratio"] >= 1.2),
    "squeeze": ("breakout follows a Bollinger squeeze (bandwidth in bottom 30% of the year)",
                lambda s, c, i: not _ok(i.get("squeeze")) or i["squeeze"] <= 0.30),
    "weekly": ("weekly trend agrees (last week's close vs 20-week EMA)",
               lambda s, c, i: s * i.get("weekly_trend", 0.0) >= 0),
    "obv": ("on-balance volume trend agrees",
            lambda s, c, i: s * i.get("obv_trend", 0.0) >= 0),
    "rsi_cap": ("RSI not over-stretched (< 75 for longs, > 25 for shorts)",
                lambda s, c, i: not _ok(i.get("rsi")) or (i["rsi"] < 75 if s > 0 else i["rsi"] > 25)),
    "calm_vol": ("volatility not extreme (ATR% below its 80th percentile of the year)",
                 lambda s, c, i: not _ok(i.get("atr_rank")) or i["atr_rank"] <= 0.80),
    "mom20": ("20-bar momentum agrees with the side",
              lambda s, c, i: not _ok(i.get("ret20")) or s * i["ret20"] > 0),
    "strong_close": ("bar closed in the top 30% of its range (bottom 30% for shorts)",
                     lambda s, c, i: not _ok(i.get("close_loc"))
                     or (i["close_loc"] >= 0.7 if s > 0 else i["close_loc"] <= 0.3)),
    # round 2
    "rel_strength": ("coin is beating BTC over 20 bars in the trade's direction",
                     lambda s, c, i: s * i.get("rel_strength", 0.0) >= 0),
    "breadth": ("most of the watchlist trends the same way (breadth agrees)",
                lambda s, c, i: s * i.get("breadth", 0.0) >= 0),
    "supertrend": ("Supertrend(10,3) points the same way",
                   lambda s, c, i: s * i.get("supertrend", 0.0) > 0),
    "ichimoku": ("price on the right side of the Ichimoku cloud",
                 lambda s, c, i: not _ok(i.get("ichimoku")) or s * i["ichimoku"] > 0),
    "aroon": ("Aroon(25) confirms a fresh high (long) / low (short) > 70",
              lambda s, c, i: not _ok(i.get("aroon_up"))
              or (i["aroon_up"] > 70 if s > 0 else i["aroon_dn"] > 70)),
    "low_chop": ("choppiness index < 50 (clean trend, not zig-zag)",
                 lambda s, c, i: not _ok(i.get("chop")) or i["chop"] < 50),
    "stretch_cap": ("not over-extended: within 6 ATR of the 200-bar EMA",
                    lambda s, c, i: not _ok(i.get("stretch200")) or s * i["stretch200"] <= 6),
    "range_x": ("breakout bar range >= 1.2x ATR (real push)",
                lambda s, c, i: not _ok(i.get("range_x")) or i["range_x"] >= 1.2),
    "no_weekend": ("signal bar is a weekday (weekend crypto is thin)",
                   lambda s, c, i: i.get("weekday", 0) < 5),
    "stoch_cap": ("stochastic not pinned (< 95 for longs, > 5 for shorts)",
                  lambda s, c, i: not _ok(i.get("stoch")) or (i["stoch"] < 95 if s > 0 else i["stoch"] > 5)),
}


def _apply_filters(side: str, why: str, row_c: float, ind, p: SwingParams) -> tuple[str, str]:
    sign = 1 if side == "long" else -1
    for name in p.filters:
        desc, check = FILTERS[name]
        if not check(sign, row_c, ind):
            return "none", f"{side} breakout rejected by {name} filter: needs {desc}"
    if p.filters:
        why += "; confirmed by " + ", ".join(p.filters)
    return side, why


def entry_signal(row_c: float, ind: pd.Series | dict, p: SwingParams,
                 round_trip_fee: float) -> tuple[str, str]:
    """('long'|'short'|'none', explanation) for the bar that just closed."""
    ema_f, ema_s, atr_v, er = ind["ema_f"], ind["ema_s"], ind["atr"], ind["er"]
    hh, ll = ind["hh"], ind["ll"]
    if any(pd.isna(x) for x in (ema_f, ema_s, atr_v, er, hh, ll)) or atr_v <= 0:
        return "none", "warming up indicators"
    risk_pct = p.init_stop_atr * atr_v / row_c
    if risk_pct < p.min_risk_fees * round_trip_fee:
        return "none", (f"expected move too small vs fees (stop {risk_pct:.2%} < "
                        f"{p.min_risk_fees:.0f}x round-trip fee {round_trip_fee:.2%})")
    if er < p.er_min:
        return "none", f"choppy market (efficiency {er:.2f} < {p.er_min})"
    regime = ind.get("regime", 0.0) if p.regime_len else None
    long_ok = regime is None or regime > 0
    short_ok = p.allow_short and (regime is None or regime < 0)
    if not long_ok and not short_ok:
        return "none", "market regime (BTC trend) against every allowed side"
    if p.mode == "pullback":
        lo, hi = ind.get("low", np.nan), ind.get("high", np.nan)
        if long_ok and ema_f > ema_s and lo <= ema_f < row_c and row_c > ind.get("open", row_c):
            return "long", (f"uptrend pullback: dipped to EMA{p.fast} {ema_f:.6g} and closed back "
                            f"above it, EMA{p.fast} > EMA{p.slow}, efficiency {er:.2f}")
        if short_ok and ema_f < ema_s and hi >= ema_f > row_c and row_c < ind.get("open", row_c):
            return "short", (f"downtrend pullback: rallied to EMA{p.fast} {ema_f:.6g} and closed back "
                             f"below it, EMA{p.fast} < EMA{p.slow}, efficiency {er:.2f}")
        return "none", "no pullback-and-reject in the direction of the trend"
    if long_ok and ema_f > ema_s and row_c > ema_s and row_c > hh:
        return _apply_filters("long", (f"uptrend breakout: close {row_c:.6g} > {p.breakout}-bar high "
                                       f"{hh:.6g}, EMA{p.fast} > EMA{p.slow}, efficiency {er:.2f}"),
                              row_c, ind, p)
    if short_ok and ema_f < ema_s and row_c < ema_s and row_c < ll:
        return _apply_filters("short", (f"downtrend breakdown: close {row_c:.6g} < {p.breakout}-bar low "
                                        f"{ll:.6g}, EMA{p.fast} < EMA{p.slow}, efficiency {er:.2f}"),
                              row_c, ind, p)
    return "none", "no breakout in the direction of the trend"


FEATURES = ["er", "atr_pct", "bo_dist", "ema_spread", "slow_slope", "rsi_dir",
            "ret20_dir", "vol_ratio", "lead_trend_dir", "lead_ret20_dir", "is_short"]


def entry_features(side: str, close: float, ind) -> dict:
    """Features describing a signal, oriented so + means 'in the trade's favour'."""
    sign = 1 if side == "long" else -1
    atr_v = float(ind["atr"])
    level = float(ind["hh"]) if sign > 0 else float(ind["ll"])
    g = lambda k, d=0.0: d if pd.isna(ind.get(k, np.nan)) else float(ind[k])  # noqa: E731
    return {
        "er": g("er"),
        "atr_pct": atr_v / close,
        "bo_dist": sign * (close - level) / atr_v,          # how far through the breakout level
        "ema_spread": sign * g("ema_spread"),
        "slow_slope": sign * g("slow_slope"),
        "rsi_dir": sign * (g("rsi", 50.0) - 50) / 50,
        "ret20_dir": sign * g("ret20"),
        "vol_ratio": min(g("vol_ratio", 1.0), 10.0),
        "lead_trend_dir": sign * g("lead_trend"),
        "lead_ret20_dir": sign * g("lead_ret20"),
        "is_short": float(sign < 0),
    }


# ---------------------------------------------------------------- dynamic exits

@dataclass
class ExitManager:
    """Per-position exit state. Serializable so live positions survive restarts."""
    side: str
    entry: float
    stop: float
    risk: float                  # initial risk per unit (entry - stop for longs)
    fee_buffer: float            # price distance that covers round-trip fees
    best: float                  # best price seen since entry (high for long, low for short)
    bars: int = 0
    mfe_r: float = 0.0           # max favorable excursion in R
    stage: str = "initial"       # initial -> breakeven -> trailing -> tight
    log: list = field(default_factory=list)

    @classmethod
    def open(cls, side: str, entry: float, atr_v: float, p: SwingParams,
             round_trip_fee: float) -> "ExitManager":
        sign = 1 if side == "long" else -1
        stop = entry - sign * p.init_stop_atr * atr_v
        return cls(side=side, entry=entry, stop=stop, risk=abs(entry - stop),
                   fee_buffer=entry * round_trip_fee, best=entry)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["log"] = d["log"][-5:]
        return d

    def intrabar_stop(self, o: float, h: float, l: float) -> float | None:
        """Fill price if this bar hit the stop (gap-through fills at the open)."""
        if self.side == "long" and l <= self.stop:
            return min(o, self.stop)
        if self.side == "short" and h >= self.stop:
            return max(o, self.stop)
        return None

    def on_bar_close(self, h: float, l: float, c: float, ind, p: SwingParams) -> str | None:
        """Update trailing state with the bar that just closed. Returns a reason to
        exit at the next open, or None to keep holding. Stops are ratchet-only."""
        self.bars += 1
        sign = 1 if self.side == "long" else -1
        self.best = max(self.best, h) if sign > 0 else min(self.best, l)
        self.mfe_r = max(self.mfe_r, sign * (self.best - self.entry) / self.risk)
        atr_v = float(ind["atr"])

        new_stop = self.stop
        if self.mfe_r >= p.be_r:
            new_stop = _better(new_stop, self.entry + sign * self.fee_buffer, sign)
            self.stage = "breakeven" if self.stage == "initial" else self.stage
        if self.mfe_r >= p.be_r:
            k = p.tight_atr if self.mfe_r >= p.tight_r else p.trail_atr
            trail = self.best - sign * k * atr_v
            if _better(new_stop, trail, sign) == trail and trail != new_stop:
                self.stage = "tight" if self.mfe_r >= p.tight_r else "trailing"
            new_stop = _better(new_stop, trail, sign)
        if new_stop != self.stop:
            self.stop = new_stop

        if sign * (c - float(ind["ema_s"])) < 0:
            return "trend broken (closed through slow EMA)"
        if self.bars >= p.stall_bars and self.mfe_r < p.stall_r:
            return f"stalled ({self.bars} bars, best +{self.mfe_r:.2f}R)"
        if self.bars >= p.max_hold_bars:
            return "max hold safety net"
        return None


def _better(a: float, b: float, sign: int) -> float:
    """The tighter (more protective) of two stops."""
    return max(a, b) if sign > 0 else min(a, b)
