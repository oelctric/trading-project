"""Data download and feature engineering.

Stock bars come from Alpaca's market data API (years of 15-minute SIP
history; the request end is lagged 16 minutes because the free plan cannot
query the most recent 15 minutes of SIP data). VIX comes from yfinance at
daily resolution, lagged one day so intraday bars only ever see the prior
day's fear reading. The live bot reuses the exact same pipeline so
train-time and trade-time inputs match.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import time as _time

import config
from strategies import STRATEGY_COLUMNS, add_setups

# "Fear-aware" features: market-wide fear (VIX) plus per-stock panic state
# (drawdown, red-bar streaks, loss-aversion asymmetry).
FEAR_COLUMNS = ["vix_z", "vix_chg", "vix_trend",
                "drawdown", "down_streak", "fear_ratio"]

# "Pro-level" features: what discretionary intraday traders actually key
# off — session VWAP, previous-day levels, the opening range, relative
# strength vs SPY, and the daily-timeframe trend.
PRO_COLUMNS = ["vwap_dist", "pdh_dist", "pdl_dist", "pdc_dist",
               "open_dist", "orh_dist", "orl_dist",
               "rel_str", "daily_trend", "gap", "mkt_above_vwap"]

# Candle-anatomy features: order-flow proxies read from each bar's shape
# (true order-book data isn't in free feeds; wick/close/volume geometry is
# the standard VSA-style proxy for what the book was doing inside the bar).
CANDLE_COLUMNS = ["upper_wick", "lower_wick", "clv", "delta_z",
                  "effort_result", "engulf", "hammer", "star", "doji"]

# Manipulation-detector outputs fed in as features so the network can learn
# to avoid the traps itself (fake_breakout alone is worth −13.6 bps when it
# fires on a breakout entry — see manipulation.py).
MANIP_COLUMNS = ["mf_stophunt_high", "mf_stophunt_low", "mf_absorption",
                 "mf_fake_breakout", "mf_churn"]

FEATURE_COLUMNS = [
    "log_ret", "range_pct", "body_pct", "vol_z",
    "rsi", "macd_hist", "sma20_dist", "sma50_dist",
    "bb_pos", "atr_pct", "hour_sin", "hour_cos",
] + FEAR_COLUMNS + PRO_COLUMNS + CANDLE_COLUMNS + STRATEGY_COLUMNS
# Note: MANIP_COLUMNS were tested as NN features (dataset_v3) and REVERTED —
# no benefit to pure-ML or the gated system (the fake-breakout FILTER already
# captures that value). Kept defined for reference / possible reuse.


_DATA_CLIENT = None


def _client() -> StockHistoricalDataClient:
    global _DATA_CLIENT
    if _DATA_CLIENT is None:
        if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
            raise RuntimeError("Alpaca keys missing from .env")
        _DATA_CLIENT = StockHistoricalDataClient(
            config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    return _DATA_CLIENT


def download_bars(ticker: str, days: int = config.HISTORY_DAYS) -> pd.DataFrame:
    """Regular-session 15-minute OHLCV bars from Alpaca, split-adjusted."""
    end = datetime.now(timezone.utc) - timedelta(minutes=16)
    if config.BAR_MINUTES % 60 == 0:  # Alpaca caps Minute amounts at 59
        timeframe = TimeFrame(config.BAR_MINUTES // 60, TimeFrameUnit.Hour)
    else:
        timeframe = TimeFrame(config.BAR_MINUTES, TimeFrameUnit.Minute)
    req = StockBarsRequest(
        symbol_or_symbols=ticker, timeframe=timeframe,
        start=end - timedelta(days=days), end=end, adjustment="all")
    df = _client().get_stock_bars(req).df
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = df.droplevel(0)  # drop the symbol level of the MultiIndex
    df.index = df.index.tz_convert("America/New_York")
    if "vwap" not in df.columns:
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
    df = df[["open", "high", "low", "close", "volume", "vwap"]]
    # Hourly bars align to the clock hour, so the 09:00 bar holds the
    # 09:30-10:00 open action — keep it.
    start_time = "09:00" if config.BAR_MINUTES % 60 == 0 else "09:30"
    return df.between_time(start_time, "15:45").dropna()


_VIX_CACHE = {"df": None, "at": 0.0}
_SPY_CACHE = {"s": None, "at": 0.0}


def _spy_state() -> pd.DataFrame:
    """SPY closes plus 'above its own session VWAP' flag (the market tape),
    cached for 15 minutes."""
    if _SPY_CACHE["s"] is None or _time.time() - _SPY_CACHE["at"] > 900:
        spy = download_bars("SPY")
        day = pd.Index(spy.index.date)
        pv = (spy["vwap"] * spy["volume"]).groupby(day).cumsum()
        cv = spy["volume"].groupby(day).cumsum()
        _SPY_CACHE["s"] = pd.DataFrame({
            "close": spy["close"],
            "above_vwap": (spy["close"] >= pv / cv).astype("float32"),
        })
        _SPY_CACHE["at"] = _time.time()
    return _SPY_CACHE["s"]


def _vix_features() -> pd.DataFrame:
    """Market-fear features from daily VIX closes (yfinance), shifted one
    day so intraday bars only see the prior day's reading (no lookahead).
    Cached for 15 minutes so a live scan downloads it once."""
    if _VIX_CACHE["df"] is None or _time.time() - _VIX_CACHE["at"] > 900:
        vix = yf.download("^VIX", period="5y", interval="1d",
                          auto_adjust=True, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix = vix["Close"]
        f = pd.DataFrame(index=vix.index)
        f["vix_z"] = ((vix - vix.rolling(100, min_periods=20).mean())
                      / vix.rolling(100, min_periods=20).std())
        f["vix_chg"] = np.log(vix / vix.shift(1))
        f["vix_trend"] = vix / vix.rolling(50, min_periods=10).mean() - 1
        f = f.shift(1)  # day D bars see day D-1 fear
        f.index = f.index.tz_localize("America/New_York")
        _VIX_CACHE["df"] = f
        _VIX_CACHE["at"] = _time.time()
    return _VIX_CACHE["df"]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _trailing_returns(high, low, close, atr, horizon,
                      trail_mult, sl_mult, tp_backstop):
    """Per-bar trailing-stop trade return (long and short), simulated over
    the next `horizon` bars, matching the live exit logic. Returns two
    float arrays; the final `horizon` entries are NaN (incomplete)."""
    n = len(close)
    rl = np.full(n, np.nan, np.float64)
    rs = np.full(n, np.nan, np.float64)
    for i in range(n - horizon):
        entry, a = close[i], atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        sl0, trail, tp = sl_mult * a, trail_mult * a, tp_backstop * a
        # long
        best, out_l = 0.0, None
        for k in range(1, horizon + 1):
            favhi = (high[i + k] - entry) / entry
            favlo = (low[i + k] - entry) / entry
            stop = max(-sl0, best - trail)
            if favlo <= stop:
                out_l = stop; break
            if favhi >= tp:
                out_l = tp; break
            best = max(best, favhi)
        rl[i] = out_l if out_l is not None else close[i + horizon] / entry - 1
        # short (mirror: profit when price falls)
        best, out_s = 0.0, None
        for k in range(1, horizon + 1):
            favlo = (entry - high[i + k]) / entry  # adverse for short
            favhi = (entry - low[i + k]) / entry   # favorable for short
            stop = max(-sl0, best - trail)
            if favlo <= stop:
                out_s = stop; break
            if favhi >= tp:
                out_s = tp; break
            best = max(best, favhi)
        rs[i] = out_s if out_s is not None else 1 - close[i + horizon] / entry
    return rl.astype(np.float32), rs.astype(np.float32)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append FEATURE_COLUMNS and the next-bar target to an OHLCV frame."""
    out = df.copy()
    close, high, low, vol = out["close"], out["high"], out["low"], out["volume"]

    out["log_ret"] = np.log(close / close.shift(1))
    out["range_pct"] = (high - low) / close
    out["body_pct"] = (close - out["open"]) / close
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std()
    out["vol_z"] = (vol - vol_mean) / vol_std

    out["rsi"] = _rsi(close) / 100.0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = (macd - signal) / close

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    out["sma20_dist"] = close / sma20 - 1
    out["sma50_dist"] = close / sma50 - 1

    bb_std = close.rolling(20).std()
    out["bb_pos"] = (close - sma20) / (2 * bb_std)

    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    out["atr_pct"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / close

    hours = out.index.hour + out.index.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # --- candle anatomy: what the order book did inside each bar ---
    body_top = np.maximum(out["open"], close)
    body_bot = np.minimum(out["open"], close)
    body = close - out["open"]
    rng = (high - low).replace(0, np.nan)

    # Wicks: lower wick = sell orders absorbed by bids (demand),
    # upper wick = buy orders absorbed by offers (supply).
    out["upper_wick"] = (high - body_top) / close
    out["lower_wick"] = (body_bot - low) / close

    # Close Location Value: -1 (sellers won the bar) .. +1 (buyers won).
    out["clv"] = ((2 * close - high - low) / rng).fillna(0)

    # Delta proxy: CLV-signed volume over 20 bars — which side of the book
    # has been eating liquidity lately.
    signed_vol = out["clv"] * vol
    out["delta_z"] = (signed_vol.rolling(20).sum()
                      / vol.rolling(20).sum()).clip(-1, 1)

    # Effort vs result (VSA): price progress per unit of relative volume.
    # Near zero on heavy volume = absorption by a large passive player.
    vol_rel = (vol / vol.rolling(20).mean()).replace(0, np.nan)
    out["effort_result"] = (body / (out["atr_pct"] * close)
                            / vol_rel).clip(-3, 3)

    # Classic reversal/indecision patterns as signed/binary flags.
    prev_body = body.shift(1)
    bull_engulf = ((body > 0) & (prev_body < 0)
                   & (close >= out["open"].shift(1))
                   & (out["open"] <= close.shift(1)))
    bear_engulf = ((body < 0) & (prev_body > 0)
                   & (close <= out["open"].shift(1))
                   & (out["open"] >= close.shift(1)))
    out["engulf"] = bull_engulf.astype("float32") - bear_engulf.astype("float32")
    body_h = body_top - body_bot
    out["hammer"] = (((body_bot - low) >= 2 * body_h)
                     & ((high - body_top) <= 0.5 * (body_bot - low))
                     ).astype("float32")
    out["star"] = (((high - body_top) >= 2 * body_h)
                   & ((body_bot - low) <= 0.5 * (high - body_top))
                   ).astype("float32")
    out["doji"] = (body.abs() <= 0.1 * rng).fillna(False).astype("float32")

    # --- fear-aware features ---
    # Per-stock panic state: how far under the recent high, how long the
    # current losing streak is, and whether downside moves are outsized
    # relative to upside moves (loss-aversion asymmetry).
    out["drawdown"] = close / close.rolling(100, min_periods=20).max() - 1
    red = out["log_ret"] < 0
    out["down_streak"] = red.groupby((~red).cumsum()).cumsum().clip(0, 10) / 10
    neg = out["log_ret"].where(out["log_ret"] < 0)
    pos = out["log_ret"].where(out["log_ret"] > 0)
    ratio = (neg.rolling(50, min_periods=10).std()
             / (pos.rolling(50, min_periods=10).std() + 1e-8))
    out["fear_ratio"] = (ratio - 1).clip(-2, 2)

    # Market-wide fear from the VIX, forward-filled onto this ticker's bars.
    vix = _vix_features().reindex(out.index, method="ffill")
    out = out.join(vix)

    # --- pro-level features (what discretionary traders key off) ---
    day = pd.Index(out.index.date)

    # Session-anchored VWAP: resets each morning, rebuilds with volume.
    pv = (out["vwap"] * vol).groupby(day).cumsum()
    cv = vol.groupby(day).cumsum()
    out["vwap_dist"] = close / (pv / cv) - 1

    # Previous-day high/low/close: the levels everyone watches.
    daily = out.groupby(day).agg(
        d_high=("high", "max"), d_low=("low", "min"),
        d_close=("close", "last"), d_open=("open", "first"))
    prev = daily.shift(1)
    out["pdh_dist"] = close / pd.Series(day.map(prev["d_high"]),
                                        index=out.index) - 1
    out["pdl_dist"] = close / pd.Series(day.map(prev["d_low"]),
                                        index=out.index) - 1
    out["pdc_dist"] = close / pd.Series(day.map(prev["d_close"]),
                                        index=out.index) - 1
    # Distance from today's open (known from the first bar onward).
    out["open_dist"] = close / pd.Series(day.map(daily["d_open"]),
                                         index=out.index) - 1

    # Opening range: the first bar's high/low (the 9:30-10:00 hour).
    out["orh_dist"] = close / out["high"].groupby(day).transform("first") - 1
    out["orl_dist"] = close / out["low"].groupby(day).transform("first") - 1

    # Relative strength vs SPY over the last 20 bars, and the market tape:
    # is SPY itself above its session VWAP right now ("check the macro
    # before trading micro").
    spy = _spy_state().reindex(out.index, method="ffill")
    out["rel_str"] = (close.pct_change(20)
                      - spy["close"].pct_change(20)).clip(-1, 1)
    out["mkt_above_vwap"] = spy["above_vwap"]

    # Overnight gap: today's open vs yesterday's close (the gap traders
    # play in the first 90 minutes).
    out["gap"] = (pd.Series(day.map(daily["d_open"]), index=out.index)
                  / pd.Series(day.map(prev["d_close"]), index=out.index) - 1)

    # Daily-timeframe trend: yesterday's close vs its own 20-day SMA
    # (yesterday's, so intraday bars never see today's close).
    d_close = daily["d_close"]
    d_trend = (d_close / d_close.rolling(20, min_periods=5).mean() - 1).shift(1)
    out["daily_trend"] = pd.Series(day.map(d_trend), index=out.index)

    out = add_setups(out)

    # Triple-barrier labels: simulate the ATR-scaled bracket the live bot
    # actually trades, in both directions. Entering at this bar's close,
    # does price touch the take-profit (TP_ATR_MULT x ATR in the trade's
    # favor) before the stop (SL_ATR_MULT x ATR against) within
    # LABEL_HORIZON bars? Ties within a bar go to the stop (conservative);
    # if neither is touched the trade exits at the horizon close.
    atr = out["atr_pct"]
    tp_dist = config.TP_ATR_MULT * atr
    sl_dist = config.SL_ATR_MULT * atr
    horizon = config.LABEL_HORIZON

    def barrier_ret(profit_lvl, stop_lvl, profit_side_hits, stop_side_hits,
                    exit_ret):
        first_tp = pd.Series(np.inf, index=out.index)
        first_sl = pd.Series(np.inf, index=out.index)
        for k in range(1, horizon + 1):
            first_tp = first_tp.mask(
                profit_side_hits(k, profit_lvl) & np.isinf(first_tp), k)
            first_sl = first_sl.mask(
                stop_side_hits(k, stop_lvl) & np.isinf(first_sl), k)
        return np.where(
            np.isfinite(first_sl) & (first_sl <= first_tp), -sl_dist,
            np.where(np.isfinite(first_tp), tp_dist, exit_ret))

    horizon_ret = close.shift(-horizon) / close - 1
    out["trade_ret"] = barrier_ret(
        close * (1 + tp_dist), close * (1 - sl_dist),
        lambda k, lvl: high.shift(-k) >= lvl,
        lambda k, lvl: low.shift(-k) <= lvl,
        horizon_ret)
    out["trade_ret_short"] = barrier_ret(
        close * (1 - tp_dist), close * (1 + sl_dist),
        lambda k, lvl: low.shift(-k) <= lvl,
        lambda k, lvl: high.shift(-k) >= lvl,
        -horizon_ret)

    # Trailing-exit returns matching the LIVE bot (SL -1 ATR, trail
    # TRAIL_ATR_MULT behind the best price, far TP backstop, horizon exit).
    # These are what the regime gate evaluates, so it scores the exit the
    # bot actually trades — not the fixed bracket above.
    tl, ts = _trailing_returns(
        high.to_numpy(), low.to_numpy(), close.to_numpy(),
        atr.to_numpy(), horizon,
        config.TRAIL_ATR_MULT, config.SL_ATR_MULT, config.TP_BACKSTOP_ATR)
    out["trail_ret"] = tl
    out["trail_ret_short"] = ts
    # Bars too recent to have a full horizon get NaN labels; they are
    # excluded from training but KEPT in the frame so the live bot can
    # read features off the newest bar.
    out.loc[out.index[-horizon:], ["trade_ret", "trade_ret_short"]] = np.nan
    out["target"] = (out["trade_ret"] > 0).astype(np.float32)

    return out.dropna(subset=FEATURE_COLUMNS)


def build_sequences(feat_df: pd.DataFrame, seq_len: int = config.SEQ_LEN):
    """Turn a feature frame into (X, y, ret_long, ret_short, setups) windows.

    `ret_long`/`ret_short` are simulated bracket-trade returns entered at
    each window's final bar; `setups` holds that bar's STRATEGY_COLUMNS
    flags. Windows whose final bar has no complete label horizon are
    dropped (they exist only for live inference).
    """
    values = feat_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets = feat_df["target"].to_numpy(dtype=np.float32)
    rets_l = feat_df["trade_ret"].to_numpy(dtype=np.float32)
    rets_s = feat_df["trade_ret_short"].to_numpy(dtype=np.float32)
    setup_flags = feat_df[STRATEGY_COLUMNS].to_numpy(dtype=np.float32)

    n = len(feat_df) - seq_len + 1
    if n <= 0:
        return (np.empty((0, seq_len, len(FEATURE_COLUMNS)), np.float32),
                np.empty(0, np.float32), np.empty(0, np.float32),
                np.empty(0, np.float32),
                np.empty((0, len(STRATEGY_COLUMNS)), np.float32))

    X = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0)
    X = np.ascontiguousarray(X.transpose(0, 2, 1))  # (n, seq_len, features)
    y = targets[seq_len - 1:]
    rl = rets_l[seq_len - 1:]
    rs = rets_s[seq_len - 1:]
    s = setup_flags[seq_len - 1:]
    ok = ~np.isnan(rl)  # exclude incomplete-horizon windows from training
    return X[ok], y[ok], rl[ok], rs[ok], s[ok]


def load_dataset(tickers=config.TICKERS, seq_len: int = config.SEQ_LEN,
                 max_cache_age_hours: float = 6.0):
    """Download all tickers and return chronological train/val/test splits.

    Each ticker is split by time BEFORE windowing so no future data leaks
    into training. Returns dict of (X, y, trade_ret, setups) per split.
    Results are cached on disk so consecutive train/backtest runs don't
    re-download everything.
    """
    cache = config.MODEL_DIR / "dataset_v2.npz"
    if cache.exists():
        age_h = (_time.time() - cache.stat().st_mtime) / 3600
        if age_h < max_cache_age_hours:
            print(f"  using cached dataset ({age_h:.1f}h old)")
            z = np.load(cache)
            return {name: tuple(z[f"{name}_{i}"] for i in range(5))
                    for name in ("train", "val", "test")}

    splits = {"train": [], "val": [], "test": []}
    for ticker in tickers:
        try:
            df = download_bars(ticker)
        except Exception as e:
            print(f"  skipping {ticker}: download failed ({e})")
            continue
        if len(df) < 200:
            print(f"  skipping {ticker}: only {len(df)} bars")
            continue
        feats = compute_features(df)
        n = len(feats)
        i_train = int(n * config.TRAIN_FRAC)
        i_val = int(n * (config.TRAIN_FRAC + config.VAL_FRAC))
        parts = {"train": feats.iloc[:i_train],
                 "val": feats.iloc[i_train:i_val],
                 "test": feats.iloc[i_val:]}
        for name, part in parts.items():
            splits[name].append(build_sequences(part, seq_len))
        print(f"  {ticker}: {n} usable bars")

    out = {}
    for name, chunks in splits.items():
        out[name] = tuple(np.concatenate([c[i] for c in chunks])
                          for i in range(5))
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{f"{name}_{i}": arrs[i]
                       for name, arrs in out.items() for i in range(5)})
    return out
