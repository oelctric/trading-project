"""Live glue for the swing strategy: bar fetching, entry decisions, and dynamic
exit management for open positions. Uses the exact same swing.py logic the
backtest (backtest_swing.py) validated.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace

import pandas as pd

import config
from brain import Decision, fmt_price
from history import RESAMPLED, fetch_coinbase, fetch_yahoo, resample
from logger import get_logger
from swing import ExitManager, SwingParams, entry_signal, indicators, market_breadth

log = get_logger("swing")

PARAMS_FILE = os.path.join(os.path.dirname(__file__), "swing_params.json")
SIGNALS_FILE = os.path.join(os.path.dirname(__file__), "swing_signals.json")
TF_SECONDS = {"1h": 3600, "2h": 7200, "4h": 14400, "8h": 28800, "12h": 43200, "1d": 86400}


def load_params() -> tuple[SwingParams, str]:
    """Tuned params + timeframe from swing_params.json (written by
    `python backtest_swing.py --tune`), falling back to defaults."""
    p, tf = SwingParams(), config.SWING_TIMEFRAME
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            blob = json.load(f)
        p = replace(p, **blob["params"])
        tf = blob.get("tf", tf)
    p = replace(p, filters=tuple(p.filters))
    return replace(p, allow_short=config.SWING_ALLOW_SHORT), tf


PARAMS, TIMEFRAME = load_params()
BAR_SECONDS = TF_SECONDS[TIMEFRAME]
_cache: dict[str, tuple[float, pd.DataFrame]] = {}


def _load_acted() -> dict[str, str]:
    """Signal bars already traded, per symbol - persisted so a restart can't re-buy
    the same (by then stale) signal after a stop-out."""
    try:
        with open(SIGNALS_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


_acted_bar: dict[str, str] = _load_acted()


def fee_per_side(symbol: str) -> float:
    """The fee the live account actually pays: futures when leverage is on."""
    if symbol.endswith("-USD") and getattr(config, "USE_LEVERAGE", False):
        return config.FUTURES_FEE_PER_SIDE
    return config.FEE_PER_SIDE["crypto" if symbol.endswith("-USD") else "stock"]


def round_trip_fee(symbol: str) -> float:
    return 2 * fee_per_side(symbol)


def closed_bars(symbol: str, is_crypto: bool) -> pd.DataFrame:
    """Enough CLOSED bars of the strategy timeframe for every indicator.
    Re-downloads only when a new bar has closed since the last fetch."""
    now = time.time()
    bar_id = now // BAR_SECONDS
    hit = _cache.get(symbol)
    if hit and hit[0] == bar_id:
        return hit[1]
    need_bars = max(PARAMS.slow * 3 + PARAMS.breakout + 50, 420)  # 1-year volatility rank needs ~1y
    base_tf, rule = RESAMPLED.get(TIMEFRAME, (TIMEFRAME, None))
    days = max(3, int(need_bars * BAR_SECONDS / 86400) + 2)
    try:
        df = fetch_coinbase(symbol, base_tf, days) if is_crypto else fetch_yahoo(symbol, base_tf, days)
    except Exception as e:
        # don't cache a failure (that would blind the bot until the next bar);
        # fall back to the last good bars and retry on the next scan
        log.warning("bar download failed for %s: %s", symbol, e)
        if hit:
            return hit[1]
        raise
    if rule and not df.empty:
        df = resample(df, rule)
    if not df.empty:
        # drop the bar that is still forming - decisions only on closed bars
        cutoff = pd.Timestamp(bar_id * BAR_SECONDS, unit="s", tz="UTC")
        df = df[df.index < cutoff]
    if df.empty:
        log.warning("no bars returned for %s - will retry next scan", symbol)
        return hit[1] if hit else df
    _cache[symbol] = (bar_id, df)
    return df


def prefetch(symbols: list[str]) -> None:
    """Fetch every symbol's closed bars up front (cheap: cached per bar) so the
    breadth indicator always sees the whole watchlist, not just the coins
    scanned so far."""
    for s in symbols:
        try:
            closed_bars(s, s.endswith("-USD"))
        except Exception as e:
            log.warning("prefetch failed for %s: %s", s, e)


def _context() -> tuple[pd.DataFrame | None, pd.Series | None]:
    """BTC bars (leader) + market breadth from the bars already fetched this scan,
    so relative-strength / breadth filters see the same inputs as in the backtest."""
    leader = _cache.get("BTC-USD", (None, None))[1]
    frames = {s: d for s, (_, d) in _cache.items() if d is not None and len(d)}
    return leader, (market_breadth(frames) if frames else None)


def decide_swing(symbol: str, df: pd.DataFrame) -> Decision:
    d = Decision(symbol=symbol, action="none", confidence=0.0)
    if df is None or len(df) < PARAMS.slow + PARAMS.breakout:
        d.reasons.append(f"=> no trade: need more {TIMEFRAME} history")
        return d
    if _acted_bar.get(symbol) == str(df.index[-1]):
        d.reasons.append("=> no trade: already traded this bar's signal")
        return d
    ind = indicators(df, PARAMS, *_context()).iloc[-1]
    c = float(df["close"].iloc[-1])
    side, why = entry_signal(c, ind, PARAMS, round_trip_fee(symbol))
    d.price, d.atr = c, float(ind["atr"])
    if side == "none":
        d.reasons.append(f"=> no trade: {why}")
        return d
    signal_age_h = (pd.Timestamp.now(tz="UTC") - df.index[-1]).total_seconds() / 3600 \
        - BAR_SECONDS / 3600
    max_age = getattr(config, "SWING_MAX_SIGNAL_AGE_HOURS", 3)
    if signal_age_h > max_age:
        # the backtest buys right as the bar closes; acting hours later (e.g. the
        # bot was off at midnight) would chase a different price than was tested
        d.reasons.append(f"=> no trade: {side} signal, but its bar closed {signal_age_h:.1f}h "
                         f"ago (> {max_age}h) - too late to act on it")
        return d
    sign = 1 if side == "long" else -1
    d.action, d.confidence = side, 0.7
    d.sl = c - sign * PARAMS.init_stop_atr * d.atr
    d.tp = 0.0  # no fixed target: the exit manager lets winners run
    d.reasons.append(f"=> {side.upper()} ({TIMEFRAME} swing): {why}; initial stop "
                     f"{fmt_price(d.sl)}, no fixed target - dynamic trailing exit")
    return d


def mark_acted(symbol: str, df: pd.DataFrame) -> None:
    """Remember that this bar's signal was TRADED. Call only after a position
    actually opened - marking it before (as new_exit_state used to) blocked a
    refused signal (slots full, sizing too small) from being taken later."""
    if df is None or df.empty:
        return
    _acted_bar[symbol] = str(df.index[-1])
    try:
        with open(SIGNALS_FILE, "w") as f:
            json.dump(_acted_bar, f, indent=1)
    except OSError as e:
        log.warning("could not save %s: %s", SIGNALS_FILE, e)


def new_exit_state(symbol: str, side: str, entry: float, atr_v: float,
                   df: pd.DataFrame, opened_at: pd.Timestamp | None = None) -> dict:
    """Exit state for a fresh position; bars up to the entry signal are already 'seen'.
    `opened_at` (default: now) lets manage() ignore prices from BEFORE the entry."""
    em = ExitManager.open(side, entry, atr_v, PARAMS, round_trip_fee(symbol))
    state = em.to_dict()
    state["opened_at"] = str(opened_at if opened_at is not None else pd.Timestamp.now(tz="UTC"))
    state["last_bar"] = str(df.index[-1]) if df is not None and not df.empty else ""
    return state


def _stop_reason(em: ExitManager) -> str:
    return {"initial": "stop loss", "breakeven": "break-even stop"}.get(
        em.stage, f"trailing stop ({em.stage})")


def manage(state: dict, df: pd.DataFrame, price: float
           ) -> tuple[str | None, dict, float | None]:
    """Advance the exit manager over any newly closed bars, then check the live
    price against the (ratcheted) stop.

    Each newly closed bar is first checked for a stop hit INSIDE the bar (its
    low/high), exactly like the backtest - so if the bot was offline, or the
    price wicked through the stop between scans, the stop still counts and fills
    at the stop price (or at the bar's open if it gapped past).

    Returns (exit reason or None, new state, fill price or None = use live price)."""
    last_bar = state.get("last_bar", "")
    opened_at = state.get("opened_at")
    em = ExitManager(**{k: v for k, v in state.items() if k not in ("last_bar", "opened_at")})
    reason, fill = None, None
    if df is not None and not df.empty:
        ind = indicators(df, PARAMS, *_context())
        new = df[df.index > pd.Timestamp(last_bar)] if last_bar else df.iloc[0:0]
        for ts, bar in new.iterrows():
            # The entry bar started BEFORE we entered (the bot enters up to a few
            # hours into it): its high/low may come from before the position
            # existed. Don't let them trigger the stop or inflate the best price -
            # use only its close (stops after entry were checked live each scan).
            pre_entry = opened_at is not None and ts < pd.Timestamp(opened_at)
            o_, h_, l_, c_ = (float(bar[k]) for k in ("open", "high", "low", "close"))
            if pre_entry:
                o_ = h_ = l_ = c_
            hit = None if pre_entry else em.intrabar_stop(o_, h_, l_)
            last_bar = str(ts)
            if hit is not None:
                reason, fill = _stop_reason(em) + " (hit during a closed bar)", hit
                break
            r = em.on_bar_close(h_, l_, c_, ind.loc[ts], PARAMS)
            if r:
                reason = r
                break
    sign = 1 if em.side == "long" else -1
    if reason is None and sign * (price - em.stop) <= 0:
        reason = _stop_reason(em)
    state = em.to_dict()
    state["last_bar"] = last_bar
    if opened_at is not None:
        state["opened_at"] = opened_at
    return reason, state, fill


def describe(state: dict, price: float) -> str:
    sign = 1 if state["side"] == "long" else -1
    r_now = sign * (price - state["entry"]) / state["risk"] if state["risk"] else 0.0
    return (f"{state['stage']} stop {fmt_price(state['stop'])}, now {r_now:+.2f}R "
            f"(best {state['mfe_r']:+.2f}R), {state['bars']} {TIMEFRAME} bars held")
