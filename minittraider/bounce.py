"""The 'price bounces' strategy (from Krafer Crypto's order-book determinism work).

The idea: minute-to-minute price is essentially random EXCEPT near large resting
orders, which absorb one side of the randomness - a fair coin where the first
few heads don't count stops being 50/50. So:

  - find the large-order zones: real walls in the live order book (crypto), plus
    zones inferred from price behavior - levels price rejected repeatedly
  - never enter in the middle of the cloud (it's 50/50 there); only act when
    price is AT a zone and showing rejection
  - trade the bounce toward the empty side, stop just beyond the zone (if the
    wall breaks, the thesis is dead), target the next zone across
  - only take it when reward/risk is clearly asymmetric

This trades rarely and aims for moves several ATRs long, which is also what a
strategy needs to survive fees at this timescale.
"""

from __future__ import annotations

import pandas as pd

import config
from brain import Decision, fmt_price
from signals import atr, volatility_regime


def swing_zones(df: pd.DataFrame, k: int = 3, lookback: int = 240) -> list[tuple[float, int]]:
    """Support/resistance levels inferred from price behavior alone.

    Swing highs/lows (extreme of their +-k neighborhood) clustered within
    0.3 ATR. Returns [(level, touches)] - more touches = a level price has
    repeatedly failed to cross, i.e. probably a large order sitting there.
    """
    d = df.iloc[-lookback:]
    highs = d["high"].to_numpy()
    lows = d["low"].to_numpy()
    a = float(atr(d).iloc[-1]) or 1e-9

    points: list[float] = []
    for i in range(k, len(d) - k):
        if highs[i] >= highs[i - k:i + k + 1].max():
            points.append(float(highs[i]))
        if lows[i] <= lows[i - k:i + k + 1].min():
            points.append(float(lows[i]))

    points.sort()
    zones: list[tuple[float, int]] = []
    for p in points:
        if zones and p - zones[-1][0] <= 0.3 * a:
            lvl, n = zones[-1]
            zones[-1] = ((lvl * n + p) / (n + 1), n + 1)
        else:
            zones.append((p, 1))
    return [(lvl, n) for lvl, n in zones if n >= config.BOUNCE_MIN_TOUCHES]


def decide_bounce(symbol: str, df: pd.DataFrame,
                  walls: dict | None = None, tape: float | None = None) -> Decision:
    d = Decision(symbol=symbol, action="none", confidence=0.0)
    if df is None or len(df) < 60:
        d.reasons.append("not enough candle history yet")
        return d

    price = d.price = float(df["close"].iloc[-1])
    a = d.atr = float(atr(df).iloc[-1])
    if a <= 0:
        d.reasons.append("no volatility data")
        return d

    regime, _, regime_note = volatility_regime(df)
    d.regime = regime
    d.reasons.append(regime_note)
    if regime in ("wild", "unknown"):
        d.reasons.append(f"=> no trade: environment is '{regime}', bounces aren't reliable here")
        return d

    # ---- gather zones: price-inferred + live order-book walls
    zones = swing_zones(df)
    supports = [(lvl, float(n)) for lvl, n in zones if lvl < price]
    resists = [(lvl, float(n)) for lvl, n in zones if lvl > price]
    if walls:
        for lvl, strength in walls.get("supports", [])[:5]:
            if lvl < price and price - lvl < 8 * a:
                supports.append((lvl, min(strength, 8.0)))
                d.reasons.append(f"book wall: buys {strength:.0f}x normal size at {fmt_price(lvl)}")
        for lvl, strength in walls.get("resists", [])[:5]:
            if lvl > price and lvl - price < 8 * a:
                resists.append((lvl, min(strength, 8.0)))
                d.reasons.append(f"book wall: sells {strength:.0f}x normal size at {fmt_price(lvl)}")
    if not supports and not resists:
        d.reasons.append("=> no trade: no meaningful order zones found nearby")
        return d

    # backtested: price-inferred zones alone have zero edge. So when we can see
    # the real book (live crypto), only trust zones backed by an actual wall.
    wall_levels = None
    if walls is not None:
        wall_levels = [lvl for lvl, _ in walls.get("supports", []) + walls.get("resists", [])]

    sup = max(supports, key=lambda z: z[0], default=None)   # nearest below
    res = min(resists, key=lambda z: z[0], default=None)    # nearest above
    at_support = sup is not None and price - sup[0] <= config.BOUNCE_ZONE_ATR * a
    at_resistance = res is not None and res[0] - price <= config.BOUNCE_ZONE_ATR * a

    if at_support and at_resistance:
        d.reasons.append("=> no trade: squeezed between two zones, no room either way")
        return d
    if not at_support and not at_resistance:
        near = []
        if sup:
            near.append(f"support {fmt_price(sup[0])} ({(price - sup[0]) / a:.1f} ATRs below)")
        if res:
            near.append(f"resistance {fmt_price(res[0])} ({(res[0] - price) / a:.1f} ATRs above)")
        d.reasons.append("nearest zones: " + ", ".join(near))
        d.reasons.append("=> no trade: price is mid-cloud (50/50 territory) - waiting for a zone")
        return d

    # ---- we are AT a zone: build the bounce trade
    last, prev = df.iloc[-1], df.iloc[-2]
    if at_support:
        action, zone, strength = "long", sup[0], sup[1]
        rejected = last["close"] > last["open"] or min(last["low"], prev["low"]) <= zone + 0.2 * a
        tape_against = tape is not None and tape < -0.4
        stop = zone - 0.6 * a
        target = (res[0] - 0.2 * a) if res else price + 3.0 * a
        room = target - price
        risk = price - stop
        side_note = f"support zone {fmt_price(zone)} (strength {strength:.0f})"
    else:
        action, zone, strength = "short", res[0], res[1]
        rejected = last["close"] < last["open"] or max(last["high"], prev["high"]) >= zone - 0.2 * a
        tape_against = tape is not None and tape > 0.4
        stop = zone + 0.6 * a
        target = (sup[0] + 0.2 * a) if sup else price - 3.0 * a
        room = price - target
        risk = stop - price
        side_note = f"resistance zone {fmt_price(zone)} (strength {strength:.0f})"

    d.reasons.append(f"price is at {side_note}, stop {fmt_price(stop)}, target {fmt_price(target)}")

    if wall_levels is not None and not any(abs(zone - w) <= 0.4 * a for w in wall_levels):
        d.reasons.append("=> no trade: no live order-book wall behind this zone "
                         "(price-only zones backtested to zero edge)")
        return d

    if action == "short" and not config.ALLOW_SHORTS:
        d.reasons.append("=> no trade: bounce is a short and shorting is disabled")
        return d
    if not rejected:
        d.reasons.append("=> no trade: at the zone but no rejection yet - the wall may be getting eaten")
        return d
    if tape_against:
        d.reasons.append(f"=> no trade: tape shows heavy aggression INTO the zone "
                         f"({tape:+.0%}) - walls that get hit like this often break")
        return d
    if risk <= 0 or room < config.BOUNCE_MIN_RR * risk:
        rr = room / risk if risk > 0 else 0.0
        d.reasons.append(f"=> no trade: reward/risk only {rr:.1f}x "
                         f"(need {config.BOUNCE_MIN_RR}x) - not asymmetric enough")
        return d
    if room < 1.2 * a:
        d.reasons.append("=> no trade: target too close to clear fees")
        return d

    d.action = action
    d.tp, d.sl = target, stop
    d.confidence = round(min(0.5 + 0.06 * min(strength, 5) + 0.05 * min(room / risk, 4), 0.95), 2)
    d.reasons.append(f"=> {action.upper()}: bounce off {side_note}, "
                     f"reward/risk {room / risk:.1f}x, room {room / a:.1f} ATRs to the empty side")
    return d
