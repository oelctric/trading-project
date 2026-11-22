"""Perception layer: translate raw broker data into a fuzzy state the machine
can reason about.

The planner and sub-agents never see raw prices. They see a State: continuous
degrees like trend_up=0.7, oversold=0.4, near_support=0.9, and a statistical
estimate of whether the nearest liquidity wall will hold or break.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bounce import swing_zones
from fuzzy import clamp, ramp_up, tri
from signals import atr, ema, rsi, volatility_regime


@dataclass
class State:
    """Everything downstream layers are allowed to know about one symbol now."""
    price: float = 0.0
    atr: float = 0.0
    hour: int = 0                 # UTC hour, for time-of-day awareness
    regime: str = ""

    # raw context the sub-agents need for concrete levels
    trend_score: float = 0.0      # -1 (strong down) .. +1 (strong up)
    boll_z: float = 0.0           # z-score vs 20-bar mean, in std devs
    rsi: float = 50.0
    ema21_dist: float = 0.0       # ATRs from the 21-EMA
    support: float | None = None  # nearest zone below / above (2+ touches)
    resistance: float | None = None

    # fuzzy memberships in [0, 1]
    fuzz: dict = field(default_factory=dict)

    # order-book statistics (live crypto only; 0.5 = no information)
    support_hold_prob: float = 0.5
    resistance_hold_prob: float = 0.5

    notes: list = field(default_factory=list)


def wall_hold_probability(wall_size: float, df: pd.DataFrame, horizon: int = 10) -> float:
    """Probability the wall absorbs incoming flow instead of breaking.

    Statistical model: over the next `horizon` minutes, expected taker flow is
    roughly recent volume-per-minute * horizon. A wall survives that flow with
    probability wall / (wall + expected_flow) - a simple odds model: a wall 3x
    the expected flow holds ~75% of the time, a wall equal to the flow is 50/50.
    """
    flow = float(df["volume"].iloc[-15:].mean()) * horizon
    if flow <= 0:
        return 0.5
    return clamp(wall_size / (wall_size + flow), 0.05, 0.95)


def perceive(df: pd.DataFrame, walls: dict | None = None,
             tape: float | None = None) -> State:
    s = State()
    close = df["close"]
    s.price = float(close.iloc[-1])
    s.atr = float(atr(df).iloc[-1]) or 1e-9
    s.hour = int(df.index[-1].hour)
    s.regime, _, _ = volatility_regime(df)

    # --- trend: agreement and separation of the EMA stack, in ATRs
    e9, e21, e50 = (float(ema(close, n).iloc[-1]) for n in (9, 21, 50))
    s.trend_score = float(np.clip(((e9 - e21) + (e21 - e50)) / (2 * s.atr), -1, 1))
    s.ema21_dist = (s.price - e21) / s.atr

    # --- stretch: how far from the recent mean, in standard deviations
    win = close.iloc[-20:]
    std = float(win.std()) or 1e-9
    s.boll_z = float((s.price - win.mean()) / std)
    s.rsi = float(rsi(close).iloc[-1])
    if np.isnan(s.rsi):
        s.rsi = 50.0

    # --- structure: nearest meaningful zones
    zones = swing_zones(df)
    sups = [lvl for lvl, _ in zones if lvl < s.price]
    ress = [lvl for lvl, _ in zones if lvl > s.price]
    s.support = max(sups) if sups else None
    s.resistance = min(ress) if ress else None

    # --- order book: statistical hold-probability of the nearest walls
    if walls:
        near_sup = [(lvl, st) for lvl, st in walls.get("supports", [])
                    if s.price - lvl < 3 * s.atr]
        near_res = [(lvl, st) for lvl, st in walls.get("resists", [])
                    if lvl - s.price < 3 * s.atr]
        med_flow = float(df["volume"].iloc[-15:].mean()) or 1e-9
        if near_sup:
            s.support_hold_prob = wall_hold_probability(near_sup[0][1] * med_flow, df)
            if s.support is None or abs(near_sup[0][0] - s.support) > 0.5 * s.atr:
                s.support = near_sup[0][0]
        if near_res:
            s.resistance_hold_prob = wall_hold_probability(near_res[0][1] * med_flow, df)
            if s.resistance is None or abs(near_res[0][0] - s.resistance) > 0.5 * s.atr:
                s.resistance = near_res[0][0]

    # --- fuzzy memberships: the vocabulary the reasoning core speaks
    vol_surge = float(df["volume"].iloc[-1] / (df["volume"].iloc[-21:-1].mean() or 1e-9))
    f = s.fuzz
    f["trend_up"] = ramp_up(s.trend_score, 0.05, 0.6)
    f["trend_down"] = ramp_up(-s.trend_score, 0.05, 0.6)
    f["ranging"] = tri(s.trend_score, -0.25, 0.0, 0.25)
    f["overbought"] = ramp_up(s.rsi, 60, 80) * ramp_up(s.boll_z, 0.5, 2.0)
    f["oversold"] = ramp_up(-s.rsi + 100, 60, 80) * ramp_up(-s.boll_z, 0.5, 2.0)
    f["stretched"] = ramp_up(abs(s.boll_z), 1.0, 2.5)
    f["near_support"] = (ramp_up(-(s.price - s.support) / s.atr + 1.0, 0.2, 1.0)
                         if s.support is not None else 0.0)
    f["near_resistance"] = (ramp_up(-(s.resistance - s.price) / s.atr + 1.0, 0.2, 1.0)
                            if s.resistance is not None else 0.0)
    f["at_ema"] = tri(abs(s.ema21_dist), -0.6, 0.0, 0.6)
    f["volume_active"] = ramp_up(vol_surge, 1.0, 2.5)
    f["buyers_aggressive"] = ramp_up(tape, 0.1, 0.6) if tape is not None else 0.0
    f["sellers_aggressive"] = ramp_up(-tape, 0.1, 0.6) if tape is not None else 0.0
    f["support_solid"] = ramp_up(s.support_hold_prob, 0.5, 0.85)
    f["resistance_solid"] = ramp_up(s.resistance_hold_prob, 0.5, 0.85)
    return s
