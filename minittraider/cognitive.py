"""Cognitive trading architecture: perception -> planner -> sub-agent -> risk veto.

  MetaAgent  (planner)   tabular Q-learning over discretized macro states.
                         Picks a strategy regime: TREND / REVERT / STAND_ASIDE.
                         It never places trades. Trained by train_cognitive.py;
                         the learned policy lives in meta_q.json.

  Sub-agents (executors) receive the mandate and use fuzzy inference to decide
                         whether a concrete, high-probability setup exists, and
                         build the exact entry/stop/target plan.

  RiskManager (breaker)  hard-coded, overrides everyone: reward/risk floor,
                         daily-drawdown circuit breaker, loss-streak cooldown,
                         position sizing. No agent can outvote it.

The learnable part is deliberately tiny (a few hundred Q-values, not millions
of weights) - it can prioritize and abstain, but it cannot memorize price paths.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import config
from brain import Decision, fmt_price
from fuzzy import AND, NOT, FuzzySystem, OR, Rule
from perception import State, perceive

Q_FILE = os.path.join(os.path.dirname(__file__), "meta_q.json")

ACTIONS = ["TREND", "REVERT", "STAND_ASIDE"]


# ---------------------------------------------------------------- planner

def macro_state_key(s: State) -> str:
    """Discretize the macro environment. Coarse on purpose: ~400 states total,
    so every state gets visited many times during training - no memorizing."""
    t = s.trend_score
    trend_b = 0 if t < config.TREND_SCORE_BINS[0] else 1 if t < config.TREND_SCORE_BINS[1] else 2 if t < config.TREND_SCORE_BINS[2] else 3 if t < config.TREND_SCORE_BINS[3] else 4
    vol_b = {"dead": 0, "normal": 1, "active": 2, "wild": 3}.get(s.regime, 1)
    z = s.boll_z
    stretch_b = 0 if z < config.STRETCH_BINS[0] else 1 if z < config.STRETCH_BINS[1] else 2 if z < config.STRETCH_BINS[2] else 3 if z < config.STRETCH_BINS[3] else 4
    hour_b = s.hour // 6
    return f"{trend_b}:{vol_b}:{stretch_b}:{hour_b}"


class MetaAgent:
    """High-level planner: which strategy regime fits this environment?"""

    def __init__(self, path: str = Q_FILE):
        self.q: dict[str, list[float]] = {}
        self.trained = False
        if os.path.exists(path):
            with open(path) as f:
                blob = json.load(f)
            self.q = blob["q"]
            self.trained = True

    def values(self, key: str) -> list[float]:
        return self.q.setdefault(key, [0.0, 0.0, 0.0])

    def choose(self, s: State) -> tuple[str, str]:
        """Greedy policy + a human-readable justification."""
        key = macro_state_key(s)
        vals = self.values(key)
        best = max(range(3), key=lambda a: vals[a])
        if not self.trained:
            # untrained fallback: modest hand heuristic until training runs
            if s.regime in ("wild", "unknown"):
                return "STAND_ASIDE", "untrained planner: environment too erratic"
            if s.fuzz.get("ranging", 0) > 0.5:
                return "REVERT", "untrained planner: ranging market -> mean-reversion mode"
            if max(s.fuzz.get("trend_up", 0), s.fuzz.get("trend_down", 0)) > 0.5:
                return "TREND", "untrained planner: trending market -> trend mode"
            return "STAND_ASIDE", "untrained planner: unclear environment"
        if vals[best] <= 0.0:
            return "STAND_ASIDE", (f"planner: learned value of every strategy in state "
                                   f"[{key}] is <= 0 - this environment doesn't pay")
        return ACTIONS[best], (f"planner: state [{key}] -> {ACTIONS[best]} mode "
                               f"(learned values T={vals[0]:+.2f} R={vals[1]:+.2f} S={vals[2]:+.2f})")

    def save(self, path: str = Q_FILE, meta: dict | None = None) -> None:
        with open(path, "w") as f:
            json.dump({"q": self.q, "meta": meta or {}}, f, indent=1)


# ---------------------------------------------------------------- sub-agents

# Mean-reversion executor: fade stretches into solid zones.
REVERT_RULES = FuzzySystem([
    Rule("oversold at solid support",
         lambda s: AND(s["oversold"], s["near_support"], s["support_solid"]), 1.0, 1.2),
    Rule("oversold at support",
         lambda s: AND(s["oversold"], s["near_support"]), 0.85),
    Rule("stretched down near support",
         lambda s: AND(s["stretched_down"], s["near_support"]), 0.7),
    Rule("overbought at solid resistance",
         lambda s: AND(s["overbought"], s["near_resistance"], s["resistance_solid"]), 1.0, 1.2),
    Rule("overbought at resistance",
         lambda s: AND(s["overbought"], s["near_resistance"]), 0.85),
    Rule("stretched up near resistance",
         lambda s: AND(s["stretched_up"], s["near_resistance"]), 0.7),
    Rule("aggression against the fade",
         lambda s: OR(AND(s["long_side"], s["sellers_aggressive"]),
                      AND(NOT(s["long_side"]), s["buyers_aggressive"])), 0.0, 1.5),
    Rule("no zone to lean on",
         lambda s: NOT(OR(s["near_support"], s["near_resistance"])), 0.0),
])

# Trend executor: join an established trend on a pullback, not on a spike.
TREND_RULES = FuzzySystem([
    Rule("uptrend pullback to the mean",
         lambda s: AND(s["trend_up"], s["at_ema"], NOT(s["overbought"])), 1.0),
    Rule("downtrend pullback to the mean",
         lambda s: AND(s["trend_down"], s["at_ema"], NOT(s["oversold"])), 1.0),
    Rule("trend with participation",
         lambda s: AND(OR(s["trend_up"], s["trend_down"]), s["volume_active"]), 0.8),
    Rule("tape agrees with the trend",
         lambda s: OR(AND(s["trend_up"], s["buyers_aggressive"]),
                      AND(s["trend_down"], s["sellers_aggressive"])), 0.9),
    Rule("chasing a stretched move",
         lambda s: AND(OR(s["trend_up"], s["trend_down"]), s["stretched"]), 0.0, 1.3),
    Rule("no real trend",
         lambda s: s["ranging"], 0.0),
])

CONFIDENCE_BAR = 0.55


def _revert_plan(s: State) -> tuple[str, float, float, str] | None:
    """(side, stop, target, note) for mean-reversion, or None."""
    if s.fuzz["near_support"] >= s.fuzz["near_resistance"] and s.support is not None:
        stop = s.support - 0.6 * s.atr
        target = (s.resistance - 0.2 * s.atr) if s.resistance else s.price + 2.0 * s.atr
        return "long", stop, target, f"fade into support {fmt_price(s.support)}"
    if s.resistance is not None:
        stop = s.resistance + 0.6 * s.atr
        target = (s.support + 0.2 * s.atr) if s.support else s.price - 2.0 * s.atr
        return "short", stop, target, f"fade into resistance {fmt_price(s.resistance)}"
    return None


def _trend_plan(s: State) -> tuple[str, float, float, str] | None:
    """(side, stop, target, note) for trend continuation, or None."""
    if s.trend_score > 0:
        stop = (s.support - 0.4 * s.atr) if s.support and s.price - s.support < 2.5 * s.atr \
            else s.price - 1.2 * s.atr
        risk = s.price - stop
        return "long", stop, s.price + 2.2 * risk, "join uptrend on pullback"
    stop = (s.resistance + 0.4 * s.atr) if s.resistance and s.resistance - s.price < 2.5 * s.atr \
        else s.price + 1.2 * s.atr
    risk = stop - s.price
    return "short", stop, s.price - 2.2 * risk, "join downtrend on pullback"


def run_sub_agent(mode: str, s: State) -> tuple[float, list, tuple | None]:
    """Returns (fuzzy confidence, fired rules, plan)."""
    plan = _revert_plan(s) if mode == "REVERT" else _trend_plan(s)
    if plan is None:
        return 0.0, [], None
    fz = dict(s.fuzz)
    fz["long_side"] = 1.0 if plan[0] == "long" else 0.0
    fz["stretched_down"] = fz["stretched"] if s.boll_z < 0 else 0.0
    fz["stretched_up"] = fz["stretched"] if s.boll_z > 0 else 0.0
    system = REVERT_RULES if mode == "REVERT" else TREND_RULES
    confidence, fired = system.infer(fz)
    return confidence, fired, plan


# ---------------------------------------------------------------- risk manager

class RiskManager:
    """Hard-coded circuit breaker. Overrides every agent, no exceptions."""

    MIN_RR = 1.5
    MAX_DAILY_LOSS_PCT = 0.03     # stop trading for the day at -3% equity
    LOSS_STREAK_LIMIT = 3         # after 3 straight losses, cool down
    COOLDOWN_MINUTES = 45

    def __init__(self) -> None:
        self.day = None
        self.day_start_equity = 0.0
        self.loss_streak = 0
        self.cooldown_until: datetime | None = None

    def record(self, pnl: float, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        self.loss_streak = self.loss_streak + 1 if pnl < 0 else 0
        if self.loss_streak >= self.LOSS_STREAK_LIMIT:
            self.cooldown_until = now + timedelta(minutes=self.COOLDOWN_MINUTES)

    def veto(self, side: str, price: float, stop: float, target: float,
             equity: float, now: datetime | None = None) -> str | None:
        """Returns a veto reason, or None if the trade may proceed."""
        now = now or datetime.now(timezone.utc)
        if self.day != now.date():
            self.day, self.day_start_equity = now.date(), equity
        if equity < self.day_start_equity * (1 - self.MAX_DAILY_LOSS_PCT):
            return (f"daily circuit breaker: down "
                    f"{1 - equity / self.day_start_equity:.1%} today, no more trades")
        if self.cooldown_until and now < self.cooldown_until:
            return (f"cooldown after {self.LOSS_STREAK_LIMIT} straight losses "
                    f"(until {self.cooldown_until.strftime('%H:%M')} UTC)")
        sign = 1 if side == "long" else -1
        risk = sign * (price - stop)
        reward = sign * (target - price)
        if risk <= 0 or reward <= 0:
            return "stop/target on the wrong side of price"
        if reward < self.MIN_RR * risk:
            return f"reward/risk {reward / risk:.1f}x below the {self.MIN_RR}x floor"
        return None


# ---------------------------------------------------------------- full pipeline

def decide_cognitive(symbol: str, df, walls=None, tape=None,
                     meta: MetaAgent | None = None,
                     risk: RiskManager | None = None,
                     equity: float = 10_000.0) -> Decision:
    d = Decision(symbol=symbol, action="none", confidence=0.0)
    if df is None or len(df) < 60:
        d.reasons.append("not enough candle history yet")
        return d

    # 1. perceive
    s = perceive(df, walls, tape)
    d.price, d.atr, d.regime = s.price, s.atr, s.regime
    top = sorted(s.fuzz.items(), key=lambda kv: -kv[1])[:4]
    d.reasons.append("perception: " + ", ".join(f"{k}={v:.2f}" for k, v in top if v > 0.05))

    # 2. plan
    meta = meta or _default_meta()
    mode, why = meta.choose(s)
    d.reasons.append(why)
    if mode == "STAND_ASIDE":
        d.reasons.append("=> no trade: planner chose to stand aside")
        return d

    # 3. execute (fuzzy reasoning over the concrete setup)
    confidence, fired, plan = run_sub_agent(mode, s)
    if fired:
        d.reasons.append(f"{mode.lower()} agent rules: "
                         + "; ".join(f"{n} ({v:.2f})" for n, v in fired[:3]))
    if plan is None or confidence < CONFIDENCE_BAR:
        d.reasons.append(f"=> no trade: {mode.lower()} agent confidence "
                         f"{confidence:.2f} below {CONFIDENCE_BAR} - setup not clean enough")
        return d
    side, stop, target, note = plan

    # 4. risk veto - the only layer with absolute authority
    risk = risk or _default_risk()
    bar_time = df.index[-1].to_pydatetime() if hasattr(df.index[-1], "to_pydatetime") else None
    veto = risk.veto(side, s.price, stop, target, equity, now=bar_time)
    if veto:
        d.reasons.append(f"=> BLOCKED by risk manager: {veto}")
        return d

    d.action, d.tp, d.sl = side, target, stop
    d.confidence = round(min(confidence, 0.95), 2)
    d.reasons.append(f"=> {side.upper()} ({mode} mode): {note}, "
                     f"stop {fmt_price(stop)}, target {fmt_price(target)}, "
                     f"fuzzy confidence {confidence:.2f}")
    return d


_META: MetaAgent | None = None
_RISK: RiskManager | None = None


def _default_meta() -> MetaAgent:
    global _META
    if _META is None:
        _META = MetaAgent()
    return _META


def _default_risk() -> RiskManager:
    global _RISK
    if _RISK is None:
        _RISK = RiskManager()
    return _RISK


def record_result(pnl: float) -> None:
    """Feed closed-trade results back to the risk manager's streak/cooldown logic."""
    _default_risk().record(pnl)
