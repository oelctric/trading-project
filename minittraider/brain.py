"""The decision brain: situational awareness through confluence.

It never trades on one signal. It looks at all the independent signals, checks
the volatility regime, and asks: is independent evidence lining up right now?

- Signals conflict, or the market is dead/erratic  ->  it sits out and says why.
- Several signals align AND the environment is tradeable  ->  confidence rises
  above the threshold and it takes the scalp.

Every decision comes with a plain-English explanation, so you can always see
what it "understood" about the situation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import config
from signals import ALL_SIGNALS, volatility_regime, atr, _clip


def fmt_price(p: float) -> str:
    """Human-friendly price: 62,173.09 for BTC, 0.07248 for DOGE."""
    return f"{p:,.2f}" if p >= 100 else f"{p:.4g}"


@dataclass
class Decision:
    symbol: str
    action: str                 # "long", "short", or "none"
    confidence: float           # 0..1
    price: float = 0.0
    atr: float = 0.0
    regime: str = ""
    tp: float = 0.0             # explicit target/stop levels (0 = derive from ATR)
    sl: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def explain(self) -> str:
        head = {
            "long": f"LONG {self.symbol} @ {fmt_price(self.price)} (confidence {self.confidence:.0%})",
            "short": f"SHORT {self.symbol} @ {fmt_price(self.price)} (confidence {self.confidence:.0%})",
            "none": f"{self.symbol}: sitting out",
        }[self.action]
        return head + "\n    " + "\n    ".join(self.reasons)


def micro_signals(micro: dict) -> list[tuple[str, float, str]]:
    """Signals from live information beyond this symbol's own price history."""
    out = []
    book = micro.get("book")
    if book is not None:
        side = "bids" if book > 0 else "asks"
        out.append(("book", _clip(book / 0.4),
                    f"order book: {side} {abs(book):.0%} heavier near the price"))
    tape = micro.get("tape")
    if tape is not None:
        side = "buying" if tape > 0 else "selling"
        out.append(("tape", _clip(tape / 0.5),
                    f"tape: {abs(tape):.0%} net aggressive {side} in recent trades"))
    leader = micro.get("leader")
    if leader is not None:
        name = micro.get("leader_name", "leader")
        out.append(("leader", _clip(leader / 1.5),
                    f"{name} moved {leader:+.1f} ATRs in 3 min (tends to lead this market)"))
    return out


def decide(symbol: str, df: pd.DataFrame, micro: dict | None = None) -> Decision:
    d = Decision(symbol=symbol, action="none", confidence=0.0)
    if df is None or len(df) < 60:
        d.reasons.append("not enough candle history yet")
        return d

    d.price = float(df["close"].iloc[-1])
    d.atr = float(atr(df).iloc[-1])

    # 1. Understand the environment first.
    regime, regime_mult, regime_note = volatility_regime(df)
    d.regime = regime
    d.reasons.append(regime_note)

    # 2. Gather independent evidence.
    scores: dict[str, float] = {}
    for name, fn in ALL_SIGNALS.items():
        score, note = fn(df)
        scores[name] = score
        d.reasons.append(f"[{name}] {note} ({score:+.2f})")
    for name, score, note in micro_signals(micro or {}):
        scores[name] = score
        d.reasons.append(f"[{name}] {note} ({score:+.2f})")

    bulls = sum(1 for s in scores.values() if s > 0.25)
    bears = sum(1 for s in scores.values() if s < -0.25)
    avg = sum(scores.values()) / len(scores)

    # 3. Confluence: how strongly does the evidence line up?
    direction = "long" if avg > 0 else "short"
    agreeing = bulls if direction == "long" else bears
    opposing = bears if direction == "long" else bulls
    alignment = agreeing / len(scores)              # fraction of signals on board
    conflict_penalty = 0.20 * opposing              # disagreement is a warning
    confidence = max(0.0, (abs(avg) * 0.5 + alignment * 0.5 - conflict_penalty)) * regime_mult
    d.confidence = round(min(confidence, 0.99), 2)

    # 4. Only act when things genuinely align.
    if regime in ("dead", "wild", "unknown"):
        d.reasons.append(f"=> no trade: environment is '{regime}', edge is unreliable here")
        return d
    if agreeing < config.MIN_AGREEING_SIGNALS:
        d.reasons.append(
            f"=> no trade: only {agreeing} signal(s) agree "
            f"(need {config.MIN_AGREEING_SIGNALS}) - situation is unclear"
        )
        return d
    if d.confidence < config.MIN_CONFIDENCE:
        d.reasons.append(
            f"=> no trade: confidence {d.confidence:.0%} below the "
            f"{config.MIN_CONFIDENCE:.0%} bar - evidence isn't strong enough"
        )
        return d
    if direction == "short" and not config.ALLOW_SHORTS:
        d.reasons.append("=> no trade: setup is bearish but shorting is disabled")
        return d

    d.action = direction
    d.reasons.append(
        f"=> {direction.upper()}: {agreeing} of {len(scores)} signals align, "
        f"{opposing} against, regime '{regime}' - evidence is lining up"
    )
    return d
