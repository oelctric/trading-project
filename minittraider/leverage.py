"""Leverage-aware position sizing, shared by the backtest and the live bot.

Leverage is just borrowing, so the question is never "how much can I borrow"
but "how much should I risk on THIS trade, right now". The sizer answers that
with four rules, then turns the answer into a position and a safe leverage:

  1. base risk       risk a fixed % of equity if the initial stop is hit
  2. drawdown brake  scale risk down while the account is below its peak
                     (losing streaks are when over-betting kills accounts)
  3. hot-hand brake  if the strategy's recent trades are net losing, halve risk
                     (trend following has long cold regimes; don't press then)
  4. heat cap        total open risk across ALL positions is capped, because the
                     coins move together - 6 trades at 2% is really one 12% bet

Then:
  - position notional = risk$ / stop distance
  - total notional is capped at MAX_LEVERAGE x equity
  - per-position (isolated) leverage is capped so the liquidation price sits at
    least LIQ_BUFFER x further away than the stop - a gap through the stop
    can't liquidate the position
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LeverageConfig:
    base_risk: float = 0.02         # fraction of equity lost if the initial stop hits
    max_leverage: float = 3.0       # total notional / equity
    heat_cap: float = 0.08          # max summed open risk (fraction of equity)
    dd_brake_start: float = 0.10    # start cutting risk at 10% below peak...
    dd_brake_full: float = 0.30     # ...down to dd_brake_floor x risk at 30% below
    dd_brake_floor: float = 0.25
    cold_lookback: int = 10         # trades
    cold_factor: float = 0.5        # risk multiplier while recent trades are net losing
    liq_buffer: float = 2.0         # liquidation distance >= this x stop distance
    maintenance_margin: float = 0.005
    min_order: float = 1.0


def risk_fraction(cfg: LeverageConfig, equity: float, peak: float,
                  recent_r: list[float]) -> tuple[float, list[str]]:
    """How much of equity to risk on the next trade, with the reasons."""
    notes = []
    r = cfg.base_risk
    dd = 1 - equity / peak if peak > 0 else 0.0
    if dd > cfg.dd_brake_start:
        span = max(cfg.dd_brake_full - cfg.dd_brake_start, 1e-9)
        k = 1 - (1 - cfg.dd_brake_floor) * min(1.0, (dd - cfg.dd_brake_start) / span)
        r *= k
        notes.append(f"drawdown brake: {dd:.0%} below peak -> risk x{k:.2f}")
    recent = recent_r[-cfg.cold_lookback:]
    if len(recent) >= cfg.cold_lookback and sum(recent) < 0:
        r *= cfg.cold_factor
        notes.append(f"cold streak: last {len(recent)} trades {sum(recent):+.1f}R -> risk x{cfg.cold_factor}")
    return r, notes


def size_position(cfg: LeverageConfig, equity: float, peak: float, recent_r: list[float],
                  stop_pct: float, open_risk: float, open_notional: float,
                  free_cash: float, quality_mult: float = 1.0) -> dict:
    """Returns {notional, margin, leverage, risk, notes} - notional 0 means skip.

    stop_pct      distance to the initial stop as a fraction of entry price
    open_risk     summed $ at risk of the positions already open
    open_notional summed notional of open positions
    free_cash     cash not already posted as margin
    """
    out = {"notional": 0.0, "margin": 0.0, "leverage": 0.0, "risk": 0.0, "notes": []}
    if equity <= 0 or stop_pct <= 0:
        return out
    r, notes = risk_fraction(cfg, equity, peak, recent_r)
    if quality_mult != 1.0:
        r *= quality_mult
        notes.append(f"trade-quality model: risk x{quality_mult:g}")
    room = cfg.heat_cap - open_risk / equity
    if room <= 0.001:
        out["notes"] = notes + [f"heat cap: {open_risk / equity:.1%} already at risk"]
        return out
    if r > room:
        notes.append(f"heat cap: risk trimmed {r:.1%} -> {room:.1%}")
        r = room
    notional = equity * r / stop_pct
    notional = min(notional, max(0.0, equity * cfg.max_leverage - open_notional))
    # isolated leverage that keeps liquidation well beyond the stop
    safe_lev = 1.0 / (cfg.liq_buffer * stop_pct + cfg.maintenance_margin)
    lev = max(1.0, min(cfg.max_leverage, safe_lev))
    margin = notional / lev
    if margin > free_cash:
        margin = free_cash
        notional = margin * lev
    if notional < cfg.min_order:
        out["notes"] = notes + ["position below minimum order"]
        return out
    out.update(notional=notional, margin=margin, leverage=lev,
               risk=notional * stop_pct, notes=notes)
    return out
