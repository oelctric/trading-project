"""Paper-trading account: fake money, real prices, honest bookkeeping.

State survives restarts (paper_state.json) and every closed trade is appended
to trades.csv so you can audit whether the strategy is actually profitable.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import config
from brain import fmt_price
from logger import get_logger

log = get_logger("portfolio")

STATE_FILE = os.path.join(os.path.dirname(__file__), "paper_state.json")
TRADES_FILE = os.path.join(os.path.dirname(__file__), "trades.csv")


@dataclass
class Position:
    symbol: str
    side: str            # "long" | "short"
    qty: float
    entry: float
    take_profit: float
    stop_loss: float
    opened_at: str       # ISO timestamp
    confidence: float
    reason: str
    max_hold: float = 0.0   # minutes; 0 = use config.MAX_HOLD_MINUTES
    exit_state: dict | None = None  # dynamic exit manager state (swing strategy)
    margin: float = 0.0     # cash posted for this position (0 = legacy: full notional)
    leverage: float = 1.0
    risk_usd: float = 0.0   # $ lost if the initial stop is hit
    fee_rate: float = 0.0   # per side; 0 = use config.FEE_PER_SIDE

    def posted(self) -> float:
        return self.margin or self.qty * self.entry

    def days_held(self) -> float:
        return self.age_minutes() / 1440

    def age_minutes(self) -> float:
        opened = datetime.fromisoformat(self.opened_at)
        return (datetime.now(timezone.utc) - opened).total_seconds() / 60

    def pnl(self, price: float) -> float:
        sign = 1 if self.side == "long" else -1
        return (price - self.entry) * self.qty * sign


class Portfolio:
    def __init__(self) -> None:
        self.cash = config.STARTING_CASH
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.peak_equity = self.cash
        self.recent_r: list[float] = []   # R-multiples of recent closed trades (cold-streak brake)
        self._load()

    # ------------------------------------------------------------ persistence
    def _load(self) -> None:
        if not os.path.exists(STATE_FILE):
            return
        with open(STATE_FILE) as f:
            state = json.load(f)
        self.cash = state["cash"]
        self.realized_pnl = state["realized_pnl"]
        self.wins = state["wins"]
        self.losses = state["losses"]
        self.positions = {s: Position(**p) for s, p in state["positions"].items()}
        self.peak_equity = state.get("peak_equity", max(self.cash, config.STARTING_CASH))
        self.recent_r = state.get("recent_r", [])

    def save(self) -> None:
        state = {
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "wins": self.wins,
            "losses": self.losses,
            "positions": {s: asdict(p) for s, p in self.positions.items()},
            "peak_equity": self.peak_equity,
            "recent_r": self.recent_r[-50:],
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    # ------------------------------------------------------------ trading
    def equity(self, prices: dict[str, float]) -> float:
        """Cash + each position's posted margin + unrealized P&L (isolated
        margin: a position can't be worth less than zero)."""
        eq = self.cash
        for sym, pos in self.positions.items():
            price = prices.get(sym, pos.entry)
            value = pos.posted() + pos.pnl(price)
            # isolated-margin futures can't be worth < 0; a spot/unlevered short can
            eq += max(0.0, value) if pos.fee_rate else value
        return eq

    def book_equity(self) -> float:
        """Cash + posted margin (no mark-to-market) - what position sizing uses."""
        return self.cash + sum(p.posted() for p in self.positions.values())

    def can_open(self, symbol: str) -> bool:
        """Check if symbol can be traded now.

        A symbol can be open if not already held and a position slot is open.
        Also checks against correlated group limits.
        """
        if symbol in self.positions:
            return False
        if len(self.positions) >= config.MAX_OPEN_POSITIONS:
            return False

        # Correlated group limits
        for group_name, symbols in config.CORRELATED_GROUPS.items():
            if symbol in symbols:
                current_held = sum(1 for s in self.positions if s in symbols)
                limit = (config.SWING_MAX_POSITIONS_PER_GROUP
                         if config.STRATEGY == config.Strategy.SWING
                         else config.MAX_POSITIONS_PER_GROUP)
                if current_held >= limit:
                    log.info("cannot open %s: too many %s positions", symbol, group_name)
                    return False
        return True

    def open(self, symbol: str, side: str, price: float, atr: float,
             confidence: float, reason: str,
             tp: float = 0.0, sl: float = 0.0, max_hold: float = 0.0,
             exit_state: dict | None = None) -> Position | None:
        """Open a paper position. tp/sl default to ATR multiples unless given.
        With exit_state (swing strategy) there is no fixed target or time limit:
        the dynamic exit manager decides when to leave."""
        if not self.can_open(symbol) or atr <= 0 or price <= 0:
            return None
        sign = 1 if side == "long" else -1
        take_profit = 0.0 if exit_state else (tp or price + sign * atr * config.TAKE_PROFIT_ATR)
        stop_loss = sl or price - sign * atr * config.STOP_LOSS_ATR
        stop_dist = abs(price - stop_loss)
        if stop_dist <= 0:
            return None
        equity = self.book_equity()
        lev, risk_usd, fee_rate, notes = 1.0, 0.0, 0.0, []
        # leverage / futures pricing is for the SWING strategy's paper futures
        # account only - the minute strategies trade spot (it used to apply to all)
        if getattr(config, "USE_LEVERAGE", False) and exit_state:
            from leverage import size_position
            sz = size_position(config.LEVERAGE, equity, self.peak_equity, self.recent_r,
                               stop_dist / price,
                               sum(p.risk_usd for p in self.positions.values()),
                               sum(p.qty * p.entry for p in self.positions.values()),
                               self.cash)
            if sz["notional"] <= 0:
                log.info("skip %s: %s", symbol, "; ".join(sz["notes"]) or "no size")
                return None
            qty, cost, lev = sz["notional"] / price, sz["margin"], sz["leverage"]
            risk_usd, fee_rate, notes = sz["risk"], config.FUTURES_FEE_PER_SIDE, sz["notes"]
        else:
            swing = exit_state is not None
            risk = config.SWING_RISK_PER_TRADE if swing else config.RISK_PER_TRADE
            cap = config.SWING_MAX_POSITION_PCT if swing else config.MAX_POSITION_PCT
            qty = equity * risk / stop_dist
            qty = min(qty, equity * cap / price)
            if not swing and qty * price > self.cash:
                return None   # minute strategies: skip (as tested) rather than open undersized
            qty = min(qty, self.cash / price)
            cost = qty * price
            risk_usd = qty * stop_dist
            if qty <= 0 or cost < getattr(config, "MIN_ORDER_USD", 0.0):
                return None
        if notes:
            reason = reason + " | sizing: " + "; ".join(notes)
        pos = Position(
            symbol=symbol, side=side, qty=qty, entry=price,
            take_profit=take_profit, stop_loss=stop_loss,
            opened_at=datetime.now(timezone.utc).isoformat(),
            confidence=confidence, reason=reason, max_hold=max_hold,
            exit_state=exit_state,
            margin=cost, leverage=lev, risk_usd=risk_usd, fee_rate=fee_rate,
        )
        self.cash -= cost
        self.positions[symbol] = pos
        self.save()
        return pos

    def liquidation_price(self, symbol: str) -> float | None:
        """Price at which the exchange would liquidate this (isolated) position."""
        pos = self.positions.get(symbol)
        if not pos or pos.leverage <= 1:
            return None
        sign = 1 if pos.side == "long" else -1
        mm = config.LEVERAGE.maintenance_margin
        return pos.entry * (1 - sign * (1 / pos.leverage - mm))

    def check_liquidation(self, symbol: str, price: float) -> bool:
        liq = self.liquidation_price(symbol)
        if liq is None:
            return False
        side = self.positions[symbol].side
        return price <= liq if side == "long" else price >= liq

    def check_exit(self, symbol: str, price: float) -> str | None:
        """Return an exit reason if this position should close now."""
        pos = self.positions.get(symbol)
        if not pos or pos.exit_state:   # swing positions are managed by swing_live.manage
            return None
        if pos.side == "long":
            if price >= pos.take_profit:
                return "take profit"
            if price <= pos.stop_loss:
                return "stop loss"
        else:
            if price <= pos.take_profit:
                return "take profit"
            if price >= pos.stop_loss:
                return "stop loss"
        limit = pos.max_hold or config.MAX_HOLD_MINUTES
        if pos.age_minutes() >= limit:
            return f"time limit ({limit:.0f} min)"
        return None

    def close(self, symbol: str, price: float, why: str) -> float:
        """Close position and record results.

        Returns final realized P&L (after fees).
        """
        pos = self.positions.pop(symbol)
        kind = "crypto" if symbol in config.CRYPTO_SYMBOLS or symbol.endswith("-USD") else "stock"
        fee = (pos.fee_rate or config.FEE_PER_SIDE[kind]) * pos.qty * (pos.entry + price)
        funding = 0.0
        if pos.fee_rate:   # futures position: pay funding for every day held
            funding = config.FUNDING_PER_DAY * pos.qty * pos.entry * max(pos.days_held(), 1 / 24)
        pnl = pos.pnl(price) - fee - funding
        if pos.fee_rate:   # isolated-margin futures only: loss capped at what was posted
            pnl = max(pnl, -pos.posted())
        log.info("CLOSED %s %s @ %s (P&L %.2f, reason: %s)", symbol, pos.side, fmt_price(price), pnl, why)
        self.cash += pos.posted() + pnl
        if pos.risk_usd and pos.exit_state:   # swing trades only: the brake sizes swing risk
            self.recent_r.append(pnl / pos.risk_usd)
        self.peak_equity = max(self.peak_equity, self.book_equity())
        self.realized_pnl += pnl
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1
        self._log_trade(pos, price, pnl, why)
        self.save()
        return pnl

    TRADE_COLUMNS = ["closed_at", "symbol", "side", "qty", "entry", "exit", "pnl",
                     "hold_minutes", "exit_reason", "confidence", "entry_reason",
                     "strategy", "r_multiple", "risk_usd", "leverage", "margin"]

    def _log_trade(self, pos: Position, exit_price: float, pnl: float, why: str) -> None:
        # files written before the strategy/R columns existed are moved aside once
        if os.path.exists(TRADES_FILE):
            with open(TRADES_FILE, newline="") as f:
                header = next(csv.reader(f), [])
            if header and header != self.TRADE_COLUMNS:
                old = TRADES_FILE.replace(".csv", "_old_format.csv")
                if os.path.exists(old):   # never append under a mismatched header
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    old = TRADES_FILE.replace(".csv", f"_old_format_{stamp}.csv")
                os.replace(TRADES_FILE, old)
                log.info("moved old-format trade log to %s", os.path.basename(old))
        new_file = not os.path.exists(TRADES_FILE)
        strategy = "swing" if pos.exit_state else "minute"
        r_mult = pnl / pos.risk_usd if pos.risk_usd else 0.0
        with open(TRADES_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(self.TRADE_COLUMNS)
            w.writerow([
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                pos.symbol, pos.side, f"{pos.qty:.6f}", f"{pos.entry:.4f}",
                f"{exit_price:.4f}", f"{pnl:+.2f}", f"{pos.age_minutes():.1f}",
                why, f"{pos.confidence:.2f}", pos.reason,
                strategy, f"{r_mult:+.3f}", f"{pos.risk_usd:.2f}", f"{pos.leverage:.2f}",
                f"{pos.posted():.2f}",
            ])
