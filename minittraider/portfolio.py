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

    def save(self) -> None:
        state = {
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "wins": self.wins,
            "losses": self.losses,
            "positions": {s: asdict(p) for s, p in self.positions.items()},
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    # ------------------------------------------------------------ trading
    def equity(self, prices: dict[str, float]) -> float:
        """Return current account equity.

        Note:’ la position is open, equity is cash + the mark-to-market
        value of positions.
        """
        eq = self.cash
        for sym, pos in self.positions.items():
            price = prices.get(sym, pos.entry)
            sign = 1 if pos.side == "long" else -1
            # For shorts: equity = cash +’ (entry - price) * qty + cost of long (entry * qty)
            #’ effectively we hold the short asset’s value
            eq += pos.qty * pos.entry + pos.pnl(price)
        return eq

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
                if current_held >= config.MAX_POSITIONS_PER_GROUP:
                    log.info("cannot open %s: too many %s positions", symbol, group_name)
                    return False
        return True

    def open(self, symbol: str, side: str, price: float, atr: float,
             confidence: float, reason: str,
             tp: float = 0.0, sl: float = 0.0, max_hold: float = 0.0) -> Position | None:
        """Open a paper position. tp/sl default to ATR multiples unless given."""
        if not self.can_open(symbol) or atr <= 0 or price <= 0:
            return None
        sign = 1 if side == "long" else -1
        take_profit = tp or price + sign * atr * config.TAKE_PROFIT_ATR
        stop_loss = sl or price - sign * atr * config.STOP_LOSS_ATR
        stop_dist = abs(price - stop_loss)
        if stop_dist <= 0:
            return None
        equity = self.cash + sum(p.qty * p.entry for p in self.positions.values())
        qty = equity * config.RISK_PER_TRADE / stop_dist
        qty = min(qty, equity * config.MAX_POSITION_PCT / price)
        cost = qty * price
        if cost > self.cash or qty <= 0:
            return None
        pos = Position(
            symbol=symbol, side=side, qty=qty, entry=price,
            take_profit=take_profit, stop_loss=stop_loss,
            opened_at=datetime.now(timezone.utc).isoformat(),
            confidence=confidence, reason=reason, max_hold=max_hold,
        )
        self.cash -= cost
        self.positions[symbol] = pos
        self.save()
        return pos

    def check_exit(self, symbol: str, price: float) -> str | None:
        """Return an exit reason if this position should close now."""
        pos = self.positions.get(symbol)
        if not pos:
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
        fee = config.FEE_PER_SIDE[kind] * pos.qty * (pos.entry + price)
        pnl = pos.pnl(price) - fee
        log.info("CLOSED %s %s @ %s (P&L %.2f, reason: %s)", symbol, pos.side, fmt_price(price), pnl, why)
        self.cash += pos.qty * pos.entry + pnl
        self.realized_pnl += pnl
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1
        self._log_trade(pos, price, pnl, why)
        self.save()
        return pnl

    def _log_trade(self, pos: Position, exit_price: float, pnl: float, why: str) -> None:
        new_file = not os.path.exists(TRADES_FILE)
        with open(TRADES_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["closed_at", "symbol", "side", "qty", "entry", "exit",
                            "pnl", "hold_minutes", "exit_reason", "confidence", "entry_reason"])
            w.writerow([
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                pos.symbol, pos.side, f"{pos.qty:.6f}", f"{pos.entry:.4f}",
                f"{exit_price:.4f}", f"{pnl:+.2f}", f"{pos.age_minutes():.1f}",
                why, f"{pos.confidence:.2f}", pos.reason,
            ])
