"""Live market-neutral cointegration pairs bot (Alpaca paper).

Standalone strategy — does NOT share logic with the hourly directional bot.
Validated by pairs_walkforward.py: +86.8 bps/trade out-of-sample, 93% of
periods positive, 22% top-period concentration on the focused universe, with
Bonferroni-corrected pair selection (no false-positive harvesting).

HONEST STATUS: positive out-of-sample but LOW-FREQUENCY (~26-42 trades/yr)
and small-sample, so magnitude is uncertain. It is market-neutral (long one
leg, short beta x the other), so it sidesteps the microstructure/latency
wall. Paper only until it builds a live record.

Mechanics:
  * every REFRESH_HOURS, re-select cointegrated pairs from a trailing window
    (correlation pre-filter + Bonferroni-corrected Engle-Granger). Pairs that
    stop being cointegrated simply aren't reselected = the decoupling monitor.
  * each hourly scan: enter a pair's spread at |z|>2 (long the underperformer,
    short the outperformer), exit at |z|<0.5 or stop at |z|>4.
  * dollar-neutral sizing, max MAX_PAIRS concurrent, daily loss halt.

Usage: python pairs_trade.py            (loop)
       python pairs_trade.py --once     (single pass)
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

import botstate
import config
from data import download_bars
from pairs_walkforward import (ENTRY_Z, EXIT_Z, STOP_Z, UNIVERSE, Z_WIN,
                               select_pairs, SELECT_BARS)

MAX_PAIRS = 4
PAIR_NOTIONAL_FRAC = 0.10     # ~10% of equity per leg (both legs ~ market-neutral)
REFRESH_HOURS = 24
DAILY_LOSS_LIMIT = 0.02
SCAN_SECONDS = 3600


def get_client() -> TradingClient:
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        sys.exit("Missing Alpaca keys in .env")
    return TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
                         paper=True)


def load_prices(tickers, days=90) -> pd.DataFrame:
    cols = {}
    for t in tickers:
        try:
            cols[t] = download_bars(t, days=days)["close"]
        except Exception as e:
            print(f"  {t}: price load failed ({e})")
    return pd.DataFrame(cols).dropna()


def zscore(spread: pd.Series) -> float:
    m = spread.rolling(Z_WIN, min_periods=20).mean().iloc[-1]
    s = spread.rolling(Z_WIN, min_periods=20).std().iloc[-1]
    return float((spread.iloc[-1] - m) / s) if s and np.isfinite(s) else np.nan


def pair_key(a, b):
    return f"{a}|{b}"


TERMINAL = ("filled", "canceled", "expired", "rejected")
FAILED = ("canceled", "expired", "rejected")


def _status(o) -> str:
    return str(getattr(o.status, "value", o.status))


def _reverse(client, symbol, side, qty):
    """Submit an order that undoes `qty` shares of a leg traded on `side`.
    Returns the new order's id (or None if the submit failed)."""
    reverse = OrderSide.BUY if side == OrderSide.SELL else OrderSide.SELL
    try:
        o = client.submit_order(MarketOrderRequest(symbol=symbol, qty=qty, side=reverse,
                                                   time_in_force=TimeInForce.DAY))
        print(f"    unwinding {symbol}: reverse order for {qty:g} shares submitted")
        return str(getattr(o, "id", "")) or None
    except Exception as e:
        print(f"    reverse of {qty:g} {symbol} failed: {e} (will retry)")
        return None


def _held_qty(client, symbol, side):
    """Shares currently held in the direction a leg traded on `side` (BUY -> long,
    SELL -> short). 0 if flat / opposite; None if it couldn't be checked."""
    try:
        pos = client.get_open_position(symbol)
    except Exception as e:
        if "position does not exist" in str(e).lower() or "404" in str(e):
            return 0.0
        return None
    qty = float(pos.qty)
    long_side = side == OrderSide.BUY
    return abs(qty) if (qty > 0) == long_side else 0.0


# In-memory mirror of the orphan list: if the file can't be written (locked by
# OneDrive) or read, the records are still tracked in this process and the
# file is retried on every save - a filled leg can't silently drop out.
_orphans_mem: dict = {}


def _load_orphans() -> dict:
    try:
        disk = botstate.load_strict(botstate.PAIRS_ORPHANS, {})
    except botstate.StateUnavailable as e:
        print(f"  orphan file unreadable ({e}) - using the in-memory copy")
        disk = {}
    merged = dict(disk)
    merged.update(_orphans_mem)            # memory holds the newest records
    return merged


def _save_orphans(orphans: dict) -> bool:
    _orphans_mem.clear()
    _orphans_mem.update(orphans)
    return botstate.save(botstate.PAIRS_ORPHANS, orphans)


def unwind_leg(client, order_id, symbol, side) -> bool:
    """Undo a leg by ORDER: cancel what hasn't filled and reverse exactly what
    did. The leg goes on a persisted orphan list right away; resolve_orphans()
    keeps working it until the leg order is terminal AND the reversing orders
    have actually FILLED (an accepted reverse order can still be rejected or
    expire) - so a leg is never left naked and untracked."""
    try:
        client.cancel_order_by_id(order_id)
    except Exception:
        pass   # already filled / not cancelable
    orphans = _load_orphans()
    # setdefault: if this leg is ALREADY being unwound, keep its record - wiping
    # reverse_orders here made a second call submit a duplicate reverse order
    orphans.setdefault(str(order_id), {"symbol": symbol, "side": side.value, "reverse_orders": []})
    persisted = _save_orphans(orphans)
    for _ in range(5):                        # give the cancel/fill a moment, then work it
        resolve_orphans(client)
        if str(order_id) not in _load_orphans():
            return True                       # fully resolved
        time.sleep(1)
    print(f"    {symbol}: unwind not confirmed yet - tracking it until resolved")
    return persisted       # False: the record exists only in memory so far


def resolve_orphans(client) -> None:
    """Drive every pending unwind to completion. An orphan is removed only when
    its leg order is terminal, no reversing order is still working, and the
    FILLED reversing quantity covers everything the leg filled."""
    orphans = _load_orphans()
    if not orphans:
        return
    for oid, o in list(orphans.items()):
        try:
            leg = client.get_order_by_id(oid)
        except Exception as e:
            print(f"  orphan {o['symbol']}: status check failed ({e}), retry next scan")
            continue
        leg_filled, leg_status = float(leg.filled_qty or 0), _status(leg)
        reversed_filled, reverse_pending = 0.0, False
        for rid in o["reverse_orders"]:
            try:
                r = client.get_order_by_id(rid)
                reversed_filled += float(r.filled_qty or 0)
                reverse_pending |= _status(r) not in TERMINAL
            except Exception:
                reverse_pending = True        # unknown -> wait, never double-reverse
        need = leg_filled - reversed_filled
        if need > 0 and not reverse_pending:
            # never reverse more than is actually still held in that direction:
            # if the shares were closed by hand / liquidated, a blind reverse
            # would OPEN a new opposite position
            held = _held_qty(client, o["symbol"], OrderSide(o["side"]))
            if held is not None and held < need:
                print(f"  orphan {o['symbol']}: only {held:g} of {need:g} still held - reversing that")
                need = held
            rid = _reverse(client, o["symbol"], OrderSide(o["side"]), need) if need > 0 else None
            if rid:
                o["reverse_orders"].append(rid)
                reverse_pending = True
        if leg_status in TERMINAL and not reverse_pending and need <= 0:
            orphans.pop(oid)
            print(f"  orphan {o['symbol']}: fully unwound")
    _save_orphans(orphans)


def reconcile_entries(client, open_pairs: dict, can_trade: bool = True) -> None:
    """A pair is recorded when both entry orders are ACCEPTED; confirm they both
    FILL. If one ends rejected/expired/partially filled, unwind both legs (the
    orphan list keeps tracking them) and drop the pair - an accepted-then-
    rejected short used to leave the other leg naked indefinitely."""
    changed = False
    for key, st in list(open_pairs.items()):
        if not st.get("entry_pending"):
            continue
        a, b = key.split("|")
        try:
            oa, ob = client.get_order_by_id(st["oa"]), client.get_order_by_id(st["ob"])
        except Exception as e:
            print(f"  {key}: entry status check failed ({e}), retry next scan")
            continue
        full_a = _status(oa) == "filled" and float(oa.filled_qty or 0) >= st["qa"]
        full_b = _status(ob) == "filled" and float(ob.filled_qty or 0) >= st["qb"]
        if full_a and full_b:
            st["entry_pending"] = False
            changed = True
            print(f"  {key}: both legs filled")
        elif (_status(oa) in FAILED or _status(ob) in FAILED
              or (_status(oa) in TERMINAL and _status(ob) in TERMINAL)):
            # a leg FAILED (or both finished without both filling fully). One
            # leg being "filled" while the other still works is normal - wait.
            if not can_trade:
                # don't send after-hours reverse orders (they'd fill at the next
                # open on a stale price) - unwind on the first scan after the open
                print(f"  {key}: entry failed (A {_status(oa)}, B {_status(ob)}) - unwinding at the open")
                continue
            print(f"  {key}: entry failed (A {_status(oa)}, B {_status(ob)}) - unwinding both legs")
            ok_a = unwind_leg(client, st["oa"], a, OrderSide(st["sa"]))
            ok_b = unwind_leg(client, st["ob"], b, OrderSide(st["sb"]))
            if ok_a and ok_b:
                open_pairs.pop(key)    # legs are safely on the (saved) orphan list
            else:
                # keep the pair on the books until the orphan records are saved:
                # otherwise a filled leg would be in NEITHER state file
                print(f"  {key}: orphan records not saved yet - keeping the pair until they are")
            changed = True
    if changed:
        botstate.save(botstate.PAIRS_STATE, open_pairs)


def release_guards(open_pairs: dict) -> None:
    """Drop placeholder entries once their unwind record is safely ON DISK (or
    already fully resolved) - until then they keep the symbols off-limits to
    the other bots."""
    try:
        disk = botstate.load_strict(botstate.PAIRS_ORPHANS, {})
    except botstate.StateUnavailable:
        return
    changed = False
    for key, st in list(open_pairs.items()):
        if st.get("guard_only") and (st["oa"] in disk or st["oa"] not in _orphans_mem):
            open_pairs.pop(key)
            changed = True
    if changed:
        botstate.save(botstate.PAIRS_STATE, open_pairs)


def manage_exits(client, open_pairs: dict, prices, position_syms: set) -> None:
    """Close pairs whose spread converged (|z| < EXIT_Z) or blew out (> STOP_Z).
    A pair whose close left a leg open is flagged `closing` and retried EVERY
    scan until flat - never left waiting for z to cross a threshold again."""
    for key in list(open_pairs):
        a, b = key.split("|")
        st = open_pairs[key]
        if st.get("entry_pending") or st.get("guard_only"):
            continue          # reconcile_entries / release_guards own these
        closing = st.get("closing", False)
        if closing and a not in position_syms and b not in position_syms:
            # both legs confirmed gone (a close order being ACCEPTED isn't enough:
            # it can still be rejected/expire, which used to orphan the leg)
            open_pairs.pop(key, None)
            botstate.save(botstate.PAIRS_STATE, open_pairs)
            print(f"  CLOSED {key} (both legs confirmed flat)")
            continue
        if not closing:
            if a not in prices or b not in prices:
                continue
            spread = prices[a] - st["beta"] * prices[b]
            z = zscore(spread)
            if not np.isfinite(z) or EXIT_Z <= abs(z) <= STOP_Z:
                continue
            reason = "converged" if abs(z) < EXIT_Z else "stop"
            print(f"  CLOSE {key} (z={z:+.2f}, {reason})")
        else:
            # a previous close hasn't flattened every leg yet: keep trying EVERY
            # scan (waiting for z to cross a threshold again left one leg naked)
            print(f"  RETRY CLOSE {key} (leg still open)")
        for leg in (a, b):
            if leg not in position_syms:
                continue          # already flat (e.g. closed by hand)
            try:
                client.close_position(leg)
            except Exception as e:
                print(f"    close error {leg}: {e} (will retry next scan)")
        st["closing"] = True      # popped only once both legs are confirmed flat
        botstate.save(botstate.PAIRS_STATE, open_pairs)


def main():
    once = "--once" in sys.argv
    client = get_client()
    print(f"Pairs bot connected -- equity "
          f"${float(client.get_account().equity):,.2f}  "
          f"(market-neutral, low-frequency, paper experiment)")

    selected: list[tuple] = []      # (a, b, beta)
    # persisted: after a restart the bot must still manage (and eventually
    # close) pairs it opened - otherwise both legs are orphaned forever
    try:
        open_pairs: dict[str, dict] = botstate.load_strict(botstate.PAIRS_STATE, {})
    except botstate.StateUnavailable as e:
        # starting with {} would overwrite the file and abandon open pairs
        sys.exit(f"pairs state unreadable ({e}) - not starting; try again in a moment")
    if open_pairs:
        print(f"  resumed {len(open_pairs)} open pair(s): {list(open_pairs)}")
    last_refresh = 0.0

    while True:
      try:
        now = time.time()
        # --- periodic Bonferroni-corrected re-selection (decoupling monitor) ---
        if now - last_refresh > REFRESH_HOURS * 3600 or not selected:
            px = load_prices(UNIVERSE, days=160)  # need > SELECT_BARS(500) bars
            if len(px) > SELECT_BARS:
                selected = select_pairs(px.iloc[-SELECT_BARS:])
                last_refresh = now
                print(f"\n[{datetime.now(timezone.utc):%Y-%m-%d %H:%M}] "
                      f"re-selected {len(selected)} cointegrated pairs: "
                      f"{[(a, b) for a, b, _ in selected]}")
            else:
                print(f"  only {len(px)} aligned bars (<{SELECT_BARS}); "
                      f"cannot select yet")

        # daily loss halt
        acct = client.get_account()
        eq, last = float(acct.equity), float(acct.last_equity)
        buying_power = float(getattr(acct, "buying_power", eq) or eq)
        halted = last > 0 and eq < last * (1 - DAILY_LOSS_LIMIT)
        if halted:
            print("  daily loss limit hit -- no new pair entries today")

        needed = {t for a, b, _ in selected for t in (a, b)} \
            | {t for k in open_pairs for t in k.split("|")}
        prices = load_prices(sorted(needed), days=30)
        position_syms = {p.symbol for p in client.get_all_positions()}
        held_syms = set(position_syms)
        # also symbols with PENDING orders (e.g. the stock bot's unfilled entry):
        # a pair leg opened there would merge into the other bot's position
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest
            held_syms |= {o.symbol for o in client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))}
        except Exception as e:
            print(f"  open-order check failed ({e}) - skipping new pair entries this scan")
            halted = True

        # --- finish unwinds, confirm entries filled, then manage exits ---
        # orders only while the market is OPEN: after-hours DAY market orders
        # queue until the open and fill on a signal hours stale
        try:
            market_open = bool(client.get_clock().is_open)
        except Exception as e:
            print(f"  clock check failed ({e}) - no orders this scan")
            market_open = False
        reconcile_entries(client, open_pairs, can_trade=market_open)
        release_guards(open_pairs)
        if market_open:
            resolve_orphans(client)
            manage_exits(client, open_pairs, prices, position_syms)
        else:
            print("  market closed - entries/exits wait for the open")
            halted = True

        # --- entries ---
        if not halted:
            for a, b, beta in selected:
                key = pair_key(a, b)
                if key in open_pairs or len(open_pairs) >= MAX_PAIRS:
                    continue
                if a in held_syms or b in held_syms:
                    continue  # leg already tied up in another pair
                if a not in prices or b not in prices:
                    continue
                spread = prices[a] - beta * prices[b]
                z = zscore(spread)
                if not np.isfinite(z) or abs(z) < ENTRY_Z:
                    continue
                if beta <= 0:
                    continue      # hedge needs a positive ratio for long/short legs
                leg = eq * PAIR_NOTIONAL_FRAC
                pa = prices[a].iloc[-1]
                # The walk-forward traded the SPREAD a - beta*b: 1 share of A vs
                # beta shares of B. Size it that way (equal dollars is a
                # different, untested hedge). Whole shares: Alpaca rejects
                # fractional SHORT orders, which used to leave one naked leg.
                pb = prices[b].iloc[-1]
                # neither leg may exceed the per-leg budget (beta*pb can be far
                # larger than pa, which used to put several x the budget into B)
                qa = int(leg / max(pa, beta * pb))
                qb = int(beta * qa)          # floor: rounding up could exceed the budget
                if qa < 1 or qb < 1:
                    continue
                if qa * pa + qb * pb > buying_power * 0.95:
                    print(f"  skip {key}: needs ${qa * pa + qb * pb:,.0f}, buying power ${buying_power:,.0f}")
                    continue
                # z>0: spread rich -> short A, long B ; z<0: mirror
                if z > 0:
                    sa, sb = OrderSide.SELL, OrderSide.BUY
                else:
                    sa, sb = OrderSide.BUY, OrderSide.SELL
                try:
                    order_a = client.submit_order(MarketOrderRequest(
                        symbol=a, qty=qa, side=sa,
                        time_in_force=TimeInForce.DAY))
                except Exception as e:
                    print(f"    entry error {key} (leg {a}): {e}")
                    continue
                try:
                    order_b = client.submit_order(MarketOrderRequest(
                        symbol=b, qty=qb, side=sb,
                        time_in_force=TimeInForce.DAY))
                except Exception as e:
                    # second leg failed -> undo the first so we're not left
                    # holding a one-sided directional bet
                    print(f"    entry error {key} (leg {b}): {e} -- unwinding {a}")
                    if not unwind_leg(client, order_a.id, a, sa):
                        # record only in memory so far: keep the leg visible to
                        # the other bots by holding a placeholder pair entry
                        # guard_only: NOT a tradable pair - manage_exits skips it
                        # (closing it too would double-sell A alongside the unwind)
                        open_pairs[key] = {"guard_only": True, "oa": str(order_a.id),
                                           "note": "leg B failed - unwinding A"}
                        botstate.save(botstate.PAIRS_STATE, open_pairs)
                    continue
                # accepted is not filled: reconcile_entries confirms both legs fill
                open_pairs[key] = {"beta": beta, "entry_z": z, "qa": qa, "qb": qb,
                                   "oa": str(order_a.id), "ob": str(order_b.id),
                                   "sa": sa.value, "sb": sb.value, "entry_pending": True}
                buying_power -= qa * pa + qb * pb
                botstate.save(botstate.PAIRS_STATE, open_pairs)
                held_syms |= {a, b}
                print(f"  OPEN {key}  z={z:+.2f}  "
                      f"{sa.value} {qa} {a} / {sb.value} {qb} {b}")

        if once:
            print(f"  open pairs: {list(open_pairs)}")
            return
        time.sleep(SCAN_SECONDS)
      except Exception as e:
        print(f"  loop error (retry 60s): {e}")
        time.sleep(60)


if __name__ == "__main__":
    main()
