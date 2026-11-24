"""Live paper-trading bot for Alpaca — 8 strategies, regime-gated,
risk-managed.

Every hour during market hours it:
  1. pulls fresh bars per ticker (same pipeline as training) and computes
     each strategy's trailing simulated performance (the regime gate: a
     strategy may only open trades if its recent trades were net profitable)
  2. finds fired setups (long and short), ranks them with the NN ensemble
     (longs by prob, shorts by 1 - prob)
  3. enters the best candidates up to MAX_POSITIONS, max MAX_PER_SECTOR per
     sector, skipping the first SKIP_OPEN_MINUTES of the session, using
     marketable-limit bracket orders (ATR-scaled TP/SL) sized for equal
     dollar risk per trade
  4. halts for the day if equity drops DAILY_LOSS_LIMIT below yesterday's
     close, and flattens everything 15 minutes before the close

Usage: python trade.py            (loop until stopped)
       python trade.py --once     (single pass, useful for testing)
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import torch
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (GetOrdersRequest, LimitOrderRequest,
                                     ReplaceOrderRequest, StopLossRequest,
                                     TakeProfitRequest)

import botstate
import config
from data import FEATURE_COLUMNS, _client, compute_features, download_bars
from manipulation import bar_flags
from model import Scaler, ensemble_probs, load_ensemble
from strategies import STRATEGY_NAMES, is_short


def quote_imbalance(symbol: str) -> float | None:
    """Live top-of-book imbalance: bid_size / (bid_size + ask_size).
    > 0.5 means more resting buy liquidity at the touch (bullish wall)."""
    try:
        from alpaca.data.requests import StockLatestQuoteRequest
        q = _client().get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
        total = float(q.bid_size) + float(q.ask_size)
        return float(q.bid_size) / total if total > 0 else None
    except Exception:
        return None

FLATTEN_MINUTES_BEFORE_CLOSE = 15
# Scan aligned to hourly-bar availability: the just-closed bar is queryable
# ~16 min after the hour (free-feed SIP lag), so wake at :18 past. This
# catches each new bar promptly instead of drifting on a fixed timer.
SCAN_BAR_MINUTE = 18


def seconds_to_next_scan(now: datetime, next_close: datetime) -> float:
    """Sleep until the next hourly bar is available (:SCAN_BAR_MINUTE past),
    capped at the pre-close flatten time so we never sleep past it."""
    nxt = now.replace(minute=SCAN_BAR_MINUTE, second=0, microsecond=0)
    while nxt <= now:
        nxt += timedelta(hours=1)
    flatten_at = next_close - timedelta(minutes=FLATTEN_MINUTES_BEFORE_CLOSE)
    target = min(nxt, flatten_at)
    return max(60.0, (target - now).total_seconds())


def local(dt: datetime) -> str:
    """Render any timestamp in the machine's local timezone (auto-detected)."""
    return dt.astimezone().strftime("%Y-%m-%d %I:%M %p %Z")


def get_client() -> TradingClient:
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        sys.exit("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in .env "
                 "-- add them and rerun.")
    return TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
                         paper=True)


def scan(models, scaler):
    """One pass over the universe.

    Returns (signals, gated_in): per-ticker dict with NN prob, fired
    setups, price, ATR; and the set of strategies whose trailing simulated
    trades were net profitable (always everything if the gate is off).
    """
    signals = {}
    regime_pnl = {s: 0.0 for s in STRATEGY_NAMES}
    regime_pnl_fast = {s: 0.0 for s in STRATEGY_NAMES}
    regime_wins = {s: 0 for s in STRATEGY_NAMES}
    regime_n = {s: 0 for s in STRATEGY_NAMES}
    regime_sumsq = {s: 0.0 for s in STRATEGY_NAMES}   # for mean/std gate
    # CALENDAR-day windows, exactly as walkforward.py validated them (this
    # used to be N*7 hourly bars = N TRADING days, ~40% longer than tested)
    now_ts = pd.Timestamp.now(tz="America/New_York")
    slow_start = now_ts - pd.Timedelta(days=config.REGIME_LOOKBACK_DAYS)
    fast_start = now_ts - pd.Timedelta(days=config.REGIME_LOOKBACK_FAST_DAYS)
    for ticker in config.TICKERS:
        try:
            df = download_bars(ticker, days=config.LIVE_LOOKBACK_DAYS)
            feats = compute_features(df)
            if len(feats) < config.SEQ_LEN:
                continue
            recent = feats[feats.index >= slow_start]
            for name in STRATEGY_NAMES:
                # Gate on TRAILING-stop returns — the exit the bot actually
                # trades — not the fixed bracket. Raw-sum dual-window rule
                # is the most robust for stocks (walkforward: 75% months+,
                # 9% single-month concentration; risk-adjusted metrics were
                # tested and did not help this non-lumpy book).
                rcol = "trail_ret_short" if is_short(name) else "trail_ret"
                fired = recent[(recent[f"setup_{name}"] > 0)
                               & recent[rcol].notna()]
                rets = fired[rcol] - config.COST_PER_TRADE
                regime_pnl[name] += float(rets.sum())
                regime_pnl_fast[name] += float(
                    rets[fired.index >= fast_start].sum())
                regime_wins[name] += int((rets > 0).sum())
                regime_n[name] += int(len(rets))
                regime_sumsq[name] += float((rets ** 2).sum())
            last = feats.iloc[-1]
            window = feats[FEATURE_COLUMNS].tail(config.SEQ_LEN) \
                .to_numpy("float32")[None, ...]
            prob = ensemble_probs(
                models, torch.from_numpy(scaler.transform(window))).item()
            fake_bo = bool(bar_flags(feats)["fake_breakout"].iloc[-1])
            signals[ticker] = {
                "prob": prob,
                "setups": [s for s in STRATEGY_NAMES
                           if last[f"setup_{s}"] > 0],
                "price": float(last["close"]),
                "atr_pct": float(last["atr_pct"]),
                "fake_breakout": fake_bo,
            }
        except Exception as e:
            print(f"  {ticker}: data/predict error: {e}")

    if config.REGIME_GATE:
        # Dual lookback (positive over slow AND fast window) PLUS, for the
        # expectancy+hit metric, a trailing win-rate floor. This generalized
        # best on unseen tickers (+9.5 vs +5.6 bps raw-sum).
        def passes(s):
            # Fast-decay cut applies to every metric.
            if not (regime_pnl[s] > 0 and regime_pnl_fast[s] > 0):
                return False
            n = regime_n[s]
            if n < 20:
                return False
            if config.GATE_METRIC == "sharpe":
                mean = regime_pnl[s] / n
                var = regime_sumsq[s] / n - mean * mean
                std = var ** 0.5 if var > 0 else 0.0
                return std > 0 and mean / std > config.GATE_SHARPE_MIN
            if config.GATE_METRIC == "expectancy_hit":
                return regime_wins[s] / n > config.GATE_HIT_FLOOR
            return True   # raw_sum
        gated_in = {s for s in STRATEGY_NAMES if passes(s)}
    else:
        gated_in = set(STRATEGY_NAMES)
    gate_str = ", ".join(
        f"{s}={regime_pnl[s]:+.2f}/{regime_pnl_fast[s]:+.2f}"
        for s in STRATEGY_NAMES)
    print(f"  regime pnl ({config.REGIME_LOOKBACK_DAYS}d/"
          f"{config.REGIME_LOOKBACK_FAST_DAYS}d): {gate_str}")
    print(f"  gated in: {sorted(gated_in) or 'none'}")
    return signals, gated_in


def cancel_stale_entries(client: TradingClient, pending: list[str]) -> None:
    """Cancel our own unfilled entry orders from the previous scan."""
    for oid in pending:
        try:
            order = client.get_order_by_id(oid)
            if order.filled_qty in (None, "0", 0):
                client.cancel_order_by_id(oid)
                print(f"  cancelled stale entry {order.symbol}")
        except Exception:
            pass
    pending.clear()


# Per-position trailing state (trail distance in $, high-water mark). Keyed
# by symbol; persisted to stock_trail.json so a restart keeps the real trail.
_trail: dict[str, dict] = botstate.load(botstate.STOCK_TRAIL, {})


def _stop_orders(open_orders) -> dict:
    """Stop orders by symbol. With nested=True, a bracket's stop LEG lives inside
    its parent's `legs`, not in the top-level list - look in both places."""
    stops = {}
    for o in open_orders:
        for leg in [o] + list(getattr(o, "legs", None) or []):
            status = str(getattr(leg.status, "value", leg.status))
            # allow-list of LIVE statuses (replaced / expired / pending_cancel
            # legs can appear too - trailing one of those fails silently)
            if leg.type is not None and str(leg.type.value) in ("stop", "stop_limit") \
                    and status in ("new", "accepted", "held", "pending_new"):
                prev = stops.get(leg.symbol)
                prev_status = str(getattr(prev.status, "value", prev.status)) if prev else ""
                if prev is None or (status in ("new", "accepted")
                                    and prev_status not in ("new", "accepted")):
                    stops[leg.symbol] = leg
    return stops


def trail_stops(client: TradingClient, positions) -> None:
    """Ratchet each position's stop to trail TRAIL_ATR_MULT ATR behind the
    best price seen — the validated exit that lets winners run. Subsumes the
    old breakeven move (the stop passes through entry as price advances)."""
    if not config.TRAILING_EXIT:
        return
    try:
        open_orders = client.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.OPEN, nested=True))
    except Exception:
        return
    stops = _stop_orders(open_orders)
    live_syms = {p.symbol for p in positions}
    for sym in list(_trail):
        if sym not in live_syms:
            del _trail[sym]  # position closed
    for p in positions:
        order = stops.get(p.symbol)
        if order is None or order.stop_price is None:
            continue
        entry = float(p.avg_entry_price)
        stop = float(order.stop_price)
        price = float(p.current_price)
        long_pos = float(p.qty) > 0
        st = _trail.get(p.symbol)
        if st is None:
            # No saved state (e.g. file deleted). If the stop is still on the
            # losing side of entry it's the original stop, so the initial
            # distance can be rebuilt. If it has already trailed past entry,
            # "entry - stop" no longer means anything (it can be ~0, which would
            # set the stop AT the price and sell immediately) - keep the
            # current gap between price and stop instead.
            still_initial = stop < entry if long_pos else stop > entry
            dist = (abs(entry - stop) * config.TRAIL_ATR_MULT / config.SL_ATR_MULT
                    if still_initial else abs(price - stop))
            st = {"dist": max(dist, 0.01), "hwm": price}
            _trail[p.symbol] = st
        st["hwm"] = max(st["hwm"], price) if long_pos else min(st["hwm"], price)
        new_stop = (st["hwm"] - st["dist"] if long_pos
                    else st["hwm"] + st["dist"])
        improved = new_stop > stop + 0.01 if long_pos else new_stop < stop - 0.01
        if improved:
            try:
                client.replace_order_by_id(order.id, ReplaceOrderRequest(
                    stop_price=round(new_stop, 2)))
                print(f"  {p.symbol}: trailing stop -> {new_stop:.2f}")
            except Exception as e:
                print(f"  {p.symbol}: trail update failed: {e}")
    botstate.save(botstate.STOCK_TRAIL, _trail)


def explain_trade(c: dict, imb: float | None, equity: float,
                  stop_dist: float, qty: float) -> str:
    """Plain-language rationale assembled from the actual decision factors —
    the trade's real 'reasoning', made legible."""
    conf = c["score"]
    conf_word = ("high" if conf > 0.55 else
                 "moderate" if conf > 0.50 else "low")
    parts = [
        f"{c['side']} {c['ticker']}: {'/'.join(c['why'])} setup fired "
        f"(regime-approved on both 30d & 90d windows)",
        f"network rank {conf:.2f} ({conf_word} confidence vs peers)",
    ]
    if imb is not None:
        side_book = imb if c["side"] == "long" else 1 - imb
        parts.append(f"order book {side_book:.0%} in favor")
    if c.get("fake_breakout"):
        parts.append("(note: fake-breakout flag present but not a "
                     "breakout-family setup)")
    risk_dollars = equity * config.RISK_PER_TRADE
    if config.TRAILING_EXIT:
        exit_desc = (f"initial stop {config.SL_ATR_MULT}xATR, then trails "
                     f"{config.TRAIL_ATR_MULT}xATR behind the best price "
                     f"(far {config.TP_BACKSTOP_ATR}xATR take-profit backstop)")
    else:
        exit_desc = (f"TP {config.TP_ATR_MULT}xATR / SL {config.SL_ATR_MULT}"
                     f"xATR, breakeven at +1R")
    parts.append(f"risking ${risk_dollars:,.0f} ({config.RISK_PER_TRADE:.2%} "
                 f"of equity); {exit_desc}")
    return "    reasoning: " + "; ".join(parts)


def rebalance(client: TradingClient, models, scaler,
              pending: list[str], allow_entries: bool) -> None:
    cancel_stale_entries(client, pending)
    account = client.get_account()
    equity = float(account.equity)
    buying_power = float(account.buying_power)   # day-trade margin (~4x)
    positions = client.get_all_positions()
    trail_stops(client, positions)
    try:
        pairs_legs = botstate.pairs_leg_symbols()
    except botstate.StateUnavailable as e:
        print(f"  pairs state unreadable ({e}) - no new entries this scan")
        return
    # the stock bot's own positions: not the pairs bot's legs, not the crypto
    # bot's coins (BTCUSD...) - those used to eat stock slots / sector caps
    own_held = {p.symbol for p in positions
                if not p.symbol.endswith("USD")} - pairs_legs
    held = {p.symbol for p in positions} | pairs_legs   # never enter these
    sector_count: dict[str, int] = {}
    for p in positions:
        if p.symbol not in own_held:
            continue          # pairs-bot legs don't count toward the stock bot's sector caps
        sec = config.SECTOR.get(p.symbol, "other")
        sector_count[sec] = sector_count.get(sec, 0) + 1

    signals, gated_in = scan(models, scaler)
    active = set(config.ACTIVE_STRATEGIES) & gated_in

    candidates = []
    for ticker, s in signals.items():
        if ticker in held:
            continue
        fired = [x for x in s["setups"] if x in active]
        if not fired:
            continue
        longs = [x for x in fired if not is_short(x)]
        shorts = [x for x in fired if is_short(x)]
        # a ticker firing both directions is a conflict — take the side
        # the NN favors
        if longs and (not shorts or s["prob"] >= 0.5):
            side, score, why = "long", s["prob"], longs
        elif shorts:
            side, score, why = "short", 1 - s["prob"], shorts
        else:
            continue
        candidates.append(s | {"ticker": ticker, "side": side,
                               "score": score, "why": why})
    candidates.sort(key=lambda c: -c["score"])

    shown = [f'{c["ticker"]}({c["side"]},{c["score"]:.3f})'
             for c in candidates[:8]]
    print(f"  candidates: {shown or 'none'}")
    if not allow_entries:
        print("  first-30-min window: no new entries")
        return

    # Leverage = more concurrent positions (diversified), capped by
    # available buying power below.
    max_positions = int(round(config.MAX_POSITIONS * config.LEVERAGE))
    slots = max(0, max_positions - len(own_held))   # pairs legs don't use stock-bot slots
    entered = 0
    for c in candidates:
        if entered >= slots:
            break
        sec = config.SECTOR.get(c["ticker"], "other")
        if sector_count.get(sec, 0) >= config.MAX_PER_SECTOR:
            continue
        # Manipulation defense: don't take a breakout-family trade on a bar
        # flagged as a fake breakout (a stop-run trap for breakout buyers).
        if (config.FAKE_BREAKOUT_FILTER and c.get("fake_breakout")
                and any(w in config.BREAKOUT_FAMILY for w in c["why"])):
            print(f"  skip {c['ticker']} ({c['side']}): fake-breakout trap")
            continue
        # Order-book check: don't step in front of a wall of opposing
        # liquidity at the touch.
        imb = quote_imbalance(c["ticker"])
        if imb is not None:
            book_support = imb if c["side"] == "long" else 1 - imb
            if book_support < config.QUOTE_IMBALANCE_MIN:
                print(f"  skip {c['ticker']} ({c['side']}): book against it "
                      f"(imbalance={imb:.2f})")
                continue
        price, atr = c["price"], c["atr_pct"]
        # With trailing exits, the take-profit is just a far backstop; the
        # trailing stop is the real exit.
        tp_mult = (config.TP_BACKSTOP_ATR if config.TRAILING_EXIT
                   else config.TP_ATR_MULT)
        if c["side"] == "long":
            limit = round(price * (1 + config.ENTRY_LIMIT_BUFFER), 2)
            tp = round(price * (1 + tp_mult * atr), 2)
            sl = round(price * (1 - config.SL_ATR_MULT * atr), 2)
            order_side = OrderSide.BUY
        else:
            limit = round(price * (1 - config.ENTRY_LIMIT_BUFFER), 2)
            tp = round(price * (1 - tp_mult * atr), 2)
            sl = round(price * (1 + config.SL_ATR_MULT * atr), 2)
            order_side = OrderSide.SELL
        stop_dist = abs(price - sl)
        if stop_dist <= 0:
            continue
        qty = int(equity * config.RISK_PER_TRADE / stop_dist)
        qty = min(qty, int(equity * config.MAX_NOTIONAL_FRACTION / price))
        # Buying-power guard: never let an order exceed remaining margin
        # buying power (leave a 5% cushion for slippage on the fill).
        qty = min(qty, int(buying_power * 0.95 / price))
        if qty < 1:
            continue
        buying_power -= qty * price
        try:
            order = client.submit_order(LimitOrderRequest(
                symbol=c["ticker"], qty=qty, side=order_side,
                limit_price=limit, time_in_force=TimeInForce.DAY,
                order_class="bracket",
                take_profit=TakeProfitRequest(limit_price=tp),
                stop_loss=StopLossRequest(stop_price=sl)))
            pending.append(str(order.id))
            _trail[c["ticker"]] = {"dist": config.TRAIL_ATR_MULT * atr * price,
                                   "hwm": price}
            botstate.save(botstate.STOCK_TRAIL, _trail)
            sector_count[sec] = sector_count.get(sec, 0) + 1
            entered += 1
            imb_str = f"{imb:.2f}" if imb is not None else "n/a"
            print(f"  {c['side'].upper()} {qty} {c['ticker']} lmt={limit}  "
                  f"tp={tp} sl={sl}  imb={imb_str}  ({','.join(c['why'])})")
            print(explain_trade(c, imb, equity, stop_dist, qty))
        except Exception as e:
            print(f"  order rejected for {c['ticker']}: {e}")
    if entered == 0:
        print("  no new entries")


def flatten_all(client: TradingClient, fallback: bool = False) -> bool:
    """Close the stock bot's own positions. Returns False if it couldn't tell
    which positions belong to the pairs bot (caller retries shortly).
    fallback=True (after repeated failures): close only the positions that are
    DEFINITELY the stock bot's - the ones it holds trailing-stop state for -
    so it never ends up holding overnight because a state file is unreadable."""
    # STOCK-ONLY: the account may be shared with the crypto bot, so never
    # close crypto positions (symbols like "BTCUSD") or cancel crypto orders
    # (symbols like "BTC/USD"). Only touch this bot's own stock symbols.
    # Also never touch the PAIRS bot's legs: it holds market-neutral pairs
    # overnight on this same account; closing one leg leaves a naked bet.
    print("  flattening stock positions")
    own_only = None
    try:
        pairs_legs = botstate.pairs_leg_symbols()
    except botstate.StateUnavailable as e:
        if not fallback:
            # don't guess: an empty set here would close the pairs bot's legs
            print(f"  pairs state unreadable ({e}) - retrying the flatten shortly")
            return False
        own_only = set(_trail)
        pairs_legs = set()
        print(f"  !!! WARNING: pairs state still unreadable ({e}) - closing ONLY the stock "
              f"bot's own tracked positions {sorted(own_only)}; check pairs_state.json !!!")
    try:
        for o in client.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.OPEN, limit=200)):
            if "/" not in o.symbol and o.symbol not in pairs_legs \
                    and (own_only is None or o.symbol in own_only):   # our stock order
                client.cancel_order_by_id(o.id)
    except Exception as e:
        print(f"  order cancel error: {e}")
    time.sleep(2)   # let cancellations settle so close_position doesn't race
    for attempt in range(3):
        stuck = []
        for p in client.get_all_positions():
            if p.symbol.endswith("USD") or p.symbol in pairs_legs:  # crypto / pairs -> skip
                continue
            if own_only is not None and p.symbol not in own_only:
                continue          # fallback: only what's definitely ours
            try:
                client.close_position(p.symbol)
            except Exception as e:
                stuck.append(p.symbol)
                print(f"  close {p.symbol} failed (try {attempt+1}): {e}")
        if not stuck:
            break
        time.sleep(2)
    return True


def daily_loss_hit(client: TradingClient) -> bool:
    try:
        account = client.get_account()
    except Exception as e:
        print(f"  loss-check skipped (transient API error): {e}")
        return False
    equity, last = float(account.equity), float(account.last_equity)
    if last > 0 and equity < last * (1 - config.DAILY_LOSS_LIMIT):
        print(f"  DAILY LOSS LIMIT: equity ${equity:,.0f} vs "
              f"yesterday ${last:,.0f} -- halting for the day")
        return True
    return False


def guarded_sleep(client: TradingClient, seconds: float, market_open: bool):
    """Sleep in short slices, waking early if the daily loss limit trips
    between scans so the halt lands near -1% instead of drifting past it."""
    end = time.time() + max(60, seconds)
    while time.time() < end:
        time.sleep(min(150, max(1, end - time.time())))
        if market_open and daily_loss_hit(client):
            return


def main():
    once = "--once" in sys.argv
    models = load_ensemble()
    scaler = Scaler.load()
    # Adaptive-model hygiene: warn when the ensemble is stale.
    from model import seed_path
    ages = [(time.time() - seed_path(s).stat().st_mtime) / 86400
            for s in config.ENSEMBLE_SEEDS if seed_path(s).exists()]
    if ages and min(ages) > 7:
        print(f"WARNING: newest model is {min(ages):.0f} days old -- "
              f"run 'python train.py' to adapt to the current regime.")
    client = get_client()
    account = client.get_account()
    tz_name = datetime.now().astimezone().tzname()
    print(f"Connected to Alpaca paper account -- "
          f"equity ${float(account.equity):,.2f}  "
          f"({len(models)}-model ensemble, regime gate "
          f"{'on' if config.REGIME_GATE else 'off'}, "
          f"local timezone {tz_name})")

    pending: list[str] = []
    halted_until_close = False
    flatten_fails = 0
    while True:
      try:
        clock = client.get_clock()
        now = datetime.now(timezone.utc)
        if clock.is_open:
            secs_to_close = (clock.next_close - now).total_seconds()
            et_now = now.astimezone(ZoneInfo("America/New_York"))
            session_open = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
            secs_since_open = (et_now - session_open).total_seconds()
            if halted_until_close or daily_loss_hit(client):
                if not halted_until_close:
                    flatten_fails += 1
                    if not flatten_all(client, fallback=flatten_fails >= 3):
                        time.sleep(30)
                        continue
                    flatten_fails = 0
                    halted_until_close = True
                if once:
                    return
                sleep_s = secs_to_close + 120
            elif secs_to_close <= FLATTEN_MINUTES_BEFORE_CLOSE * 60:
                flatten_fails += 1
                if not flatten_all(client, fallback=flatten_fails >= 3):
                    time.sleep(30)          # state file busy: try again, don't skip to the close
                    continue
                flatten_fails = 0
                if once:
                    return
                sleep_s = secs_to_close + 120
            else:
                allow = secs_since_open >= config.SKIP_OPEN_MINUTES * 60
                print(f"\n[{local(now)}] scanning")
                rebalance(client, models, scaler, pending, allow)
                if once:
                    return
                sleep_s = seconds_to_next_scan(now, clock.next_close)
        else:
            halted_until_close = False
            if once:
                print("Market closed -- signals only (no orders):")
                signals, gated_in = scan(models, scaler)
                for t, s in sorted(signals.items(),
                                   key=lambda kv: -kv[1]["prob"]):
                    tag = ",".join(s["setups"]) if s["setups"] else "-"
                    print(f"  {t:6s} p={s['prob']:.3f}  setups=[{tag}]")
                return
            sleep_s = (clock.next_open - now).total_seconds() + 60
            print(f"Market closed -- sleeping until open "
                  f"({local(clock.next_open)})")
        guarded_sleep(client, sleep_s,
                      clock.is_open and not halted_until_close)
      except Exception as e:
        # A transient network/API error must never kill the bot; log and
        # retry after a short pause (positions stay protected by their
        # resting stop orders on Alpaca's side meanwhile).
        print(f"  loop error (retrying in 60s): {e}")
        time.sleep(60)


if __name__ == "__main__":
    main()
