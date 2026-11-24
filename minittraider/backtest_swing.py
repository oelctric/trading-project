"""Honest walk-forward backtest for the swing strategy.

    python backtest_swing.py                      # params from swing_params.json, 9 coins, daily, ~10y
    python backtest_swing.py --tune               # tune on the older 2/3, judge on the newest 1/3
    python backtest_swing.py --exits fixed        # baseline: fixed TP/SL/time exits
    python backtest_swing.py --cash 100           # also simulate one shared account, trades in parallel
    python backtest_swing.py --levels             # compare aggressiveness levels on a $100 account

Realism:
  - signals use only closed bars; entry at the NEXT bar's open + slippage
  - stops checked intrabar; a gap through the stop fills at the open (worse)
  - signal exits (trend broken / stalled) fill at the next open + slippage
  - fees on both sides (default 0.25%/side = Coinbase taker, mid tier)
  - one position per symbol, 1% equity risked per trade, max 25% equity per position
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd

import config
from history import load
from swing import ExitManager, SwingParams, entry_features, entry_signal, indicators

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD",
           "ADA-USD", "LINK-USD", "AVAX-USD", "LTC-USD"]
FEE = config.FUTURES_FEE_PER_SIDE if config.USE_LEVERAGE else config.FEE_PER_SIDE["crypto"]
SLIP = 0.0005                    # 0.05% per side
RISK = config.SWING_RISK_PER_TRADE
MAX_POS = config.SWING_MAX_POSITION_PCT
PARAMS_FILE = os.environ.get("SWING_PARAMS",
                             os.path.join(os.path.dirname(__file__), "swing_params.json"))


def simulate(df: pd.DataFrame, p: SwingParams, exits: str = "dynamic",
             fee: float = FEE, start: int = 0, stop_at: int | None = None,
             leader: pd.DataFrame | None = None,
             breadth: pd.Series | None = None) -> list[dict]:
    """Return the trade list. Trades may only OPEN in [start, stop_at)."""
    ind = indicators(df, p, leader, breadth)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    cols = {k: ind[k].to_numpy() for k in ind.columns}
    n = len(df)
    stop_at = n - 1 if stop_at is None else min(stop_at, n - 1)
    rt_fee = 2 * fee
    trades: list[dict] = []
    i = max(start, p.slow + p.breakout)
    while i < stop_at:
        row = {k: v[i] for k, v in cols.items()}
        side, why = entry_signal(c[i], row, p, rt_fee)
        if side == "none":
            i += 1
            continue
        sign = 1 if side == "long" else -1
        e = i + 1                                   # fill at next open
        entry = o[e] * (1 + sign * SLIP)
        em = ExitManager.open(side, entry, row["atr"], p, rt_fee)
        if exits == "fixed":
            tp = entry + sign * 2.0 * em.risk
        exit_px, reason, j = None, None, e
        while j < n:
            if exits == "fixed":
                if sign > 0 and l[j] <= em.stop or sign < 0 and h[j] >= em.stop:
                    exit_px, reason = (min(o[j], em.stop) if sign > 0 else max(o[j], em.stop)), "stop"
                    break
                if sign > 0 and h[j] >= tp or sign < 0 and l[j] <= tp:
                    exit_px, reason = tp, "target"
                    break
                if j - e + 1 >= p.stall_bars:
                    exit_px, reason = c[j], "time limit"
                    break
                j += 1
                continue
            hit = em.intrabar_stop(o[j], h[j], l[j])
            if hit is not None:
                exit_px = hit
                reason = {"initial": "stop loss", "breakeven": "break-even stop"}.get(
                    em.stage, f"trailing stop ({em.stage})")
                break
            r = em.on_bar_close(h[j], l[j], c[j], {k: v[j] for k, v in cols.items()}, p)
            if r and j + 1 < n:
                exit_px, reason = o[j + 1], r
                j += 1
                break
            j += 1
        if exit_px is None:
            j = n - 1
            exit_px, reason = c[j], "end of data"
        exit_px = exit_px * (1 - sign * SLIP)
        gross_ret = sign * (exit_px - entry) / entry
        net_ret = gross_ret - fee * (1 + exit_px / entry)
        trades.append({
            "open_time": df.index[e], "close_time": df.index[j], "side": side,
            "entry": entry, "exit": exit_px, "bars": j - e + 1, "reason": reason,
            "risk_pct": em.risk / entry, "gross_ret": gross_ret, "net_ret": net_ret,
            "r": net_ret / (em.risk / entry), "mfe_r": em.mfe_r,
            "atr_pct": float(row["atr"]) / c[i],
            "features": entry_features(side, c[i], row),
        })
        # the live bot can act on the signal of the bar it exited on (its close comes
        # after the exit), so look for the next entry starting at that bar
        i = j if j > i else i + 1
    return trades


def equity_curve(trades: list[dict], start_equity: float = 10_000.0) -> tuple[float, float]:
    """Compound fixed-fractional sizing: risk RISK of equity, cap MAX_POS notional."""
    eq, peak, dd = start_equity, start_equity, 0.0
    for t in trades:
        notional_frac = min(RISK / t["risk_pct"], MAX_POS)
        eq *= 1 + notional_frac * t["net_ret"]
        peak = max(peak, eq)
        dd = max(dd, 1 - eq / peak)
    return eq / start_equity - 1, dd


def portfolio_sim(trades: list[dict], start: float = 100.0, risk: float = RISK,
                  max_pos: float = MAX_POS, max_open: int = 6, min_order: float = 1.0) -> dict:
    """All symbols on ONE account, trades held in parallel, in date order.
    Each trade risks `risk` of current equity (capped at `max_pos` of equity per
    position and by available cash - no leverage). Trades that arrive while
    `max_open` positions are already open, or that would be below the exchange
    minimum order, are skipped - exactly like the live bot."""
    cash, open_, taken, skipped = start, [], 0, 0
    peak, dd, curve = start, 0.0, []

    def settle(upto):
        nonlocal cash, peak, dd
        for o in sorted([o for o in open_ if o["close_time"] <= upto], key=lambda o: o["close_time"]):
            cash += o["notional"] * (1 + o["net_ret"])
            open_.remove(o)
            eq = cash + sum(x["notional"] for x in open_)
            peak, dd = max(peak, eq), max(dd, 1 - eq / peak)
            curve.append((o["close_time"], eq))

    for t in sorted(trades, key=lambda t: t["open_time"]):
        settle(t["open_time"])
        eq = cash + sum(o["notional"] for o in open_)
        if len(open_) >= max_open:
            skipped += 1
            continue
        notional = min(eq * risk / t["risk_pct"], eq * max_pos, cash)
        if notional < min_order:
            skipped += 1
            continue
        open_.append(dict(t, notional=notional))
        cash -= notional
        taken += 1
    settle(pd.Timestamp.max.tz_localize("UTC"))
    years = 1.0
    if trades:
        first = min(t["open_time"] for t in trades)
        last = max(t["close_time"] for t in trades)
        years = max((last - first).days / 365.25, 0.25)
    final = cash
    return {"final": final, "ret": final / start - 1, "cagr": (final / start) ** (1 / years) - 1,
            "dd": dd, "taken": taken, "skipped": skipped, "curve": curve,
            "max_concurrent": max_open}


FUNDING_PER_DAY = 0.0003   # perpetual-futures funding, fraction of notional per day held


def leverage_sim(trades: list[dict], cfg, start: float = 100.0, max_open: int = 6,
                 funding: float = FUNDING_PER_DAY) -> dict:
    """One margin account, trades in parallel, sized by leverage.size_position.
    Funding is charged on notional for every day held."""
    from leverage import size_position
    cash, open_, peak, dd, recent = start, [], start, 0.0, []
    lev_used = []

    def equity():
        return cash + sum(o["margin"] for o in open_)

    def settle(upto):
        nonlocal cash, peak, dd
        for o in sorted([o for o in open_ if o["close_time"] <= upto], key=lambda o: o["close_time"]):
            days = (o["close_time"] - o["open_time"]).days + 1
            pnl = o["notional"] * (o["net_ret"] - funding * days)
            pnl = max(pnl, -o["margin"])          # isolated margin: can't lose more than posted
            cash += o["margin"] + pnl
            recent.append(pnl / o["risk"] if o["risk"] else 0.0)
            open_.remove(o)
            e = equity()
            peak, dd = max(peak, e), max(dd, 1 - e / peak)

    for t in sorted(trades, key=lambda t: t["open_time"]):
        settle(t["open_time"])
        e = equity()
        if t.get("quality_mult", 1.0) <= 0:
            continue
        if e < cfg.min_order:
            break
        if len(open_) >= max_open:
            continue
        sz = size_position(cfg, e, peak, recent, t["risk_pct"],
                           sum(o["risk"] for o in open_), sum(o["notional"] for o in open_), cash,
                           t.get("quality_mult", 1.0))
        if sz["notional"] <= 0:
            continue
        open_.append(dict(t, **sz))
        cash -= sz["margin"]
        lev_used.append(sum(o["notional"] for o in open_) / e)
    settle(pd.Timestamp.max.tz_localize("UTC"))
    years = 1.0
    if trades:
        years = max((max(t["close_time"] for t in trades) - min(t["open_time"] for t in trades)).days / 365.25, 0.25)
    final = equity()
    return {"final": final, "cagr": (final / start) ** (1 / years) - 1 if final > 0 else -1.0,
            "dd": dd, "avg_lev": float(np.mean(lev_used)) if lev_used else 0.0,
            "max_lev": float(np.max(lev_used)) if lev_used else 0.0}


def stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "win": 0, "avg_r": 0, "sum_r": 0, "pf": 0, "ret": 0, "dd": 0, "hold": 0}
    r = np.array([t["r"] for t in trades])
    wins, losses = r[r > 0].sum(), -r[r <= 0].sum()
    ret, dd = equity_curve(trades)
    return {"n": len(r), "win": float((r > 0).mean()), "avg_r": float(r.mean()),
            "sum_r": float(r.sum()), "pf": float(wins / losses) if losses else float("inf"),
            "ret": ret, "dd": dd, "hold": float(np.mean([t["bars"] for t in trades]))}


def cut_date(frames: dict, frac: float) -> pd.Timestamp:
    """A calendar date `frac` of the way through the combined history, so every
    symbol's train/test split happens on the same day."""
    first = min(d.index[0] for d in frames.values())
    last = max(d.index[-1] for d in frames.values())
    return first + (last - first) * frac


def run_all(frames: dict, p: SwingParams, exits: str, lo: float, hi: float) -> dict:
    """Simulate every symbol, opening trades only between the calendar dates that
    are lo and hi of the way through the combined history."""
    a, z = cut_date(frames, lo), cut_date(frames, hi)
    out = {}
    for s, df in frames.items():
        start = int(df.index.searchsorted(a)) if lo > 0 else 0
        stop = int(df.index.searchsorted(z)) if hi < 1 else len(df)
        out[s] = simulate(df, p, exits, start=start, stop_at=stop,
                          leader=frames.get("BTC-USD"))
    return out


def print_table(title: str, res: dict) -> dict:
    print(f"\n{title}")
    print(f"{'symbol':<10}{'trades':>7}{'win%':>7}{'avg R':>8}{'sum R':>8}{'PF':>6}"
          f"{'return':>9}{'maxDD':>8}{'avg bars':>10}")
    allt = []
    for s, tr in res.items():
        st = stats(tr)
        allt += tr
        print(f"{s:<10}{st['n']:>7}{st['win']:>7.0%}{st['avg_r']:>+8.2f}{st['sum_r']:>+8.1f}"
              f"{st['pf']:>6.2f}{st['ret']:>+9.1%}{st['dd']:>8.1%}{st['hold']:>8.0f}b")
    st = stats(allt)
    print(f"{'ALL':<10}{st['n']:>7}{st['win']:>7.0%}{st['avg_r']:>+8.2f}{st['sum_r']:>+8.1f}"
          f"{st['pf']:>6.2f}{'':>9}{'':>8}{st['hold']:>8.0f}b")
    prof = sum(1 for tr in res.values() if stats(tr)["sum_r"] > 0)
    print(f"profitable symbols: {prof}/{len(res)}   "
          f"avg per-symbol return: {np.mean([stats(tr)['ret'] for tr in res.values()]):+.1%}")
    return st


GRID = {   # daily-scale values; rescale if tuning on intraday bars
    "breakout": [10, 20, 30, 50],
    "slow": [50, 100, 200],
    "init_stop_atr": [2.0, 2.5, 3.0],
    "trail_atr": [2.5, 3.0, 4.0],
    "stall_bars": [10, 15, 30],
    "er_min": [0.1, 0.2, 0.3],
}


def tune(frames: dict, base: SwingParams, split: float) -> SwingParams:
    """Score = median per-symbol sum-R on the TRAIN slice (robust to one lucky coin),
    with at least 5 trades/symbol on average. Never looks at the test slice."""
    keys = list(GRID)
    best, best_score = base, -1e9
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    t0 = time.time()
    for n_done, vals in enumerate(combos, 1):
        p = replace(base, **dict(zip(keys, vals)))
        res = run_all(frames, p, "dynamic", 0.0, split)
        per = [stats(tr)["sum_r"] for tr in res.values()]
        ntr = np.mean([len(tr) for tr in res.values()])
        score = float(np.median(per)) if ntr >= 5 else -1e9
        if score > best_score:
            best, best_score = p, score
        if n_done % 25 == 0:
            print(f"  tuned {n_done}/{len(combos)} ({time.time() - t0:.0f}s), "
                  f"best median sum-R so far {best_score:+.1f}", flush=True)
    print(f"  best train params: { {k: getattr(best, k) for k in keys} }  score {best_score:+.1f}")
    return best


LEVELS = [("current (old)", 0.01, 0.25, 2), ("moderate (live)", 0.02, 0.50, 6),
          ("aggressive", 0.03, 0.50, 6), ("very aggressive", 0.05, 1.00, 9)]


def portfolio_report(frames: dict, p: SwingParams, args) -> None:
    """One shared account, all coins, trades held in parallel - like the live bot."""
    fee = args.fee if args.fee is not None else FEE
    trades = [t for s, df in frames.items()
              for t in simulate(df, p, args.exits, fee=fee, leader=frames.get("BTC-USD"))]
    cut = cut_date(frames, args.split)
    last = frames[next(iter(frames))].index[-1] - pd.Timedelta(days=365)
    periods = [("train", [t for t in trades if t["open_time"] < cut]),
               ("test (unseen)", [t for t in trades if t["open_time"] >= cut]),
               ("last 12 months", [t for t in trades if t["open_time"] >= last])]
    levels = LEVELS if args.levels else [("live config", RISK, MAX_POS, config.MAX_OPEN_POSITIONS)]
    print(f"\nONE ${args.cash:,.0f} ACCOUNT, all coins in parallel, fee {fee:.2%}/side")
    print(f"{'level':<17}{'risk':>5}{'maxpos':>7}{'open':>5}" +
          "".join(f" | {name:>14} {'CAGR':>5} {'maxDD':>5}" for name, _ in periods))
    for name, risk, mp, mo in levels:
        line = f"{name:<17}{risk:>5.0%}{mp:>7.0%}{mo:>5}"
        for _, tr in periods:
            r = portfolio_sim(tr, args.cash, risk, mp, mo, config.MIN_ORDER_USD)
            line += f" | {'$' + format(r['final'], ',.2f'):>14} {r['cagr']:>5.0%} {r['dd']:>5.0%}"
        print(line)


def leverage_report(frames: dict, p: SwingParams, args) -> None:
    """Live leverage config vs alternatives, futures fees + funding, one shared account."""
    from leverage import LeverageConfig
    fee = args.fee if args.fee is not None else config.FUTURES_FEE_PER_SIDE
    trades = [t for s, df in frames.items()
              for t in simulate(df, p, args.exits, fee=fee, leader=frames.get("BTC-USD"))]
    cut = cut_date(frames, args.split)
    last = frames[next(iter(frames))].index[-1] - pd.Timedelta(days=365)
    periods = [("train", [t for t in trades if t["open_time"] < cut]),
               ("test (unseen)", [t for t in trades if t["open_time"] >= cut]),
               ("last 12 months", [t for t in trades if t["open_time"] >= last])]
    off = dict(heat_cap=9.0, dd_brake_start=9.0, dd_brake_full=10.0, cold_factor=1.0)
    setups = [("2% risk, no leverage", LeverageConfig(base_risk=0.02, max_leverage=1.0, **off)),
              ("3% risk, 2x, no brake", LeverageConfig(base_risk=0.03, max_leverage=2.0, **off)),
              ("LIVE: config.LEVERAGE", config.LEVERAGE),
              ("5% risk, 3x, cold brake", replace(config.LEVERAGE, base_risk=0.05, max_leverage=3.0))]
    print(f"\nLEVERAGE: one ${args.cash:,.0f} account, all coins in parallel, fee {fee:.2%}/side "
          f"+ funding {FUNDING_PER_DAY:.2%}/day")
    print(f"{'setup':<25}" + "".join(f" | {n:>14} {'CAGR':>5} {'maxDD':>5} {'avgLev':>6}" for n, _ in periods))
    for name, cfg in setups:
        line = f"{name:<25}"
        for _, tr in periods:
            r = leverage_sim(tr, cfg, args.cash)
            line += (f" | {'$' + format(r['final'], ',.2f'):>14} {r['cagr']:>5.0%} {r['dd']:>5.0%}"
                     f" {r['avg_lev']:>5.2f}x")
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default=None, help="bar size (default: from swing_params.json)")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--exits", choices=["dynamic", "fixed"], default="dynamic")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--split", type=float, default=2 / 3)
    ap.add_argument("--short", action="store_true", help="allow shorts")
    ap.add_argument("--long-only", action="store_true", help="disable shorts")
    ap.add_argument("--save-trades", action="store_true")
    ap.add_argument("--cash", type=float, default=100.0, help="start cash for the portfolio sim")
    ap.add_argument("--fee", type=float, default=None, help="fee per side (default config)")
    ap.add_argument("--levels", action="store_true", help="compare aggressiveness levels")
    ap.add_argument("--leverage", action="store_true",
                    help="compare leverage sizing (config.LEVERAGE) vs no leverage, futures fees")
    args = ap.parse_args()

    saved = {}
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            saved = json.load(f)
    args.tf = args.tf or saved.get("tf", "1d")
    args.days = args.days or (3650 if args.tf == "1d" else 1095)
    frames = {s: load(s, args.tf, args.days) for s in args.symbols}
    frames = {s: d for s, d in frames.items() if len(d) > 500}
    cut = cut_date(frames, args.split)
    print(f"{len(frames)} symbols, {args.tf} bars, {cut_date(frames, 0):%Y-%m-%d} -> "
          f"{cut_date(frames, 1):%Y-%m-%d}; train < {cut:%Y-%m-%d} <= test")
    print(f"fees {FEE:.2%}/side + slippage {SLIP:.2%}/side; risk {RISK:.0%}/trade")

    base = SwingParams()
    if saved and not args.tune:
        base = replace(base, **saved["params"])
        print(f"loaded params from {os.path.basename(PARAMS_FILE)}")
    if args.short:
        base = replace(base, allow_short=True)
    if args.long_only:
        base = replace(base, allow_short=False)
    p = tune(frames, base, args.split) if args.tune else base

    train = run_all(frames, p, args.exits, 0.0, args.split)
    test = run_all(frames, p, args.exits, args.split, 1.0)
    print_table(f"TRAIN (in-sample, {args.exits} exits)", train)
    st = print_table(f"TEST (out-of-sample, never seen while tuning, {args.exits} exits)", test)

    bh = {s: d["close"].iloc[-1] / d["close"].iloc[min(int(d.index.searchsorted(cut)), len(d) - 1)] - 1
          for s, d in frames.items()}
    print(f"buy & hold over the test period (for context): avg {np.mean(list(bh.values())):+.1%}")

    if args.leverage:
        leverage_report(frames, p, args)
    else:
        portfolio_report(frames, p, args)

    if args.tune:
        with open(PARAMS_FILE, "w") as f:
            json.dump({"params": p.to_dict(), "tf": args.tf,  # noqa: E501
                       "test": {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in st.items()}}, f, indent=1)
        print(f"saved tuned params to {os.path.basename(PARAMS_FILE)}")
    if args.save_trades:
        rows = [dict(t, symbol=s, sample="train") for s, tr in train.items() for t in tr]
        rows += [dict(t, symbol=s, sample="test") for s, tr in test.items() for t in tr]
        pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(__file__), "swing_trades.csv"),
                                  index=False)


if __name__ == "__main__":
    main()
