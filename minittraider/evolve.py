"""Evolutionary training: breed bounce-strategy variants, only the best survives.

    python evolve.py

Method (and the honesty rules that make it meaningful):
  - 45 days of 1-minute crypto history + 28 days of stocks (Yahoo's limit)
  - each genome = one setting of the bounce strategy's knobs
  - fitness = net P&L AFTER taker fees on the TRAIN period (first ~2/3 of data)
  - evolution: tournament selection, uniform crossover, mutation, elitism
  - the final survivors are then judged on the TEST period they have never
    seen. Train results always look good - that's what optimization does.
    Only the test column tells you whether anything real was learned.

Zone detection (the slow part) is precomputed once per symbol and cached in
backtest_data/, so re-runs and future experiments are fast.
"""

from __future__ import annotations

import os
import random
import time

import numpy as np
import pandas as pd

from backtest import CACHE_DIR, FEES, get_history
from bounce import swing_zones
from signals import atr as atr_fn
from signals import volatility_regime

DAYS_CRYPTO = 45
DAYS_STOCK = 28
WINDOW = 260                 # bars of history each decision may look at
TRAIN_FRACTION = 2 / 3
EQUITY = 10_000.0
FEE_CLEAR_ATR = 1.2          # target must be at least this far (fixed, clears fees)

POP = 40
GENERATIONS = 12
ELITE = 4
TOURNAMENT = 3
MUT_RATE = 0.35
MIN_TRAIN_TRADES = 40        # a genome that barely trades can't be trusted

SYMBOLS = [("BTC-USD", True), ("ETH-USD", True), ("SOL-USD", True),
           ("SPY", False), ("TSLA", False), ("NVDA", False)]

rng = random.Random(7)

# ---------------------------------------------------------------- genome

GENES = {
    "touches":    [2, 3, 4],                 # rejections needed to trust a zone
    "zone_atr":   (0.3, 1.2),                # how close to a zone counts as "at" it
    "min_rr":     (1.2, 3.0),                # required reward/risk
    "stop_atr":   (0.4, 1.0),                # stop distance beyond the zone
    "target_atr": (2.0, 4.5),                # fallback target when no opposite zone
    "hold":       [10, 20, 30, 45],          # max minutes in a trade
    "reject":     [0, 1],                    # require a rejection candle at the zone
    "dead_ok":    [0, 1],                    # allowed to trade in dead-volatility regime
}


def random_genome() -> dict:
    g = {}
    for k, spec in GENES.items():
        g[k] = rng.choice(spec) if isinstance(spec, list) else round(rng.uniform(*spec), 2)
    return g


def mutate(g: dict) -> dict:
    g = dict(g)
    for k, spec in GENES.items():
        if rng.random() < MUT_RATE:
            if isinstance(spec, list):
                g[k] = rng.choice(spec)
            else:
                lo, hi = spec
                g[k] = round(min(hi, max(lo, g[k] + rng.gauss(0, (hi - lo) * 0.15))), 2)
    return g


def crossover(a: dict, b: dict) -> dict:
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in GENES}


# ---------------------------------------------------------------- precompute

def precompute(symbol: str, df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Per bar: regime, ATR, and the nearest zone above/below for each touch level."""
    cache = os.path.join(CACHE_DIR, f"{symbol}_{days}d_evo.csv")
    src = os.path.join(CACHE_DIR, f"{symbol}_{days}d.csv")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(src):
        return pd.read_csv(cache)

    t0 = time.time()
    closes = df["close"].to_numpy()
    rows = []
    for i in range(WINDOW, len(df) - 1):
        window = df.iloc[i - WINDOW:i]
        regime, _, _ = volatility_regime(window)
        price = closes[i - 1]
        zones = swing_zones(window)  # uses BOUNCE_MIN_TOUCHES=2 as the floor
        row = {
            "i": i,
            "gap": (df.index[i] - df.index[i - 1]).total_seconds() > 180,
            "atr": float(atr_fn(window).iloc[-1]),
            "regime": regime,
        }
        for t in GENES["touches"]:
            sups = [lvl for lvl, n in zones if lvl < price and n >= t]
            ress = [lvl for lvl, n in zones if lvl > price and n >= t]
            row[f"sup{t}"] = max(sups) if sups else np.nan
            row[f"res{t}"] = min(ress) if ress else np.nan
        rows.append(row)
    pre = pd.DataFrame(rows)
    pre.to_csv(cache, index=False)
    print(f"  {symbol}: scored {len(pre):,} bars in {time.time() - t0:.0f}s", flush=True)
    return pre


class SymbolData:
    """Numpy arrays for one symbol, ready for fast genome evaluation."""

    def __init__(self, symbol: str, is_crypto: bool, df: pd.DataFrame, pre: pd.DataFrame):
        self.symbol = symbol
        self.fee = FEES["crypto" if is_crypto else "stock"]
        self.opens = df["open"].to_numpy()
        self.highs = df["high"].to_numpy()
        self.lows = df["low"].to_numpy()
        self.closes = df["close"].to_numpy()
        self.times = df.index.asi8
        self.n = len(df)
        self.idx = pre["i"].to_numpy(dtype=int)
        self.gap = pre["gap"].to_numpy(dtype=bool)
        self.atr = pre["atr"].to_numpy()
        self.regime = pre["regime"].to_numpy()
        self.zones = {t: (pre[f"sup{t}"].to_numpy(), pre[f"res{t}"].to_numpy())
                      for t in GENES["touches"]}
        self.split = int(len(pre) * TRAIN_FRACTION)   # row index separating train/test


# ---------------------------------------------------------------- simulation

def run(sd: SymbolData, g: dict, part: str) -> tuple[float, int]:
    """Net P&L after fees and trade count for one genome on train or test rows."""
    lo, hi = (0, sd.split) if part == "train" else (sd.split, len(sd.idx))
    sups, ress = sd.zones[g["touches"]]
    hold = int(g["hold"])
    net, trades = 0.0, 0

    k = lo
    while k < hi:
        if sd.gap[k]:
            k += 1
            continue
        regime = sd.regime[k]
        if regime in ("wild", "unknown") or (regime == "dead" and not g["dead_ok"]):
            k += 1
            continue
        a = sd.atr[k]
        if not a > 0:
            k += 1
            continue
        i = sd.idx[k]
        price = sd.closes[i - 1]
        sup, res = sups[k], ress[k]
        at_sup = not np.isnan(sup) and price - sup <= g["zone_atr"] * a
        at_res = not np.isnan(res) and res - price <= g["zone_atr"] * a
        if at_sup == at_res:   # neither (mid-cloud) or both (squeezed)
            k += 1
            continue

        if at_sup:
            sign, zone = 1, sup
            # probed the zone AND closed back up (OR made it always true here,
            # so the "reject" gene had no effect - same fix as bounce.py)
            rejected = (sd.closes[i - 1] > sd.opens[i - 1]
                        and min(sd.lows[i - 1], sd.lows[i - 2]) <= zone + 0.2 * a)
            stop = zone - g["stop_atr"] * a
            target = (res - 0.2 * a) if not np.isnan(res) else price + g["target_atr"] * a
        else:
            sign, zone = -1, res
            rejected = (sd.closes[i - 1] < sd.opens[i - 1]
                        and max(sd.highs[i - 1], sd.highs[i - 2]) >= zone - 0.2 * a)
            stop = zone + g["stop_atr"] * a
            target = (sup + 0.2 * a) if not np.isnan(sup) else price - g["target_atr"] * a

        room, risk = sign * (target - price), sign * (price - stop)
        if (g["reject"] and not rejected) or risk <= 0 \
                or room < g["min_rr"] * risk or room < FEE_CLEAR_ATR * a:
            k += 1
            continue

        entry = sd.opens[i]
        if sign * (target - entry) <= 0 or sign * (entry - stop) <= 0:
            k += 1
            continue
        qty = min(EQUITY * 0.01 / (sign * (entry - stop)), EQUITY * 0.25 / entry)

        exit_price, j = None, i
        for j in range(i, min(i + hold, sd.n)):
            if j > i and (sd.times[j] - sd.times[j - 1]) > 180e9:
                exit_price = sd.closes[j - 1]
                break
            if (sd.lows[j] <= stop if sign == 1 else sd.highs[j] >= stop):
                exit_price = stop
                break
            if (sd.highs[j] >= target if sign == 1 else sd.lows[j] <= target):
                exit_price = target
                break
        if exit_price is None:
            j = min(i + hold, sd.n) - 1
            exit_price = sd.closes[j]

        net += sign * (exit_price - entry) * qty - self_fee(sd, qty, entry, exit_price)
        trades += 1
        k = max(k + 1, j + 1 - (sd.idx[0]))   # jump past the exit bar
    return net, trades


def self_fee(sd: SymbolData, qty: float, entry: float, exit_price: float) -> float:
    return sd.fee * qty * (entry + exit_price)


def fitness(data: list[SymbolData], g: dict, part: str) -> tuple[float, int]:
    total, trades = 0.0, 0
    for sd in data:
        n, t = run(sd, g, part)
        total += n
        trades += t
    return total, trades


# ---------------------------------------------------------------- evolution

def main() -> None:
    print("downloading history (cached after first run)...")
    data = []
    for symbol, is_crypto in SYMBOLS:
        days = DAYS_CRYPTO if is_crypto else DAYS_STOCK
        df = get_history(symbol, is_crypto, days)
        if df.empty or len(df) < WINDOW + 100:
            print(f"  {symbol}: not enough data, skipped")
            continue
        pre = precompute(symbol, df, days)
        data.append(SymbolData(symbol, is_crypto, df, pre))
    train_days = DAYS_CRYPTO * TRAIN_FRACTION
    print(f"\nevolving: pop {POP}, {GENERATIONS} generations, "
          f"fitness = net P&L after fees on first ~{train_days:.0f} days")

    pop = [random_genome() for _ in range(POP)]
    for gen in range(GENERATIONS):
        t0 = time.time()
        scored = []
        for g in pop:
            net, trades = fitness(data, g, "train")
            if trades < MIN_TRAIN_TRADES:
                net = -1e9  # refusing to trade is not a survival strategy
            scored.append((net, trades, g))
        scored.sort(key=lambda x: -x[0])
        best = scored[0]
        print(f"gen {gen + 1:>2}: best train P&L {best[0]:+,.2f} "
              f"({best[1]} trades) | median {scored[POP // 2][0]:+,.2f} "
              f"| {time.time() - t0:.0f}s", flush=True)

        elite = [g for _, _, g in scored[:ELITE]]
        children = list(elite)
        while len(children) < POP:
            def pick():
                return min(rng.sample(scored, TOURNAMENT), key=lambda x: -x[0])[2]
            children.append(mutate(crossover(pick(), pick())))
        pop = children

    # ---- judgment day: survivors meet data they have never seen
    print(f"\n{'=' * 92}")
    print("SURVIVORS ON UNSEEN DATA (test = most recent ~1/3 of history)")
    print(f"{'=' * 92}")
    head = (f"{'rank':<5}{'train P&L':>11}{'trades':>8}{'test P&L':>11}{'trades':>8}   genome")
    print(head)
    print("-" * 92)
    seen = set()
    survivors = []
    for net, trades, g in scored:
        key = tuple(sorted(g.items()))
        if key in seen or net <= -1e9:
            continue
        seen.add(key)
        survivors.append((net, trades, g))
        if len(survivors) == 5:
            break
    for rank, (net, trades, g) in enumerate(survivors, 1):
        tnet, ttrades = fitness(data, g, "test")
        gene_str = " ".join(f"{k}={v}" for k, v in g.items())
        print(f"{rank:<5}{net:>+11.2f}{trades:>8}{tnet:>+11.2f}{ttrades:>8}   {gene_str}")
    print("-" * 92)
    print("read it like this: if test P&L is positive and roughly proportional to train,")
    print("something real survived. if train is great and test is flat/negative, the")
    print("evolution just memorized the past - that 'best' genome is noise.")


if __name__ == "__main__":
    main()
