"""Limit-order-book Monte Carlo simulator (order book dynamics + random walk).

Implements the classic demonstration that resting-liquidity asymmetry biases
price even when incoming order flow is a fair coin flip:

  * an order book of (price, qty) levels — bids descending, asks ascending
  * a matching engine: market buys eat the ask side, market sells eat the
    bid side; the last fill price becomes the new price
  * a batch runner: reset the book to its snapshot template, run N random
    steps, log the price path; repeat R times headlessly

Heavy resting size acts as an absorbing boundary in the stochastic process,
so the price-path distribution tilts toward the thin side of the book.

Real Level-2 depth isn't available on free market-data plans, so the book
is seeded from the live top-of-book quote (real bid/ask prices and sizes)
with synthetic depth decaying away from the touch, scaled per side by the
observed top-of-book imbalance.

Usage:
  python lobsim.py SPY              # seed from live quote, 1000 runs
  python lobsim.py SPY 5000         # more runs
  python lobsim.py SPY 1000 5000    # more orders per path (reach the barriers)
"""
from __future__ import annotations

import sys

import numpy as np

import config
from data import _client, download_bars

N_LEVELS = 400          # synthetic depth per side (deep enough to not exhaust)
STEPS = 500             # market orders per simulated path
RUNS = 1000
MAX_SANE_SPREAD = 0.005   # >0.5% bid/ask spread = stale/off-hours quote


def build_book(bid: float, ask: float, tick: float, bid_top: float, ask_top: float,
               rng: np.random.Generator):
    """Synthetic depth template: real top-of-book sizes at the touch,
    power-law decaying (noisy) size deeper in, on each side's real price."""
    i = np.arange(N_LEVELS)
    decay = (1 + i) ** -0.3
    noise = rng.lognormal(0, 0.5, size=(2, N_LEVELS))
    # the touch uses the REAL quoted sizes exactly; only deeper (synthetic)
    # levels are noisy. Noise at level 0 gave even a balanced 40x40 book a
    # random, persistent tilt (SPY showed a 98% "win probability").
    noise[:, 0] = 1.0
    bid_qty = bid_top * decay * noise[0]
    ask_qty = ask_top * decay * noise[1]
    # levels start AT the real touch prices (the spread used to be thrown away
    # by building both sides outward from the mid)
    bid_px = bid - tick * i
    ask_px = ask + tick * i
    return (bid_px, bid_qty), (ask_px, ask_qty)


def simulate_path(book_template, mean_order: float,
                  rng: np.random.Generator) -> np.ndarray:
    """One headless run: coin-flip market orders hit the book; log price.

    Liquidity REPLENISHES: after each trade the book is re-quoted around the
    new price with the same shape (market makers re-quote). Each order walks
    as many levels as its size consumes, so a thin side gives bigger steps -
    price is a random walk tilted by the book's imbalance.

    (The old version never refilled consumed levels: the bid and ask edges
    each crept outward in one direction and "price" just alternated between
    them, so whichever barrier was nearer won almost every run.)
    """
    (bid_px, bid_qty0), (ask_px, ask_qty0) = book_template
    mid0 = (bid_px[0] + ask_px[0]) / 2
    up_off, dn_off = ask_px - mid0, mid0 - bid_px      # level distances from mid
    up_cum, dn_cum = np.cumsum(ask_qty0), np.cumsum(bid_qty0)
    sizes = rng.exponential(mean_order, STEPS)
    buys = rng.random(STEPS) < 0.5
    # Price impact moves CONTINUOUSLY through the book: eating a fraction of a
    # level moves price that fraction of the way to the next level. (Snapping
    # to whole levels meant small orders never left level 0, so a 2:1 book
    # walked exactly like a balanced one.) Thin side -> bigger steps -> tilt.
    up_x = np.concatenate([[0.0], up_cum[:-1]])
    dn_x = np.concatenate([[0.0], dn_cum[:-1]])
    up_move = np.interp(sizes, up_x, up_off)
    dn_move = np.interp(sizes, dn_x, dn_off)
    steps = np.where(buys, up_move, -dn_move)
    return mid0 + np.cumsum(steps)


def batch(book_template, mean_order: float, runs: int,
          rng: np.random.Generator) -> np.ndarray:
    return np.stack([simulate_path(book_template, mean_order, rng)
                     for _ in range(runs)])


def p_barrier_first(paths: np.ndarray, start: float,
                    up: float, dn: float) -> float:
    """P(price touches `up` before `dn`) across paths — the Monte Carlo
    estimate of our bracket's win probability given this book shape."""
    wins = losses = 0
    for path in paths:
        hit_up = np.argmax(path >= up) if (path >= up).any() else STEPS + 1
        hit_dn = np.argmax(path <= dn) if (path <= dn).any() else STEPS + 1
        if hit_up < hit_dn:
            wins += 1
        elif hit_dn < hit_up:
            losses += 1
    decided = wins + losses
    return wins / decided if decided else float("nan")


def latest_quote(symbol: str):
    from alpaca.data.requests import StockLatestQuoteRequest
    q = _client().get_stock_latest_quote(
        StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
    return (float(q.bid_price), float(q.ask_price),
            float(q.bid_size), float(q.ask_size))


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else RUNS
    global STEPS
    STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else STEPS
    rng = np.random.default_rng(0)

    bid, ask, bid_sz, ask_sz = latest_quote(symbol)
    imbalance = bid_sz / (bid_sz + ask_sz)
    bars = download_bars(symbol, days=30)
    atr = float((bars["high"] - bars["low"]).tail(50).mean())
    tick = max(0.01, round(atr / 50, 2))
    # Off-hours / stale quotes can be absurdly wide (e.g. a 6% "spread" on SPY
    # over a weekend). Building the book from those prices produces garbage, so
    # fall back to a book centred on the last trade.
    mid = (bid + ask) / 2
    if bid <= 0 or ask <= 0 or (ask - bid) / mid > MAX_SANE_SPREAD:
        last = float(bars["close"].iloc[-1])
        print(f"WARNING: quote {bid}/{ask} looks stale or off-hours "
              f"(spread {(ask - bid) / max(mid, 1e-9):.1%}) - centring the book on the "
              f"last trade {last:.2f} instead")
        mid, bid, ask = last, last - tick / 2, last + tick / 2
        # the dead quote's SIZES are meaningless too - use a balanced book
        neutral = max(1.0, (bid_sz + ask_sz) / 2)
        bid_sz = ask_sz = neutral
        imbalance = 0.5
        print("         (sizes replaced by a balanced book - imbalance is not meaningful off-hours)")

    print(f"{symbol}: bid {bid}x{bid_sz:.0f} / ask {ask}x{ask_sz:.0f}  "
          f"imbalance={imbalance:.2f} (0.5 = balanced book)")
    template = build_book(bid, ask, tick, max(bid_sz, 1), max(ask_sz, 1), rng)
    # Mean order ~10% of a top level: real books also replenish, so orders
    # must be small relative to depth or every run just exhausts the book.
    mean_order = max(bid_sz, ask_sz) * 0.1
    paths = batch(template, mean_order=mean_order, runs=runs, rng=rng)

    up = mid * (1 + config.TP_ATR_MULT * atr / mid)
    dn = mid * (1 - config.SL_ATR_MULT * atr / mid)
    p = p_barrier_first(paths, mid, up, dn)
    finals = paths[:, -1]
    lo, med, hi = np.percentile(finals, [5, 50, 95])
    print(f"{runs} runs x {STEPS} random orders from this book:")
    if np.isnan(p):
        print(f"  no path reached +{config.TP_ATR_MULT} or -{config.SL_ATR_MULT} ATR within "
              f"{STEPS} orders - pass more orders per path (3rd argument)")
    else:
        print(f"  P(hit +{config.TP_ATR_MULT} ATR before -{config.SL_ATR_MULT} "
              f"ATR) = {p:.1%}   (long-bracket win probability)")
    print(f"  90% price cone at horizon: {lo:.2f} .. {hi:.2f}  "
          f"(median {med:.2f}, start {mid:.2f})")


if __name__ == "__main__":
    main()
