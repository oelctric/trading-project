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
"""
from __future__ import annotations

import sys

import numpy as np

import config
from data import _client, download_bars

N_LEVELS = 400          # synthetic depth per side (deep enough to not exhaust)
STEPS = 500             # market orders per simulated path
RUNS = 1000


def build_book(mid: float, tick: float, bid_top: float, ask_top: float,
               rng: np.random.Generator):
    """Synthetic depth template: real top-of-book sizes at the touch,
    power-law decaying (noisy) size deeper in, on each side's real price."""
    i = np.arange(N_LEVELS)
    decay = (1 + i) ** -0.3
    noise = rng.lognormal(0, 0.5, size=(2, N_LEVELS))
    bid_qty = bid_top * decay * noise[0]
    ask_qty = ask_top * decay * noise[1]
    bid_px = mid - tick * (i + 1)
    ask_px = mid + tick * (i + 1)
    return (bid_px, bid_qty), (ask_px, ask_qty)


def simulate_path(book_template, mean_order: float,
                  rng: np.random.Generator) -> np.ndarray:
    """One headless run: coin-flip market orders eat the book; log price."""
    (bid_px, bid_qty0), (ask_px, ask_qty0) = book_template
    bid_qty = bid_qty0.copy()
    ask_qty = ask_qty0.copy()
    bid_i = ask_i = 0  # index of best level on each side
    price = (bid_px[0] + ask_px[0]) / 2
    path = np.empty(STEPS)
    for t in range(STEPS):
        size = rng.exponential(mean_order)
        if rng.random() < 0.5:  # market buy eats asks
            while size > 0 and ask_i < N_LEVELS:
                take = min(size, ask_qty[ask_i])
                ask_qty[ask_i] -= take
                size -= take
                price = ask_px[ask_i]
                if ask_qty[ask_i] <= 0:
                    ask_i += 1
        else:  # market sell eats bids
            while size > 0 and bid_i < N_LEVELS:
                take = min(size, bid_qty[bid_i])
                bid_qty[bid_i] -= take
                size -= take
                price = bid_px[bid_i]
                if bid_qty[bid_i] <= 0:
                    bid_i += 1
        path[t] = price
    return path


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
    rng = np.random.default_rng(0)

    bid, ask, bid_sz, ask_sz = latest_quote(symbol)
    mid = (bid + ask) / 2
    imbalance = bid_sz / (bid_sz + ask_sz)
    bars = download_bars(symbol, days=30)
    atr = float((bars["high"] - bars["low"]).tail(50).mean())
    tick = max(0.01, round(atr / 50, 2))

    print(f"{symbol}: bid {bid}x{bid_sz:.0f} / ask {ask}x{ask_sz:.0f}  "
          f"imbalance={imbalance:.2f} (0.5 = balanced book)")
    template = build_book(mid, tick, max(bid_sz, 1), max(ask_sz, 1), rng)
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
    print(f"  P(hit +{config.TP_ATR_MULT} ATR before -{config.SL_ATR_MULT} "
          f"ATR) = {p:.1%}   (long-bracket win probability)")
    print(f"  90% price cone at horizon: {lo:.2f} .. {hi:.2f}  "
          f"(median {med:.2f}, start {mid:.2f})")


if __name__ == "__main__":
    main()
