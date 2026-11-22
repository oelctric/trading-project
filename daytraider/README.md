# Day Traider — Multi-Strategy Neural Network Day-Trading Bot

Detects sixteen intraday setups — momentum, breakout, reversal, pullback,
opening-range breakout, gap-and-go, gap-fade (long and short mirrors),
plus two **machine-discovered strategies** — across 40 liquid US names,
applies the pro-desk VWAP rule (longs above session VWAP, shorts below),
gates strategies by recent regime performance, ranks candidates with a
fear-aware LSTM ensemble, and trades through an Alpaca **paper trading**
account with ATR-scaled bracket orders, equal dollar risk per trade,
breakeven stop moves at +1R, sector caps, and a daily loss limit.

**What the walk-forward validates** (57 out-of-sample months, gate
decisions always using only past data): the current configuration scores
**+4.3 bps/trade gated, 56% positive months** vs +0.5 bps taking every
setup indiscriminately. Best standalone strategies over 5 years:
disc_orb_wide (+10.7 bps), gap_fade (+9.7), disc_strongday_dip (+7.2),
gap_fade_dn (+6.3, the only positive short). A hard SPY-direction market
filter was tested and REJECTED (it blocks gap-fade shorts exactly when
they work). Caveat: the two disc_* strategies were mined from the earlier
80% of this history, so their in-span numbers are flattered — their
final-year test slice (+17.1/+16.2 bps) and live paper results are the
clean evidence. Thin edges, honestly measured; not a money printer.

**Strategy discovery** (`discover.py`) — generates thousands of random
2-condition rules and reports only those that survive train → validation
→ test. Survivors land in artifacts/discovered.json; promote by hand.

## Architecture

**Setups drive entries; the neural network ranks them.** Backtesting showed
the rule-based setups carry the edge, while using the NN as a hard veto
destroyed it — so the NN's job is prioritization when more setups fire than
there are position slots.

1. **Data** (`data.py`) — 5 years of hourly bars for 40 tickers from
   **Alpaca's market data API** (regular session only, split-adjusted;
   15-minute bars were tested and rejected — per-trade edge falls below
   trading costs at that resolution) with 22 features per bar: returns,
   range/body, volume
   z-score, RSI, MACD, SMA distances, Bollinger position, ATR,
   time-of-day, the four setup flags, and a **fear-awareness block** that
   models how humans react: VIX level/change/trend (market-wide fear,
   daily, lagged one day), drawdown from the recent high, consecutive
   red-bar streak, and a loss-aversion ratio (downside vs upside
   volatility).
   **Labels are triple-barrier**: each bar is labeled by simulating the
   exact ATR bracket the live bot trades (take-profit +1.5 ATR vs
   stop-loss −1.0 ATR within 8 bars), so the network learns to predict
   trade outcomes, not bar direction.
2. **Setups** (`strategies.py`) — five long patterns (momentum surge,
   20-bar breakout, oversold reversal, uptrend pullback, opening-range
   breakout) plus five short mirrors. The **VWAP filter** applies the
   pro-desk rule: long setups only fire above the session VWAP, shorts
   only below — walk-forward validated to improve nearly every strategy.
   All ten are structurally active; the regime gate decides which may
   trade on any given day.
3. **Model** (`model.py`) — an **ensemble of five** 2-layer LSTMs (64
   hidden, different seeds) reading the last 32 bars; their averaged
   up-probability ranks candidates. Ensembling washes out single-run
   training noise.
4. **Training** (`train.py`) — chronological 70/15/15 split (split before
   windowing: no lookahead), scaler fit on train only, early stopping per
   seed.
5. **Backtest** (`backtest.py`) — per-strategy hit rate / avg return / total
   on the held-out test period, with costs.
6. **Live bot** (`trade.py`) — hourly scan; regime-gates strategies, ranks
   fired setups with the ensemble (longs by prob, shorts by 1 − prob), and
   enters up to 4 concurrent positions with **marketable-limit bracket
   orders** (+1.5 ATR take-profit / −1.0 ATR stop) **sized so a stop-out
   loses ~0.25% of equity** (capped at 25% of equity per position).
   Risk rules: max 2 positions per sector, no entries in the first 30
   minutes of the session, trading halts for the day at −1% equity, stale
   unfilled entries are cancelled each scan, and everything is flattened
   15 minutes before the close.
7. **Walk-forward harness** (`walkforward.py`) — rolls month-by-month
   through the full history to evaluate the setups and the regime gate
   out-of-sample. Use it to judge any future strategy change instead of a
   single backtest window.

## What 5 years of testing actually showed

All numbers are simulated **bracket outcomes** (TP/SL/horizon exit), the
same rules the live bot trades, with costs charged.

- **No strategy holds its edge across regimes.** Every one of the eight
  flips sign between adjacent multi-month periods. Only long momentum is
  positive over the full 5 years standalone (+1.9 bps/trade).
- **Taking every fired setup loses money** (−2.8 bps/trade over 57
  months). The regime gate cuts that to −1.4 bps and lifts positive
  months from 30% to 42% — the only mechanism validated across many
  windows rather than one.
- **The NN's ranking value is inconsistent** across windows and
  strategies: treat it as a tiebreaker for filling limited slots, not as
  the edge.
- Hit rates sit below 50% by design: winners (+1.5 ATR) are 1.5x losers
  (−1.0 ATR), so profits come from asymmetry, not win rate.

The honest bottom line: this is a disciplined experimentation platform
with real risk management, not a proven money-maker. Judge it by live
paper results, and evaluate any change with `python walkforward.py`.

**Retrain weekly** (`python train.py`) so the models and calibration
track the current regime.

⚠️ One good backtest period is *not* proof of future profitability. Hit
rates near 50% with positive average return mean the edge comes from
asymmetry, not from "winning most trades" — expect losing days and weeks.

## Setup

```
pip install -r requirements.txt
```

`.env` holds the Alpaca key pair (already configured). Both trading and
market data use the same keys.

## Usage

```
python train.py        # download data, train, calibrate (artifacts/)
python backtest.py     # per-strategy evaluation on held-out data
python trade.py --once # single scan (signals only if market closed)
python trade.py        # run the bot continuously
```

Tune tickers, active strategies, bracket levels, and position sizing in
`config.py`.

## Disclaimer

This targets Alpaca's **paper** endpoint — simulated money. Do not point it
at a live account without months of paper results and your own validation.
