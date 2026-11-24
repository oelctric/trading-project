# minittraider

An AI paper-trader for crypto. The default strategy is **swing**: daily-bar trend
trades with fee-aware entries and a **dynamic exit manager** that decides, every
bar, how long a position deserves to live.

## Why it changed (and why the old scalper lost money)

The original bot scalped 1–5 minute moves. A 1-minute candle moves ~0.05%, but a
Coinbase round trip costs ~0.5% in fees — every trade started ~10 candles in the
hole. The old `backtest_trades.csv` shows it: ~$1 gross vs ~$12.50 fees per trade.
No signal quality can fix that, so the bot now only takes trades whose stop
distance is at least 3× the round-trip fee.

Hourly and 4-hour versions were also tested (`backtest_swing.py --tf 1h|4h`):
roughly break-even after costs, and tuned hourly settings fell apart out-of-sample.
Daily bars are where the edge held up.

## How it trades (`swing.py`)

**Entry** (on a closed daily bar): 20-day breakout in the direction of the trend
(EMA20 vs EMA100, price on the right side of EMA100), trend quality (efficiency
ratio) ≥ 0.2, and the move must be big enough relative to fees. Shorts mirror longs.

**Dynamic exit** — no fixed target, no fixed holding time:

| rule | what it does |
|---|---|
| initial stop | 2.0 × ATR from entry |
| break-even | after +1R, stop moves to entry + fees (a winner can't become a loser) |
| trailing stop | follows the best price minus 4 × ATR, tightens to 2 × ATR after +3R |
| trend broken | daily close back through EMA100 → exit |
| stalled | no +0.5R progress after 10 days → exit, capital goes elsewhere |

Winners are held as long as the trend keeps paying (trailing-stop exits average
+1.8R over ~17 days); dead and losing trades are cut early.

## Swing strategy results (walk-forward, fees + slippage included)

9 coins, Coinbase daily data 2016-11 → 2026-11. Settings are standard textbook
values (not optimized). Fees 0.25%/side + 0.05%/side slippage; R = the amount risked per trade.

| period | trades | win % | avg R/trade | profit factor | coins profitable |
|---|---|---|---|---|---|
| train 2016-11 → 2023-07 | 235 | 46% | +0.23 | 1.60 | 7/9 |
| **test 2023-07 → 2026-11 (unseen)** | 206 | 48% | **+0.28** | **1.81** | **8/9** |

### Aggressiveness, one $100 account, trades in parallel (fee 0.6%/side)

The live bot holds up to 6 coins at once on one shared account. Levels compared with
`python backtest_swing.py --levels --fee 0.006` (add `--long-only` for Coinbase spot):

| level | risk/trade | max/position | open at once | $100 → test Jul 2023–now | worst drawdown | last 12 months |
|---|---|---|---|---|---|---|
| old | 1% | 25% | 2 | $106 | 14% | $107 |
| **moderate (live)** | 2% | 50% | 6 | **$170** | 32% | $131 |
| aggressive | 3% | 50% | 6 | $128 | 44% | $137 |
| very aggressive | 5% | 100% | 9 | $72 | 68% | $127 |

Long-only, moderate: $100 → $152 over the test period (worst drawdown 26%), but $94 over
the last 12 months. More risk than "moderate" made *less* money out-of-sample — big
drawdowns compound against you. Expect to see the account down ~25–30% from a peak at some point.

### Leverage (paper futures account, `leverage.py`)

Leverage is borrowing, so the sizer decides how much to *risk*, then borrows only
what that needs: 3% of equity risked per trade, total borrowing capped at 2× equity
(average use ~0.6×), risk halved while the last 10 trades are net losing, and each
position's leverage capped so its liquidation price is ≥ 2× further away than its stop.
Futures costs are modelled: 0.05%/side fees + 0.03%/day funding.

`python backtest_swing.py --leverage` ($100, one account, coins in parallel):

| setup | test Jul 2023–now | worst drawdown | last 12 months |
|---|---|---|---|
| 2% risk, no leverage | $200 | 29% | $137 |
| 3% risk, 2×, no brake | $250 | 41% | $156 |
| **live: 3% risk, 2×, cold-streak brake** | **$250** | **32%** | $144 |
| 5% risk, 3×, cold-streak brake | $346 | 49% | $173 |

The cold-streak brake is what makes the leverage usable: same return as plain 3%/2×,
with the worst drawdown cut from 41% to 32% (and 38% → 27% on the 2016–23 data).
A drawdown brake and a total-heat cap were also tested and *hurt* out-of-sample, so
they're off. Long-only (`--long-only`): $193 over the test period, $94 over the last 12 months.
Set `USE_LEVERAGE = False` in `config.py` for the old unleveraged spot sizing.

Robustness checks:
- all 972 nearby parameter combinations were profitable out-of-sample (median +0.27R/trade)
- still profitable at 3× fees (+0.29R/trade out-of-sample)
- dynamic exits beat fixed TP/SL/time exits (+0.39R vs +0.28R/trade, long-only, full history)
- the live code path replays history with identical exits (`tests/test_swing_parity.py`)

Honest caveats: it is a trend follower, so it bleeds in choppy years (2025 was
−9.7R with shorts on). Backtests are not live results — trade-to-trade variance is
high and ~200 trades is still a modest sample. Shorts are on for the paper account;
**Coinbase spot can't short** — for real money you'd need perps/margin, or set
`SWING_ALLOW_SHORT = False` (long-only was also profitable overall, but flat in bear years).

## Retraining (`retrain.py`) — Nov 2026

The tables above were produced with the original settings (stop 2.5 ATR, trail 3 ATR,
stall 15 days). Retraining changed them to stop 2.0, trail 4.0, stall 10:

- `python retrain.py` simulates 324 candidate settings and shows, for every year since
  2019, what retraining on *only the data before that year* would have picked and how it
  did the following year.
- Re-picking every year and switching blindly was **worse** than fixed settings
  ($100 → $262 vs $419 since 2019): the yearly winner jumps around, which means noise.
- But the 2024, 2025 and 2026 retrains all converged on stop 2.0 / trail 4.0, and the
  Jan-2024 pick beat the old settings on the unseen 2024→now data: **$175 vs $156** per
  $100 (worst drawdown 37% vs 32%), last 12 months $152 vs $144. That pick is live now.
- `--apply` only writes a new pick if it matches the last two yearly retrains.

**Trade-quality model** (`retrain.py --quality`): a logistic regression scoring each
signal from 11 entry features, retrained walk-forward each year. Its weights show
high-volatility coins (−0.51) and over-stretched RSI (−0.33) make breakouts fail, steady
20-day momentum (+0.34) helps. It's only weakly predictive (AUC 0.56) and using it to
skip or shrink trades did **not** improve returns, so it isn't used live.

The old `meta_q.json` (cognitive strategy Q-table) learned that 299 of 300 market states
have value ≤ 0 — i.e. minute-scale trading doesn't pay — consistent with the backtests.

## Extra indicators (`swing.py` FILTERS) — Nov 2026

Ten indicators were added as optional entry confirmations: ADX, MACD, volume surge,
Bollinger squeeze, weekly trend, OBV, RSI cap, volatility regime (`calm_vol`), 20-day
momentum and strong close. Each was tested alone and in combinations, choosing on
2016–Jul 2023 and judging on unseen Jul 2023→now ($100, live leverage settings):

| entry rule | 2016–2023 | Jul 2023→now (unseen) | worst DD (unseen) |
|---|---|---|---|
| breakout only (before) | $352 | $338 | 37% |
| **+ calm_vol (live)** | **$455** | **$403** | **32%** |
| + macd | $374 | $373 | 34% |
| + adx | $253 | $376 | 28% |
| + weekly trend | $426 | $298 | 38% |
| + strong close | $346 | $242 | 34% |
| best 5-indicator stack (picked on 2016–23) | $388 (DD 11%) | $211 | 21% |

- **`calm_vol`** — skip breakouts when volatility (ATR %) is in the top 20% of its past
  year. Best single indicator on 2016–23, better on the unseen period, helped 9 of 10
  years, and every threshold from 60% to 90% beat no filter. The trade-quality model's
  biggest weight (high volatility −0.51) pointed at the same thing. **Enabled.**
- Stacking indicators is a trap: each one added made 2016–23 look better and the unseen
  period worse (5 filters: $211). Only one is on.
- All indicators are causal (`tests/test_indicators.py` checks nothing changes when
  future bars are added). Enable others via `"filters"` in `swing_params.json`.

### Round 2 (Nov 2026): 11 more indicators, tested on top of `calm_vol`

Relative strength vs BTC, market breadth, Supertrend, Ichimoku, Aroon, choppiness,
stretch cap, breakout-bar range, no-weekend, stochastic cap, MACD. Rule: it must help on
2016–Jul 2023 **and** on unseen Jul 2023→now.

| added to calm_vol | 2016–2023 | Jul 2023→now | worst DD (now) | last 12m |
|---|---|---|---|---|
| nothing | $455 | $403 | 32% | $149 |
| **+ MACD (live)** | **$465** | **$430** | **31%** | **$161** |
| + breadth | $482 | $293 | 31% | $146 |
| + stochastic cap | $267 | $562 | 30% | $182 |
| + no weekend | $447 | $453 | 24% | $158 |
| + Supertrend / Ichimoku / Aroon | $386–389 | $328–355 | 34–35% | $144–160 |

Only MACD passed both. It held up with 4 of 5 different MACD speeds, at 2× fees, and
mostly by lowering drawdowns; the gain is small. Indicators that only looked good on the
newer period (stochastic cap, no weekend) were rejected — choosing them would be hindsight.

## More coins? (tested Nov 2026 — no)

Universes of the top 15/20/30/50 Coinbase coins by trading volume, chosen using only the
year *before* each period (no hindsight), all did worse on Jul 2023→now than the 9 majors:
average trade fell from +0.36R to +0.18–0.25R and worst drawdowns rose to 41–58%
(last 12 months: $204 for 9 coins vs $83–$134 for bigger lists). Smaller alts make
noisier, more false breakouts. Survivorship bias (delisted coins are missing) flatters the
big lists, so the real gap is likely wider. The 9-coin list stays.

## Live safety & monitoring

- A stale 1-minute feed no longer force-closes a multi-day swing position; it's held and
  re-checked next scan.
- If the bot was offline (or price wicked through the stop between scans), each missed
  daily bar is checked for a stop hit on restart and filled at the stop price — the same
  rule as the backtest (`tests/test_swing_parity.py::test_offline_stop_is_not_missed`).
- `trades.csv` now records strategy, R-multiple, risk, leverage and margin per trade
  (the old-format log is moved to `trades_old_format.csv`).
- `python report.py` compares live swing trades with the backtest's expectation and only
  calls a gap real when it's statistically significant (needs ~20 trades, ~4 months).

## Bug hunt (Nov 2026) — fixed

1. Live break-even stop and fee check used spot fees (0.25%) while the account pays futures
   fees (0.05%) — exits didn't match the backtest. Both now use `swing_live.fee_per_side`.
2. A failed daily-bar download was cached until the next bar — the bot was blind for a day.
   Failures are no longer cached; it falls back to the last good bars and retries.
3. History downloads silently skipped chunks that failed → holes in the data. Now they raise.
4. "Already traded this signal" was forgotten on restart → could re-buy a stale signal.
   Now persisted in `swing_signals.json`.
5. Market breadth only saw coins scanned earlier in the same scan. All bars are prefetched.
6. A download error on the entry path crashed the whole scan (later coins' stops unchecked).
7. Signals acted on hours late (bot off at midnight) — now refused after 3h
   (`SWING_MAX_SIGNAL_AGE_HOURS`), since the backtest buys right at the bar close.
8. Swing mode downloaded stock data every minute just to skip it.
9. GUI "reset" could be undone by a scan in progress (race). Scans now hold a lock.
10. Weekly-trend indicator was a week stale (double shift).
11. Backtest skipped the exit bar's own signal; live doesn't → backtest now matches live.
12. A stop hit in an earlier bar could be reported as a liquidation at today's price.
13. Lint: unused imports/variables.

14. File logging never worked: modules created the logger before `trader.py` asked for a
    log file, and that request was silently ignored (the `logs/` folder stayed empty).
    `LOG_LEVEL` was ignored the same way. The GUI never logged to a file at all.
15. The log file never rolled over at midnight — a multi-day run wrote into its start day's file.
16. Cognitive strategy's risk manager used the wall clock in backtests (daily loss limit and
    loss-streak cooldown never applied to historical bars).
17. Bounce strategy's "rejection" check was OR'ed with "touched the zone" (always true there),
    so it never actually required a rejection candle.

18. GUI could re-enter on the same scan right after a swing exit (stale signal bar); now it
    moves on like `trader.py`. Multi-day position ages now show as hours/days.

19. `evolve.py` had the same OR'ed rejection check as bounce.py, so its `reject` gene did nothing.

20. (review) a swing signal was recorded as "traded" before the position opened -> a refused
    entry could never be retried; now recorded only on open.
21. (review) `USE_LEVERAGE` applied futures sizing/fees/funding to EVERY strategy -> swing only.
22. (review) a second trades.csv format migration could append rows under the old header ->
    each old log gets a unique name.
23. (review) `backtest.py` (default strategy swing) silently ignored `--days/--symbols` -> forwarded.

24. (review) the swing risk increase had also doubled the risk of the untested minute
    strategies -> separate SWING_RISK_PER_TRADE / SWING_MAX_POSITION_PCT; minute strategies
    back to their tested 1% / 25%. `backtest.py` forwards swing-only options (--tune, --leverage...).

25. (review) the per-group limit raised to 6 for swing also applied to the minute strategies ->
    SWING_MAX_POSITIONS_PER_GROUP (6) for swing, MAX_POSITIONS_PER_GROUP back to its tested 2.

26. (review) isolated-margin loss cap / zero-equity floor now only for leveraged futures
    positions (spot and unlevered shorts can lose more than they posted); minute strategies skip a
    trade that doesn't fit in cash instead of opening it undersized; GUI reset runs on the scanner
    thread (no UI freeze); swing entries log "tp dynamic", not 0.

27. (review) the first daily bar after a swing entry started before the entry; its earlier
    low/high could trigger the stop or inflate the best price -> only its close is used (live
    stops after entry are checked every scan). A gap through the liquidation level is labelled
    as a liquidation. `backtest.py` defaults to the minute strategy it documents again.

28. (review) only swing trades feed the swing cold-streak brake (minute-strategy closes used to
    halve swing risk).

Results after the fixes are within noise of before ($434 vs $430 per $100, unseen period).
`tests/test_live_robustness.py` covers 2, 3, 4, 7 and the fee match.

## Running it

```
python gui.py               # desktop app (or double-click run_gui.bat)
python trader.py            # terminal loop, scans every 60s
python trader.py --once     # one scan then exit
python backtest_swing.py    # re-run the walk-forward backtest
python backtest_swing.py --long-only
python backtest_swing.py --tune     # re-tune on the train slice only (writes swing_params.json)
python tests/test_swing_parity.py
python tests/test_leverage.py
python retrain.py [--quality] [--apply]
python report.py            # live vs backtest check
python tests/test_indicators.py
python tests/test_live_robustness.py
```

The bot scans every minute: stops are checked against the live price all day,
while entries and stop adjustments happen once per day when the daily bar closes
(00:00 UTC). Expect only a few trades per month across all coins.

## Files

| file | what it does |
|---|---|
| `config.py` | all the knobs (`STRATEGY`, risk, fees, swing options) |
| `swing.py` | swing entries + `ExitManager` (shared by backtest and live) |
| `swing_live.py` | live bar fetching, decisions and position management |
| `swing_params.json` | the strategy settings in use |
| `backtest_swing.py` | walk-forward backtest / tuner |
| `history.py` | historical candle downloader + cache (any bar size) |
| `portfolio.py` | paper account, sizing, trade log |
| `trader.py` / `gui.py` | terminal loop / desktop app |
| `cognitive.py`, `bounce.py`, `brain.py` | older minute-scale strategies (kept; backtested with no edge) |

Paper state lives in `paper_state.json` and closed trades in `trades.csv`
(delete `paper_state.json` to reset the $10,000 account).
