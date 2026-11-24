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


## Bug hunt (Nov 2026) — fixed

Live bots (`tests/test_bot_fixes.py` covers the starred ones, offline with fake brokers):
- *Stock bot: after a restart, a stop that had already trailed to entry rebuilt a ~0 trail
  distance and moved the stop onto the price -> instant exit. Trail state is now persisted
  (`stock_trail.json`) with a safe fallback.
- *Stock bot: with `nested=True`, bracket stop legs live inside the parent order's `legs`;
  the trailer only looked at top-level orders.
- *Stock bot `flatten_all` (3:45pm) closed the PAIRS bot's overnight legs on the shared
  account. Pairs legs (`pairs_state.json`) are now skipped by the stock bot.
- Stock bot "minutes since open" assumed 6.5h sessions -> broke on half-days. Uses 09:30 ET.
- *Live decisions used a 2-minute stub of the current hourly bar (scan at :18, end=:02).
  Unfinished bars are dropped in `download_bars`.
- *Regime gate scored trades that rode overnight, but the bot flattens daily. Simulated
  trades now end at their session close.
- Training labels leaked LABEL_HORIZON bars across train/val/test boundaries -> purged.
  (NN labels still span overnight; fixing that needs a retrain - see note below.)
- Pairs bot forgot open pairs on restart (orphaned legs) -> persisted; used fractional SHORT
  shares (Alpaca rejects -> one naked leg) -> whole shares + unwind leg A if leg B fails;
  sized equal dollars instead of the tested 1:beta hedge -> beta-weighted.
- *Crypto bot rounded limit prices to 2 decimals ($0.08 coins never filled, sub-cent -> $0)
  and forgot trailing highs on restart -> 6-significant-digit prices, state persisted.
- *Robinhood (REAL MONEY): `keygen` silently overwrote an existing private key -> now refuses
  without `--force`; order quantities could serialize as "1e-05" and weren't rounded to the
  pair's increment -> plain decimals rounded down to `asset_increment`.
- Research: pairs backtests measured notional at exit (and without abs(beta)) -> at entry.
- Lint: unused imports/variables and placeholder-less f-strings.

- NN training labels let a trade run past the session close (the bot flattens daily).
  Labels are now session-capped and the 5-model ensemble was retrained (Nov 22 2026).
  `python compare_models.py` on the held-out test set, scored on same-day outcomes:

  FINAL numbers (after the code-review fix that made each session's last bar NaN instead
  of a fake 0-return label; earlier figures of AUC 0.63 / +5.87 bps were inflated by it):

  | | overnight-label models | first session models (fake-0 last bar) | **final (live)** |
  |---|---|---|---|
  | AUC, all bars | 0.486 | 0.509 | 0.508 |
  | top-20%-ranked setup signals, avg return | +3.05 bps | +5.88 bps | +2.33 bps |

  All three are ~coin-flip rankers (setup-signal AUC 0.499-0.504); the top-20% gaps are within
  noise (~2.3 bps standard error on ~2,800 trades). The final models are live because they're
  trained on correct labels - picking by test score would be selection on the test set.
  The NN is a tie-breaker at best; the edge comes from setups + the regime gate.
  Backups: `artifacts/backup_overnight_labels_2026-11-22/`, `artifacts/backup_session_labels_v1_2026-11-22/`.

Round 2 (full read of every remaining file):
- *`manipulation.fake_breakout` peeked at the NEXT bar (`failed.shift(-1)`) - impossible live,
  and it manufactured the "-13.6 bps" evidence for `FAKE_BREAKOUT_FILTER`. Now causal.
  Re-measured honestly (`measure_fake_breakout.py`, 63k signals): flagged trades averaged
  **+12.4 bps vs -2.3** unflagged (t=+3.2, final labels) -> the filter was skipping the better trades. It is
  now **off**.
- *`stophunt_high/low` compared against the previous hour's high/low, not yesterday's.
- Live regime gate used N*7 bars (= N trading days, ~40% longer) instead of the N CALENDAR
  days `walkforward.py` validated. Now calendar days.
- `discover.py` scored rules on the fixed bracket while the bot trades the trailing exit.

Round 3 (line-by-line read of every remaining research script):
- `backtest.py` compounded simple returns with exp(sum) (log-return formula) -> prod(1+r)-1;
  its docstring claimed the live exit (it scores the fixed bracket).
- `walkforward.py` gated all 16 strategies incl. the 4 shorts disabled in config -> now only
  ACTIVE_STRATEGIES (`--all` for everything). `trail_sweep.py` simulated overnight holds -> session-capped.
- Crypto: validation + live regime gate charged 15 bps round trip, but the bot pays maker in
  (~15) + MARKET out (~25) = ~40 bps; live bars could include the unfinished hour;
  `crypto_feasibility.py` labelled 25 bps "realistic" (it's 30-50 round trip).
- Test-set leaks: `explain.py` ranked features on TEST and `pure_ml_slim.py` selected features
  from that ranking -> now validation, and the SLIM list was re-derived from it; `pure_ml(_slim).py` took
  band cut-offs from test predictions -> validation. `big_model.py` lacked the boundary purge.
- Minute scripts: 15-min ATR was start-labelled (leaked up to 14 future minutes) in
  `minute_bot_backtest.py` and `hft_rl/env.py`; trades ran overnight; `minute_rigorous.py` /
  `minute_timing.py` entered INSIDE the signal's own hourly bar (bars are start-labelled).
- `hft_rl`: env advanced 1 minute after a trade (overlapping positions); each env normalised with
  its own data (test peeked at itself) -> shared training stats.
- `lobsim.py` built both book sides from the mid, discarding the real bid/ask.
- `weekly_retrain.py`: a mid-way failure left new + old seeds with a new scaler -> one-run
  retrain with automatic backup/restore; scheduler path pointed at the old folder.

Re-run with the fixes (Nov 22 2026):
- `explain.py` (validation, final labels): top features vix_z +32, sma50_dist +27,
  effort_result +21, hour_cos +18 bps. (An intermediate run showed hour-of-day at +190 bps -
  that was an artifact of the fake-0 last-bar label, since fixed.)
- `pure_ml_slim.py` with the honest 10-feature list: the network trading ALONE loses
  -4.0 to 0.0 bps/trade at every band -> no standalone edge; setups must drive entries.
- `crypto_validate.py` at the real 40 bps round trip: trailing exit +3.9 bps/trade, only 44% of
  months positive, and Feb 2026 alone exceeds the whole net total -> no reliable crypto edge.
- `measure_fake_breakout.py` (final labels): flagged trades +12.4 vs -2.3 bps (t=+3.2)
  -> FAKE_BREAKOUT_FILTER off.

Round 4 - every research script was RUN end to end (24 scripts across both projects, all exit 0):
- `lobsim.py`: off-hours quotes (e.g. SPY 671/714, a 6% "spread") produced garbage -> stale
  quotes are detected and the book is centred on the last trade. The simulator itself never
  refilled consumed levels, so its "random walk" was two edges creeping outward (the nearer
  barrier won ~100%) -> the book now re-quotes around each trade (balanced book: ~32-36% for
  +1.5/-1 ATR, theory ~40%; thin asks 88%, thin bids 0%). Undecided runs print a clear message.
- `big_model.py` compared against a hard-coded 0.556 accuracy from old models -> now measures the
  current production ensemble on the same test set.

Round 5 - code review of all of the above changes (`/code-review`) found 8 more, all fixed:
- Extension: a new run could be broken by the old run's cleanup (debugger detach, running flag)
  -> a new run waits for the previous one to finish; `drag` added to the tool schema.
- Pairs unwind used close_position on a possibly-unfilled order -> cancel by id + reverse only
  the filled qty.
- Session-capped labels gave each session's last bar a fake 0 return (1/7 of rows) -> NaN;
  ensemble retrained again; dataset caches renamed (dataset_v3 / dataset_big_v3) so stale label
  caches can't be reused. Old caches `dataset_v2.npz` (2.3 GB) and `dataset_big.npz` (6.7 GB)
  are now unused and can be deleted.
- Crypto bars filtered against now-5min dropped the just-closed hour -> filtered against now.
- weekly_retrain could restore a stale backup -> fresh backup dir per run, restore only this run's.
- minittraider marked a swing signal "traded" before the position opened -> marked only on open.

Round 6 - second code review:
- The pairs bot now also treats symbols with PENDING orders as taken, so a pair leg can't merge
  into the stock bot's not-yet-filled position (a symbol is never shared by both bots, which is
  what lets the stock bot's flatten skip pair legs safely).

Round 7 - third code review:
- Pairs: a pair whose close failed on one leg was only retried when z crossed a threshold again
  (one leg could sit naked for days) -> flagged `closing` and retried every scan
  (`manage_exits()`, tested).
- `botstate.save` could raise PermissionError on Windows/OneDrive file locks right after orders
  were placed -> retries, never raises.
- Pair legs no longer consume the stock bot's position slots (they're only skipped).

Round 8 - fourth code review: pairs-bot legs no longer count toward the stock bot's per-sector caps.

Round 9 - fifth code review:
- Pairs unwind only trusts "no fill" after SEEING a terminal status with 0 filled; unknown /
  pending / partial unwinds go on a persisted orphan list (`pairs_orphans.json`) that is resolved
  every scan, and other bots keep off those symbols.
- A pair stays `closing` until both legs are confirmed flat (an accepted close order can still be
  rejected/expire); both legs are sized within the per-leg budget and checked against buying power.
- Pairs research skips negative-beta pairs, like the live bot.
- lobsim: price impact now interpolates through each level, so a 2:1 book really tilts the walk
  (it used to need ~10:1 before small orders ever left level 0).
- lobsim (found by re-running after the fix above): random size noise was applied to the REAL
  touch sizes too, so even a balanced 40x40 book got a persistent random tilt (SPY: 98% "win").
  The touch now uses the quoted sizes exactly; SPY balanced -> 36.7% (theory ~40%).

Round 10 - sixth code review:
- A locked/unreadable pairs_state.json was treated as "no pairs", so the stock bot could close
  the pairs bot's legs (and a restarting pairs bot could forget open pairs). `botstate.load_strict`
  retries and raises StateUnavailable; the stock bot then skips entries / retries the flatten in
  30 s, and the pairs bot refuses to start rather than overwrite its state.
- Pair entries are confirmed: both order ids are stored and `reconcile_entries()` waits for both to
  FILL; if one is rejected/expires the other is unwound. Unwinds track their reversing orders
  until those FILL (re-issued if rejected) - an accepted reverse order no longer counts as done.
- Leg B quantity is floored so it can't exceed the per-leg budget.

Round 11 - seventh code review:
- reconcile_entries treated "one leg filled, the other still filling" as a failure ("filled" is a
  terminal status) and unwound healthy pairs -> unwinds only when a leg FAILED (canceled /
  expired / rejected) or both finished without both filling fully.
- Orphan reverses are capped at the shares still held in that direction (a position closed by
  hand no longer gets "reversed" into a new opposite position; nothing is sent for 0 shares).
- The pairs bot only places entries/exits/reverses while the market is open (after-hours DAY
  orders filled on hours-stale signals).
- Stock bot: trailing picks only LIVE stop legs (new/accepted preferred); crypto-bot coins no
  longer use stock-bot slots or sector caps. minute_bot_backtest counts only tradable signals.

Round 12 - eighth code review:
- reconcile_entries detects a failed entry at any time but only sends unwind orders while the
  market is open (it used to queue after-hours reverse orders).
- unwind_leg keeps an existing orphan record (and its working reverse orders) instead of
  resetting it, so a repeated unwind can't send a duplicate reverse order.
- lobsim's stale-quote fallback also replaces the dead quote's sizes with a balanced book.

Round 13 - ninth code review:
- Orphan (unwind) records are mirrored in memory, so a locked/unwritable orphan file can't make a
  filled leg drop out of tracking; a failed-entry pair (or a guard entry) stays in pairs_state until
  the orphan record is actually saved - the legs stay off-limits to the other bots meanwhile.
- The stock bot's close-out can no longer get stuck: if pairs state stays unreadable for 3 tries,
  it closes ONLY the positions it definitely owns (its trailing-stop state) and warns loudly.
