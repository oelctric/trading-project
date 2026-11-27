# Trading Project

Three side-by-side projects for testing trading ideas with **fake money first**,
plus a browser helper for a local AI model.

| folder | what it is | trades | money |
|---|---|---|---|
| [`minittraider/`](minittraider/README.md) | Crypto **swing** bot (holds trades for days) with a desktop app | 9 big coins on Coinbase daily bars | paper account ($10,000 simulated) |
| [`daytraider/`](daytraider/README.md) | US stock **day-trading** bot + a pairs bot + research lab | 40 liquid US stocks, hourly bars | Alpaca **paper** account |
| [`browser-model-extension/`](browser-model-extension/README.md) | Chrome extension that lets a local LLM read and click web pages | — | — |

> **Not financial advice.** Everything here points at paper (simulated) accounts.
> The edges found are small and honestly measured. A good backtest is not a
> promise of future profit.

---

## Quick words you'll see a lot

- **Backtest** — replaying old price data to see how a strategy *would have* done.
- **Walk-forward** — a stricter backtest: pick settings using only the past, then
  test on the next chunk you haven't looked at. Like studying old exams, then
  sitting a new one.
- **Out-of-sample / unseen** — data the strategy was never tuned on. The only
  numbers that really count.
- **Paper trading** — real market prices, fake money.
- **R** — the amount you risk on one trade. +2R means you made twice what you risked.
- **bps** — basis points; 1 bp = 0.01%. So +4 bps = +0.04% per trade.
- **Drawdown** — how far the account drops from its highest point before recovering.
- **ATR** — Average True Range; how much a price usually moves in one bar. Used to
  place stops a sensible distance away.
- **Leverage** — borrowing to make a position bigger. Bigger wins *and* bigger losses.
- **Lookahead / leak** — a bug where the test accidentally peeks at future data.
  It makes results look great and then fails live.

---

## 1. minittraider — crypto swing bot

**Idea:** Small, fast crypto trades lose to fees (a 1-minute move is ~0.05%, but
a round trip costs ~0.5%). So this bot only trades **daily** trend breakouts and
lets winners run.

- **Entry:** 20-day breakout in the trend's direction (EMA20 vs EMA100), skipped
  when volatility is unusually high (`calm_vol`) or MACD disagrees.
- **Exit (dynamic):** 2 ATR starting stop → moves to break-even at +1R → trails
  4 ATR behind the best price (2 ATR after +3R) → exits if the trend breaks or
  the trade stalls for 10 days.
- **Sizing:** 3% of the account risked per trade, borrowing capped at 2×, and
  risk halved after a losing streak ("cold-streak brake").

**Results on unseen data (Jul 2023 → Nov 2026, fees included):** ~+0.28R per trade,
profit factor 1.81, 8 of 9 coins profitable. With the live settings $100 grew to
roughly $400+ in the backtest, but with drawdowns around **30%** along the way.
It's a trend follower, so it loses in choppy years (2025 was negative).

```bash
cd minittraider
python gui.py                 # desktop app (or double-click run_gui.bat)
python trader.py              # terminal loop, scans every 60 s
python trader.py --once       # one scan, then exit
python backtest_swing.py      # re-run the walk-forward backtest
python report.py              # compare live trades vs backtest
```

Settings: `config.py` (strategy, risk, `USE_LEVERAGE`, `SWING_ALLOW_SHORT`) and
`swing_params.json` (the strategy numbers). Delete `paper_state.json` to reset the account.

Needs: `pandas numpy requests yfinance` (tkinter comes with Python).

---

## 2. daytraider — stock day-trading bot

**Idea:** Rule-based setups (breakouts, gap fades, opening-range breaks, pullbacks…)
pick the trades. A "regime gate" only lets a strategy trade if it has been working
lately. A 5-model LSTM neural network only **ranks** setups when there are more
than open slots — testing showed it's close to a coin flip on its own.

- Bracket orders: +1.5 ATR take-profit / −1.0 ATR stop, trailing stop after entry.
- Risk: each stop-out loses ~0.25% of the account, max 4 positions, max 2 per
  sector, stop trading for the day at −1%, close everything 15 min before the bell.
- Also includes a **pairs bot** (trades two related stocks against each other) and
  a **crypto bot**; all three can share one Alpaca account without stepping on
  each other's positions.

**Results:** +4.3 bps per trade with the gate over 57 unseen months, 56% of months
positive. Thin edge — the README calls it "an experimentation platform, not a
money printer".

```bash
cd daytraider
pip install -r requirements.txt
python train.py          # download data + train the models (into artifacts/)
python backtest.py       # per-strategy results on held-out data
python walkforward.py    # the main way to judge any change
python trade.py --once   # one scan
python trade.py          # run the stock bot
python pairs_trade.py    # pairs bot
python crypto_trade.py   # crypto bot
```

Create `daytraider/.env` with your Alpaca **paper** keys:

```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_ENDPOINT=https://paper-api.alpaca.markets/v2
```

⚠️ `robinhood.py` talks to a **real-money** Robinhood account
(`ROBINHOOD_API_KEY`, `ROBINHOOD_PRIVATE_KEY`). Leave it alone unless you mean it.

The other ~30 scripts (`discover.py`, `pure_ml.py`, `lobsim.py`, `hft_rl/`, …) are
research experiments. Most of them showed **no** edge, which is written up in the
folder's README.

---

## 3. browser-model-extension — local AI browser agent

A Chrome side panel that connects to LM Studio (or any OpenAI-compatible server)
and gives the model tools: read page, click, type, navigate, scroll, screenshot.

1. `chrome://extensions` → turn on **Developer mode** → **Load unpacked** → pick this folder.
2. Open the side panel, set the endpoint (e.g. `http://<your-lm-studio-ip>:1234/v1`)
   and the model name, then **Test connection**.
3. Type a task and press **Run**.

Safety: it acts in *your* logged-in tab. Password/OTP fields are hidden from the
model, screenshots are never saved, and "Confirm each action" asks before every
click/type/navigate.

---

## Tests

```bash
# minittraider
python minittraider/tests/test_swing_parity.py
python minittraider/tests/test_leverage.py
python minittraider/tests/test_indicators.py
python minittraider/tests/test_live_robustness.py

# daytraider (offline, fake broker)
python daytraider/tests/test_bot_fixes.py

# browser extension (Node)
node browser-model-extension/tests/background.test.js
node browser-model-extension/tests/stop_midbatch.test.js
JSDOM_PATH=<folder with node_modules/jsdom> node browser-model-extension/tests/content.test.js
```

---

## What's not in git

On purpose, `.gitignore` keeps these out (they're secret, huge, or change every run):

- `.env` files and keys
- `artifacts/`, datasets and trained models (`*.npz`, `*.pt`, … — several GB; rebuild with `train.py`)
- logs, `trades.csv`, `paper_state.json`, and live bot state files (`pairs_state.json`, `stock_trail.json`, …)

## Repo layout

```
trading project/
├── README.md                  ← you are here
├── minittraider/              crypto swing bot + GUI
├── daytraider/                stock day-trading, pairs and crypto bots + research
│   └── hft_rl/                reinforcement-learning experiment (no edge found)
└── browser-model-extension/   Chrome extension for a local LLM
```

Each folder's own README has the full details: every test, every number, and the
list of bugs found and fixed in the Nov 2026 bug hunt.
