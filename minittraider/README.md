# minittraider

An AI paper day-trader for **crypto and stocks** that scalps **1–5 minute trades** —
but only when it understands the situation.

## The idea

Markets are usually unpredictable on a 1–5 minute horizon. But sometimes several
*independent* things line up at once — trend, momentum, a volume surge, price
position vs. VWAP, orderly candle structure — and in those moments a short scalp
becomes meaningfully more predictable. This bot's whole personality is:

- **It knows when it doesn't know.** If signals conflict, or the market is asleep
  (no volatility) or erratic (too much volatility), it sits out and tells you why.
- **It trades confluence, not memory.** Every decision is computed fresh from live
  data; nothing is a canned prediction. It needs 3+ independent signals agreeing
  *and* 60%+ confidence *and* a tradeable volatility regime before it acts.
- **Every trade is a scalp.** Take-profit at 1.5× the average candle range,
  stop-loss at 1×, and a hard 5-minute time limit no matter what.

## Running it

**Desktop app (recommended):** double-click `run_gui.bat`, or run `python gui.py`.
Press **Start** to begin scanning. Click any watchlist row to see the bot's full
reasoning for that symbol. `python gui.py --autostart` starts scanning immediately.

**Terminal version:**
```
python trader.py          # scans every 60s, trades, prints decisions
python trader.py -v       # same, but shows the full reasoning for every symbol
python trader.py --once   # one scan then exit (good for testing)
```

Both share the same brain and the same paper account.

Stop it with Ctrl+C. Crypto trades 24/7; stocks only during US market hours
(9:30am–4pm ET) — off-hours the stock symbols are skipped automatically.

## Money

It's **paper trading**: fake $10,000, live real prices, honest bookkeeping.
- `paper_state.json` — current cash, open positions, running P&L (survives restarts)
- `trades.csv` — every closed trade with entry/exit, P&L, and the reason it entered

To reset the account, delete `paper_state.json`.

## Tuning

Everything lives in `config.py`: watchlist, risk per trade (1%), max hold time,
confidence bar, how many signals must agree, whether shorts are allowed.

## Files

| file | what it does |
|---|---|
| `gui.py` / `run_gui.bat` | native Windows desktop app (dashboard) |
| `config.py` | all the knobs |
| `data.py` | live 1-min candles (Coinbase public API for crypto, Yahoo for stocks) |
| `signals.py` | six independent signals + volatility-regime detector |
| `brain.py` | confluence engine: combines evidence, decides, explains |
| `portfolio.py` | paper account, position sizing, TP/SL/time exits, trade log |
| `trader.py` | main loop |

## Before you ever consider real money

Let it paper trade for **at least several weeks**, then look at `trades.csv`:
win rate, average win vs. average loss, and whether realized P&L is positive
*after* you mentally subtract fees (~0.1–0.6% per crypto trade — this sim
doesn't charge fees, real exchanges do, and fees kill most scalping strategies).
Most minute-scale bots lose money in real conditions; demand proof first.
