"""Central configuration for the day-trading neural network."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# ---- Alpaca (paper trading) ----
ALPACA_ENDPOINT = os.getenv("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# ---- Universe: liquid, high-volume names suited to intraday trading ----
TICKERS = [
    # index ETFs
    "SPY", "QQQ", "IWM", "DIA",
    # megacap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL",
    # semis / software
    "AMD", "AVGO", "QCOM", "MU", "INTC", "CRM", "ORCL", "ADBE", "NFLX",
    # high-beta growth
    "UBER", "PLTR", "COIN", "SHOP", "PYPL",
    # financials
    "JPM", "BAC", "GS", "WFC", "C",
    # energy / industrials / consumer / health
    "XOM", "CVX", "WMT", "COST", "HD", "DIS", "BA", "CAT", "UNH", "LLY",
]

# ---- Data (Alpaca market data API) ----
# Hourly bars: 15-minute bars were tested and lose money — the per-trade
# edge shrinks below trading costs at that resolution.
BAR_MINUTES = 60             # bar size for training and live signals
HISTORY_DAYS = 1825          # ~5 years of hourly bars for training
LIVE_LOOKBACK_DAYS = 150     # covers indicator warmup + the regime window
SEQ_LEN = 32                 # bars of history the model sees (~4.5 days)

# ---- Model ----
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3

# ---- Training ----
BATCH_SIZE = 256
EPOCHS = 60
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
TRAIN_FRAC = 0.70            # chronological split fractions
VAL_FRAC = 0.15              # remainder (~0.15) is the held-out test set
EARLY_STOP_PATIENCE = 10

# ---- Trading ----
# Entries are driven by setups; the NN ranks candidates when more setups
# fire than there are free slots. All eight setups (4 long + 4 short) stay
# structurally active: 5-year walk-through testing showed every strategy's
# edge flips sign between adjacent periods, so cherry-picking from any
# single backtest window is curve-fitting. Which ones trade on a given day
# is decided by the regime gate below.
# Out-of-sample test on 56 UNSEEN tickers: all long strategies generalized
# (positive), but reversal_dn/pullback_dn/orb_dn/gap_go_dn were NEGATIVE on
# unseen names — the short side largely doesn't generalize. Dropped those 4;
# kept the shorts that held up (breakdown, gap_fade_dn, momentum_dn).
ACTIVE_STRATEGIES = ["momentum", "breakout", "reversal", "pullback", "orb",
                     "gap_go", "gap_fade",
                     "disc_strongday_dip", "disc_orb_wide",
                     "momentum_dn", "breakdown", "gap_fade_dn"]

# Gate metric: "sharpe" (trailing mean/std of net returns > SHARPE_MIN) is
# the shipped choice — it beat raw-sum on BOTH independent unseen ticker
# sets (+8.5/+7.4 vs +5.6/+4.8 bps), unlike expectancy_hit which won on set
# #1 only (a fluke) and had a win-rate-floor failure mode. A Sharpe-like
# gate rewards risk-adjusted return without fighting the ~38% trailing hit
# rate. Still requires 30d net sum > 0 as the fast-decay cut.
GATE_METRIC = "sharpe"      # "raw_sum" | "sharpe" | "expectancy_hit"
GATE_SHARPE_MIN = 0.10
GATE_HIT_FLOOR = 0.40

# Pro-desk rule: long setups only fire above the session VWAP, short
# setups only below it (validated with walkforward.py).
VWAP_FILTER = True
# "Check the macro before trading micro" as a HARD gate was walk-forward
# tested and REJECTED: it blocks gap-fade shorts exactly when they work
# (gated avg +1.4 bps with it vs +3.1 bps without). The tape state stays
# available to the NN as the mkt_above_vwap feature instead.
MARKET_FILTER = False

# Move the stop to breakeven once a position is up this multiple of its
# initial risk ("make it a free trade").
BREAKEVEN_AT_R = 1.0

# Manipulation defense: skip breakout-family entries whose entry bar shows
# the fake-breakout signature. OFF since Nov 2026: the old "-13.6 bps" result
# came from a detector that peeked at the NEXT bar (it called a breakout fake
# because the following bar failed). With the causal detector
# (measure_fake_breakout.py, 63k signals, 5y, same-day trailing exits) flagged
# trades averaged +12.4 bps vs -2.3 unflagged (t=+3.2; final session-capped labels)
# - the filter was skipping the BETTER trades.
FAKE_BREAKOUT_FILTER = False
BREAKOUT_FAMILY = ["breakout", "orb", "disc_orb_wide", "breakdown", "orb_dn"]

# Order-book overlay: skip an entry when the live top-of-book is stacked
# hard against it (bid_size/(bid_size+ask_size) — longs need >= MIN, shorts
# need <= 1-MIN). Conservative threshold: only blocks extreme opposition.
# Not backtestable (no historical quotes on the free feed) — every entry
# logs its imbalance so the filter can be judged from live fills later.
QUOTE_IMBALANCE_MIN = 0.20

# ---- Regime gate: a strategy may only open trades if its own simulated
# bracket trades were net profitable over BOTH trailing windows (the fast
# window benches decaying edges early — walk-forward: +6.4 bps/trade and
# 63% positive months vs +4.3/56% for the slow window alone) ----
REGIME_GATE = True
REGIME_LOOKBACK_DAYS = 90
REGIME_LOOKBACK_FAST_DAYS = 30

# ---- Portfolio risk rules ----
DAILY_LOSS_LIMIT = 0.01      # stop trading for the day at -1% equity
MAX_PER_SECTOR = 2           # max concurrent positions per sector
SECTOR = {
    "SPY": "index", "QQQ": "index", "IWM": "index", "DIA": "index",
    "AAPL": "megatech", "MSFT": "megatech", "AMZN": "megatech",
    "META": "megatech", "GOOGL": "megatech",
    "NVDA": "semis", "AMD": "semis", "AVGO": "semis", "QCOM": "semis",
    "MU": "semis", "INTC": "semis",
    "CRM": "software", "ORCL": "software", "ADBE": "software",
    "NFLX": "software",
    "TSLA": "growth", "UBER": "growth", "PLTR": "growth", "COIN": "growth",
    "SHOP": "growth", "PYPL": "growth",
    "JPM": "banks", "BAC": "banks", "GS": "banks", "WFC": "banks",
    "C": "banks",
    "XOM": "energy", "CVX": "energy",
    "WMT": "consumer", "COST": "consumer", "HD": "consumer",
    "DIS": "consumer",
    "BA": "industrial", "CAT": "industrial",
    "UNH": "health", "LLY": "health",
}

# ---- Execution ----
SKIP_OPEN_MINUTES = 30       # no new entries in the first 30 min of chop
ENTRY_LIMIT_BUFFER = 0.001   # marketable limit: cap slippage at 10 bps
# Fallback threshold; the calibrated one (written by train.py to
# artifacts/threshold.json) wins. Used for diagnostics, not as a hard gate.
PROB_THRESHOLD = 0.55
THRESHOLD_PERCENTILE = 80
MAX_POSITIONS = 4            # base position count at 1x (no leverage)

# ---- Leverage (margin) ----
# The bot flattens before the close, so it qualifies for day-trade margin
# (up to ~4x buying power). LEVERAGE scales the number of concurrent
# positions (more DIVERSIFIED bets at the same per-trade risk), NOT the size
# of each — so 2x means up to 8 names, not 8% swings on one. A buying-power
# guard prevents over-ordering. WARNING: leverage multiplies losses and
# variance by the same factor; it does NOT improve the edge. Do not raise
# above ~2x until the edge is proven on live fills.
LEVERAGE = 2.0
COST_PER_TRADE = 0.0005      # 5 bps round-trip slippage assumed in backtest

# ---- Brackets & labels: ATR-scaled so hot names get wider stops ----
TP_ATR_MULT = 1.5            # take-profit distance = 1.5 x ATR (labels only)
SL_ATR_MULT = 1.0            # stop-loss distance = 1.0 x ATR
LABEL_HORIZON = 8            # bars a simulated bracket trade may run (~1 day)

# ---- Trailing-stop exit: validated to beat fixed brackets by a wide,
# out-of-sample margin (+21 vs +3 bps/trade on the strategies the gate
# keeps). Enter with SL -1 ATR and a far TP backstop, then ratchet the stop
# to trail TRAIL_ATR_MULT ATR behind the best price each scan. ----
TRAILING_EXIT = True
TRAIL_ATR_MULT = 1.0
TP_BACKSTOP_ATR = 3.0        # far take-profit so trailing does the work

# ---- Volatility-scaled sizing: equal dollar risk per trade ----
RISK_PER_TRADE = 0.0025      # lose ~0.25% of equity if the stop is hit
MAX_NOTIONAL_FRACTION = 0.25  # cap any one position at 25% of equity

# ---- Ensemble: average the probabilities of N independently seeded nets
# to wash out single-run training noise ----
ENSEMBLE_SEEDS = [0, 1, 2, 3, 4]

# ---- Artifacts ----
MODEL_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = MODEL_DIR / "model.pt"  # single-model fallback
SCALER_PATH = MODEL_DIR / "scaler.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"
