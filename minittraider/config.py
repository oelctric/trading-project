"""Settings for the mini trader. Edit this file to change what it watches and how it trades."""

from enum import Enum

# --- What to watch ---
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD",   # Coinbase product ids
                  "XRP-USD", "ADA-USD", "LINK-USD", "AVAX-USD", "LTC-USD"]
STOCK_SYMBOLS = ["SPY", "QQQ", "TSLA", "NVDA", "AAPL", "AMD", "META"]  # Yahoo tickers

# --- Paper account ---
STARTING_CASH = 10_000.0

# --- Risk / trade management (all trades are 1-5 minute scalps) ---
RISK_PER_TRADE = 0.01        # risk 1% of equity per trade
MAX_POSITION_PCT = 0.25      # never put more than 25% of equity in one trade
MAX_OPEN_POSITIONS = 6
MAX_HOLD_MINUTES = 5         # force-exit after this long, win or lose
TAKE_PROFIT_ATR = 1.5        # take profit at 1.5x recent average candle range
STOP_LOSS_ATR = 1.0          # stop loss at 1.0x recent average candle range

# --- Strategy ---
class Strategy(Enum):
    """Trading strategy modes.

    COGNITIVE: layered architecture: fuzzy perception -> learned meta-agent
                planner (picks TREND / REVERT / STAND_ASIDE per environment) ->
                fuzzy sub-agent executors -> hard risk-manager veto.
                Train the planner with: python train_cognitive.py
    BOUNCE: trade rebounds off large-order zones (order-book walls + price
            levels with repeated rejections), stop just beyond the zone,
            target the empty side. Few trades, asymmetric risk/reward.
    CONFLUENCE: the original multi-signal momentum engine (backtested: no edge)
    """
    COGNITIVE = "cognitive"
    BOUNCE = "bounce"
    CONFLUENCE = "confluence"


# Active strategy - use Strategy enum values
STRATEGY = Strategy.COGNITIVE
COGNITIVE_MAX_HOLD_MINUTES = 30
BOUNCE_MAX_HOLD_MINUTES = 30   # bounces need more room than momentum scalps
BOUNCE_MIN_RR = 1.5            # reward must be at least this many times the risk
BOUNCE_MIN_TOUCHES = 2         # price rejections needed to trust an inferred zone
BOUNCE_ZONE_ATR = 0.8          # how close (in ATRs) price must be to a zone to act
BOUNCE_WALL_RATIO = 3.0        # order-book bucket must be this many x median to count as a wall

# --- Simulated fees (charged on the paper account so results stay honest) ---
FEE_PER_SIDE = {"crypto": 0.0025, "stock": 0.0005}

# --- Brain: how much agreement is needed before it will trade ---
MIN_CONFIDENCE = 0.60        # 0..1, below this it sits out
MIN_AGREEING_SIGNALS = 3     # at least this many independent signals must point the same way
ALLOW_SHORTS = True          # paper account can short both stocks and crypto

# --- Backtest parameters ---
BACKTEST_SLIPPAGE_ATR = 0.15        # slippage as fraction of ATR (0.15 = 15% of ATR per trade)
BACKTEST_FEE_MULTIPLIER = 1.0       # multiply fees by this factor for stress testing

# --- Cognitive state discretization thresholds ---
# Trend score bins (from -1 to +1): dead down, somewhat down, neutral, somewhat up, strong up
TREND_SCORE_BINS = [-0.5, -0.15, 0.15, 0.5]  # 5 bins: very bearish to very bullish

# Bollinger stretch bins (from -3 to +3 standard deviations)
STRETCH_BINS = [-1.5, -0.5, 0.5, 1.5]  # 5 bins: very oversold to very overbought

# --- Logging ---
LOG_LEVEL = "INFO"          # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_FILE = True          # Enable file logging to logs/ directory

# --- Correlated symbol groups ---
# Symbols in the same group count as 1 position for risk management
CORRELATED_GROUPS = {
    "crypto": set(CRYPTO_SYMBOLS),  # All crypto moves together
    "nasdaq": set(["TSLA", "NVDA", "AMD", "AAPL", "META"]),
    "etf": set(["SPY", "QQQ"]),
}

# Maximum positions per correlated group
MAX_POSITIONS_PER_GROUP = 2

# --- Loop ---
SCAN_INTERVAL_SECONDS = 60   # how often to re-scan the market
STALE_DATA_MINUTES = 4       # skip a symbol if its latest candle is older than this (e.g. market closed)
