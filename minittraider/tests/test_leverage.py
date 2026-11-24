"""Leverage sizing safety + live/backtest sizing parity.

    python tests/test_leverage.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config  # noqa: E402
import portfolio  # noqa: E402
from leverage import LeverageConfig, size_position  # noqa: E402


def fresh_portfolio(cash=100.0):
    d = tempfile.mkdtemp()
    portfolio.STATE_FILE = os.path.join(d, "state.json")
    portfolio.TRADES_FILE = os.path.join(d, "trades.csv")
    config.STARTING_CASH = cash
    return portfolio.Portfolio()


def test_liquidation_is_beyond_stop():
    cfg = config.LEVERAGE
    for stop_pct in (0.02, 0.05, 0.1, 0.2):
        sz = size_position(cfg, 100, 100, [], stop_pct, 0, 0, 100)
        liq_dist = 1 / sz["leverage"] - cfg.maintenance_margin
        assert liq_dist >= cfg.liq_buffer * stop_pct - 1e-9 or sz["leverage"] == 1.0, stop_pct


def test_risk_is_what_config_says():
    sz = size_position(config.LEVERAGE, 100, 100, [], 0.10, 0, 0, 100)
    assert abs(sz["risk"] - 100 * config.LEVERAGE.base_risk) < 1e-9


def test_cold_streak_halves_risk():
    cfg = LeverageConfig(base_risk=0.03, cold_factor=0.5, cold_lookback=10)
    hot = size_position(cfg, 100, 100, [1.0] * 10, 0.10, 0, 0, 100)
    cold = size_position(cfg, 100, 100, [-1.0] * 10, 0.10, 0, 0, 100)
    assert abs(cold["risk"] - hot["risk"] / 2) < 1e-9


def test_total_leverage_capped():
    cfg = LeverageConfig(base_risk=0.5, max_leverage=2.0)
    sz = size_position(cfg, 100, 100, [], 0.01, 0, 150, 100)
    assert sz["notional"] <= 50 + 1e-9


def test_live_portfolio_sizes_like_backtest():
    p = fresh_portfolio(100.0)
    pos = p.open("BTC-USD", "long", 50_000.0, 1000.0, 0.7, "t", sl=45_000.0,
                 exit_state={"stop": 45_000.0})
    expect = size_position(config.LEVERAGE, 100, 100, [], 0.10, 0, 0, 100)
    assert pos is not None
    assert abs(pos.qty * pos.entry - expect["notional"]) < 1e-6
    assert abs(pos.margin - expect["margin"]) < 1e-6
    assert abs(p.cash - (100 - expect["margin"])) < 1e-6


def test_loss_capped_at_margin_and_brake_learns():
    p = fresh_portfolio(100.0)
    p.open("ETH-USD", "long", 1000.0, 50.0, 0.7, "t", sl=900.0, exit_state={"stop": 900.0})
    margin = p.positions["ETH-USD"].margin
    pnl = p.close("ETH-USD", 1.0, "crash")      # price collapses far past the stop
    assert pnl >= -margin - 1e-9
    assert len(p.recent_r) == 1 and p.recent_r[0] < 0
    assert p.equity({}) >= 0


def test_non_swing_positions_get_no_leverage():
    p = fresh_portfolio(100.0)
    pos = p.open("BTC-USD", "long", 100.0, 2.0, 0.7, "minute scalp", sl=98.0)   # no exit_state
    assert pos is not None
    assert pos.leverage == 1.0 and pos.fee_rate == 0.0      # spot pricing, no futures funding


def test_trade_log_migration_never_mixes_headers():
    import csv
    p = fresh_portfolio(100.0)
    d = os.path.dirname(portfolio.TRADES_FILE)
    for _ in range(2):   # two generations of old-format logs
        with open(portfolio.TRADES_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["closed_at", "symbol", "old", "format"])
        p.open("ETH-USD", "long", 100.0, 2.0, 0.7, "t", sl=95.0, exit_state={"stop": 95.0})
        p.close("ETH-USD", 101.0, "test")
        with open(portfolio.TRADES_FILE, newline="") as f:
            assert next(csv.reader(f)) == portfolio.Portfolio.TRADE_COLUMNS
    olds = [x for x in os.listdir(d) if x.startswith("trades_old_format")]
    assert len(olds) == 2, olds


def test_group_limit_depends_on_strategy():
    orig = config.STRATEGY
    try:
        for strat, expect in ((config.Strategy.COGNITIVE, 2), (config.Strategy.SWING, 6)):
            config.STRATEGY = strat
            p = fresh_portfolio(10_000.0)
            opened = 0
            for s in config.CRYPTO_SYMBOLS:
                if p.can_open(s):
                    p.positions[s] = portfolio.Position(s, "long", 1, 1, 0, 0, "t", 0.5, "t")
                    opened += 1
            assert opened == min(expect, config.MAX_OPEN_POSITIONS), (strat, opened)
    finally:
        config.STRATEGY = orig


def test_unlevered_short_loss_not_capped_and_minute_trades_not_shrunk():
    orig = config.USE_LEVERAGE
    try:
        config.USE_LEVERAGE = False
        p = fresh_portfolio(1000.0)
        pos = p.open("ETH-USD", "short", 100.0, 2.0, 0.7, "t", sl=110.0, exit_state={"stop": 110.0})
        posted = pos.posted()
        pnl = p.close("ETH-USD", 300.0, "squeeze")          # price triples against the short
        assert pnl < -posted                                  # real spot loss, not capped at margin
    finally:
        config.USE_LEVERAGE = orig
    p = fresh_portfolio(100.0)
    p.cash = 3.0                                              # too little cash for the sized trade
    assert p.open("BTC-USD", "long", 100.0, 2.0, 0.7, "minute", sl=98.0) is None


def test_minute_trades_do_not_feed_swing_cold_brake():
    p = fresh_portfolio(1000.0)
    p.open("BTC-USD", "long", 100.0, 2.0, 0.7, "minute", sl=98.0)     # no exit_state
    p.close("BTC-USD", 90.0, "stop")
    assert p.recent_r == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
