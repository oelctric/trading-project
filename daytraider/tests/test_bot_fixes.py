"""Offline tests for the live-bot fixes (no broker calls - fakes only).

    python tests/test_bot_fixes.py
"""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

import botstate  # noqa: E402
import config  # noqa: E402


def _tmp_state(monkey):
    pt = sys.modules.get("pairs_trade")
    if pt is not None:
        pt._orphans_mem.clear()            # module-level memory must not leak between tests
    d = Path(tempfile.mkdtemp())
    for name in ("PAIRS_STATE", "STOCK_TRAIL", "CRYPTO_TRAIL", "PAIRS_ORPHANS"):
        monkey.append((botstate, name, getattr(botstate, name)))
        setattr(botstate, name, d / f"{name.lower()}.json")


def _restore(monkey):
    for obj, name, val in monkey:
        setattr(obj, name, val)


def test_partial_bar_dropped():
    import data
    now = pd.Timestamp.now(tz="UTC").floor("h") + pd.Timedelta(minutes=18)
    idx = pd.MultiIndex.from_product([["SPY"], [now - pd.Timedelta(hours=k) for k in (2, 1, 0)]])
    raw = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                        "volume": 1.0, "vwap": 1.0}, index=idx)
    raw.index = idx.set_levels(idx.levels[1].floor("h"), level=1)

    class FakeClient:
        def get_stock_bars(self, req):
            return NS(df=raw)
    orig = data._client
    data._client = lambda: FakeClient()
    try:
        out = data.download_bars("SPY", days=1)
    finally:
        data._client = orig
    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)
    bar_len = pd.Timedelta(minutes=config.BAR_MINUTES)
    assert all(ts.tz_convert("UTC") + bar_len <= end for ts in out.index), out.index


def test_stop_found_inside_bracket_legs():
    import trade
    leg = NS(symbol="AAPL", type=NS(value="stop"), status=NS(value="new"), stop_price=90.0)
    tp = NS(symbol="AAPL", type=NS(value="limit"), status=NS(value="new"))
    parent = NS(symbol="AAPL", type=NS(value="limit"), status=NS(value="filled"), legs=[tp, leg])
    assert trade._stop_orders([parent])["AAPL"] is leg
    flat = NS(symbol="MSFT", type=NS(value="stop"), status=NS(value="new"), legs=None)
    assert trade._stop_orders([flat])["MSFT"] is flat


def test_restart_does_not_collapse_trail():
    """Stop already trailed to entry; no saved state -> must NOT jump to price."""
    import trade
    monkey = []
    _tmp_state(monkey)
    try:
        trade._trail.clear()
        replaced = []

        class C:
            def get_orders(self, req):
                return [NS(symbol="AAPL", id="o1", type=NS(value="stop"),
                           status=NS(value="new"), stop_price=100.0, legs=None)]

            def replace_order_by_id(self, oid, req):
                replaced.append(req.stop_price)
        pos = NS(symbol="AAPL", avg_entry_price="100.0", current_price="110.0", qty="10")
        trade.trail_stops(C(), [pos])
        assert not replaced or replaced[-1] < 109.0, replaced   # old code: 110.0 -> instant exit
        assert trade._trail["AAPL"]["dist"] >= 9.99
        assert botstate.load(botstate.STOCK_TRAIL, {})["AAPL"]   # persisted
    finally:
        trade._trail.clear()
        _restore(monkey)


def test_flatten_skips_pairs_and_crypto():
    import trade
    monkey = []
    _tmp_state(monkey)
    try:
        botstate.save(botstate.PAIRS_STATE, {"KO|PEP": {"beta": 1.0}})
        closed, cancelled = [], []

        class C:
            def get_orders(self, req):
                return [NS(symbol=s, id=s) for s in ("AAPL", "KO", "BTC/USD")]

            def cancel_order_by_id(self, oid):
                cancelled.append(oid)

            def get_all_positions(self):
                return [NS(symbol=s) for s in ("AAPL", "KO", "PEP", "BTCUSD")]

            def close_position(self, s):
                closed.append(s)
        trade.time.sleep = lambda s: None
        trade.flatten_all(C())
        assert closed == ["AAPL"] and cancelled == ["AAPL"], (closed, cancelled)
    finally:
        _restore(monkey)


def test_crypto_price_rounding():
    import crypto_trade as ct
    assert abs(ct.price_round(0.082345) - 0.082345) / 0.082345 < 1e-4
    assert ct.price_round(0.0004567) > 0
    assert ct.price_round(64123.456) == 64123.46


def test_robinhood_keygen_refuses_overwrite_and_quantizes():
    import robinhood
    d = Path(tempfile.mkdtemp())
    (d / ".env").write_text("ROBINHOOD_API_KEY=x\nROBINHOOD_PRIVATE_KEY=abc\n")
    orig = config.PROJECT_ROOT
    config.PROJECT_ROOT = d
    try:
        try:
            robinhood.keygen()
            raise AssertionError("should refuse")
        except SystemExit:
            pass
        assert "ROBINHOOD_PRIVATE_KEY=abc" in (d / ".env").read_text()
    finally:
        config.PROJECT_ROOT = orig
    rc = robinhood.RobinhoodCrypto.__new__(robinhood.RobinhoodCrypto)
    rc.trading_pairs = lambda s: {"results": [{"asset_increment": "0.00001"}]}
    q = rc._quantize("BTC-USD", 0.0000123456)
    assert format(q, "f") == "0.00001", format(q, "f")       # plain decimal, rounded down
    rc.trading_pairs = lambda s: {"results": [{"asset_increment": "1"}]}
    assert rc._quantize("DOGE-USD", 99.9) == 99


class _Broker:
    """Fake broker: orders have ids, fill/status settable per order."""
    def __init__(self):
        self.orders, self.calls, self.n = {}, [], 0

    def add(self, oid, filled, status):
        self.orders[oid] = NS(id=oid, filled_qty=str(filled), status=NS(value=status))

    def cancel_order_by_id(self, oid):
        self.calls.append(("cancel", oid))

    def get_order_by_id(self, oid):
        return self.orders[oid]

    def submit_order(self, req):
        self.n += 1
        oid = f"r{self.n}"
        self.calls.append(("submit", req.symbol, float(req.qty), req.side))
        self.add(oid, 0, "accepted")                  # accepted, NOT filled yet
        return NS(id=oid)

    positions = None   # symbol -> signed qty; None = don't model positions

    def get_open_position(self, s):
        if self.positions is None:
            raise RuntimeError("not modelled")
        if s not in self.positions:
            raise RuntimeError("position does not exist")
        return NS(qty=str(self.positions[s]))

    def close_position(self, s):
        raise AssertionError("must not close_position (would hit other shares)")


def test_pairs_unwind_by_order():
    import pairs_trade
    from alpaca.trading.enums import OrderSide
    monkey = []
    _tmp_state(monkey)
    pairs_trade.time.sleep = lambda s: None
    try:
        br = _Broker()
        br.add("o1", 0, "canceled")                   # cancelled before any fill
        pairs_trade.unwind_leg(br, "o1", "KO", OrderSide.SELL)
        assert br.calls == [("cancel", "o1")] and botstate.load(botstate.PAIRS_ORPHANS, {}) == {}
        br = _Broker()
        br.add("o2", 30, "canceled")                  # 30 of 50 short shares filled
        pairs_trade.unwind_leg(br, "o2", "KO", OrderSide.SELL)
        assert br.calls[:2] == [("cancel", "o2"), ("submit", "KO", 30.0, OrderSide.BUY)], br.calls
        assert len([c for c in br.calls if c[0] == "submit"]) == 1          # no duplicate reverses
        assert "o2" in botstate.load(botstate.PAIRS_ORPHANS, {})             # reverse not filled yet
        br.orders["r1"] = NS(id="r1", filled_qty="0", status=NS(value="rejected"))
        pairs_trade.resolve_orphans(br)                                      # reverse was rejected
        assert br.calls[-1] == ("submit", "KO", 30.0, OrderSide.BUY)         # -> re-issued
        br.orders["r2"] = NS(id="r2", filled_qty="30", status=NS(value="filled"))
        pairs_trade.resolve_orphans(br)
        assert botstate.load(botstate.PAIRS_ORPHANS, {}) == {}               # only now resolved
    finally:
        _restore(monkey)


def test_failed_entry_leg_is_unwound():
    import pairs_trade
    from alpaca.trading.enums import OrderSide
    monkey = []
    _tmp_state(monkey)
    pairs_trade.time.sleep = lambda s: None
    try:
        br = _Broker()
        br.add("ea", 100, "filled")                   # leg A filled
        br.add("eb", 0, "rejected")                   # leg B (short) rejected after acceptance
        pairs = {"KO|PEP": {"beta": 1.0, "qa": 100, "qb": 100, "oa": "ea", "ob": "eb",
                            "sa": OrderSide.BUY.value, "sb": OrderSide.SELL.value,
                            "entry_pending": True}}
        pairs_trade.reconcile_entries(br, pairs)
        assert "KO|PEP" not in pairs
        assert ("submit", "KO", 100.0, OrderSide.SELL) in br.calls         # A reversed
        assert "KO" in botstate.pairs_leg_symbols()                         # still protected
    finally:
        _restore(monkey)


def test_unreadable_pairs_state_is_not_treated_as_empty():
    monkey = []
    _tmp_state(monkey)
    try:
        botstate.save(botstate.PAIRS_STATE, {"KO|PEP": {}})
        orig_read = Path.read_text
        botstate.time.sleep = lambda s: None
        Path.read_text = lambda self, *a, **k: (_ for _ in ()).throw(PermissionError("locked"))
        try:
            try:
                botstate.pairs_leg_symbols()
                raise AssertionError("expected StateUnavailable")
            except botstate.StateUnavailable:
                pass
            import trade
            assert trade.flatten_all(NS()) is False        # refuses to guess, closes nothing
        finally:
            Path.read_text = orig_read
    finally:
        _restore(monkey)


def test_last_bar_of_session_label_is_nan():
    import numpy as np
    from data import _trailing_returns
    close = np.array([100, 100, 100, 150, 150, 150.])
    rl, rs = _trailing_returns(close + .1, close - .1, close, np.full(6, .01), 3, 1., 1., 3.,
                               session=np.array([1, 1, 1, 2, 2, 2]))
    assert np.isnan(rl[2]) and np.isnan(rs[2])   # 15:00-style last bar: no trade, not a fake 0


def test_weekly_retrain_restores_only_this_runs_backup():
    import shutil as _sh
    import types
    import weekly_retrain as wr
    d = Path(tempfile.mkdtemp())
    orig = (config.MODEL_DIR, config.SCALER_PATH, config.THRESHOLD_PATH, config.PROJECT_ROOT, wr.LOG)
    try:
        config.MODEL_DIR, config.SCALER_PATH, config.THRESHOLD_PATH = d, d / "scaler.json", d / "threshold.json"
        config.PROJECT_ROOT, wr.LOG = d, d / "log.txt"
        (d / "scaler.json").write_text("OLD")
        (d / "model_seed0.pt").write_text("OLD0")
        stale = d / "_pre_retrain_backup"
        stale.mkdir()
        (stale / "threshold.json").write_text("STALE")      # left over from an older run

        def failing_train(*a, **k):
            (d / "scaler.json").write_text("NEW")
            (d / "model_seed0.pt").write_text("NEW0")
            (d / "model_seed1.pt").write_text("NEW1")          # had no previous version
            return types.SimpleNamespace(returncode=1, stderr="boom", stdout="")
        orig_run = wr.subprocess.run
        wr.subprocess.run = failing_train
        try:
            wr.main()
        finally:
            wr.subprocess.run = orig_run
        assert (d / "scaler.json").read_text() == "OLD"
        assert (d / "model_seed0.pt").read_text() == "OLD0"
        assert not (d / "model_seed1.pt").exists()
        assert not (d / "threshold.json").exists()            # stale backup NOT restored
    finally:
        (config.MODEL_DIR, config.SCALER_PATH, config.THRESHOLD_PATH, config.PROJECT_ROOT, wr.LOG) = orig
        _sh.rmtree(d, ignore_errors=True)


def test_half_closed_pair_is_retried_every_scan():
    import numpy as np
    import pairs_trade
    monkey = []
    _tmp_state(monkey)
    try:
        fail = {"PEP": 1}          # PEP's close fails once

        class C:
            def __init__(self):
                self.closed = []

            def close_position(self, s):
                if fail.get(s):
                    fail[s] -= 1
                    raise RuntimeError("rate limited")
                self.closed.append(s)
        # converged spread: a - 1*b flat at the end -> |z| < EXIT_Z
        idx = pd.RangeIndex(60)
        a = pd.Series(100 + np.sin(np.arange(60)), index=idx)
        a.iloc[-1] = a.iloc[:-1].mean() + 0.0
        prices = pd.DataFrame({"KO": a, "PEP": pd.Series(0.0, index=idx)})
        pairs = {"KO|PEP": {"beta": 1.0}}
        c = C()
        pairs_trade.manage_exits(c, pairs, prices, {"KO", "PEP"})
        assert pairs["KO|PEP"].get("closing") is True and c.closed == ["KO"]
        # next scan: z now sits BETWEEN the thresholds - must still retry PEP
        b2 = prices.copy()
        b2.loc[59, "KO"] = a.iloc[:-1].mean() + 1.0 * a.iloc[:-1].std()
        pairs_trade.manage_exits(c, pairs, b2, {"PEP"})
        assert c.closed == ["KO", "PEP"]
        assert pairs["KO|PEP"]["closing"] is True        # accepted != filled: still tracked
        pairs_trade.manage_exits(c, pairs, b2, set())       # both legs confirmed flat
        assert "KO|PEP" not in pairs
        assert botstate.load(botstate.PAIRS_STATE, None) == {}
    finally:
        _restore(monkey)


def test_botstate_save_survives_file_locks():
    d = Path(tempfile.mkdtemp())
    target = d / "x.json"
    orig = Path.replace
    calls = {"n": 0}

    def flaky(self, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("locked by OneDrive")
        return orig(self, dst)
    botstate.time.sleep = lambda s: None
    Path.replace = flaky
    try:
        assert botstate.save(target, {"a": 1}) is True
    finally:
        Path.replace = orig
    assert botstate.load(target, None) == {"a": 1}
    Path.replace = lambda self, dst: (_ for _ in ()).throw(PermissionError("always locked"))
    try:
        assert botstate.save(target, {"a": 2}) is False      # gives up quietly, never raises
    finally:
        Path.replace = orig


def test_lobsim_imbalance_tilts_the_walk():
    import numpy as np
    import lobsim
    rng = np.random.default_rng(0)
    res = {}
    for lab, bt, at in (("bal", 40, 40), ("thin_ask", 40, 20), ("thin_bid", 20, 40)):
        t = lobsim.build_book(99.95, 100.05, 0.1, bt, at, rng)
        paths = lobsim.batch(t, 4.0, 1500, rng)
        res[lab] = lobsim.p_barrier_first(paths, 100, 101.5, 99.0)
    assert res["thin_ask"] > res["bal"] + 0.05 > res["thin_bid"] + 0.10, res   # a 2:1 book must tilt
    # a balanced REAL touch must not be tilted by synthetic depth noise
    for seed in range(5):
        r2 = np.random.default_rng(seed)
        t = lobsim.build_book(99.95, 100.05, 0.1, 40, 40, r2)
        assert t[0][1][0] == 40 and t[1][1][0] == 40
        p = lobsim.p_barrier_first(lobsim.batch(t, 4.0, 1500, r2), 100, 101.0, 99.0)
        assert 0.35 < p < 0.65, (seed, p)   # symmetric barriers -> ~50%


def test_pair_with_one_leg_still_filling_is_kept():
    import pairs_trade
    from alpaca.trading.enums import OrderSide
    monkey = []
    _tmp_state(monkey)
    try:
        br = _Broker()
        br.add("ea", 100, "filled")
        br.add("eb", 40, "partially_filled")           # still working - NOT a failure
        pairs = {"KO|PEP": {"beta": 1.0, "qa": 100, "qb": 100, "oa": "ea", "ob": "eb",
                            "sa": OrderSide.BUY.value, "sb": OrderSide.SELL.value,
                            "entry_pending": True}}
        pairs_trade.reconcile_entries(br, pairs)
        assert "KO|PEP" in pairs and pairs["KO|PEP"]["entry_pending"] and br.calls == []
        br.add("eb", 100, "filled")
        pairs_trade.reconcile_entries(br, pairs)
        assert pairs["KO|PEP"]["entry_pending"] is False
    finally:
        _restore(monkey)


def test_orphan_reverse_never_exceeds_held_position():
    import pairs_trade
    from alpaca.trading.enums import OrderSide
    monkey = []
    _tmp_state(monkey)
    pairs_trade.time.sleep = lambda s: None
    try:
        br = _Broker()
        br.positions = {}                              # the KO shares were already closed by hand
        br.add("o5", 100, "filled")
        pairs_trade.unwind_leg(br, "o5", "KO", OrderSide.BUY)
        assert not [c for c in br.calls if c[0] == "submit"], br.calls   # no new short opened
        assert botstate.load(botstate.PAIRS_ORPHANS, {}) == {}
        br = _Broker()
        br.positions = {"KO": 30}                      # only 30 of 100 still held
        br.add("o6", 100, "filled")
        pairs_trade.unwind_leg(br, "o6", "KO", OrderSide.BUY)
        assert ("submit", "KO", 30.0, OrderSide.SELL) in br.calls, br.calls
    finally:
        _restore(monkey)


def test_stop_leg_selection_prefers_live_legs():
    import trade
    dead = NS(symbol="AAPL", type=NS(value="stop"), status=NS(value="replaced"))
    held = NS(symbol="AAPL", type=NS(value="stop"), status=NS(value="held"))
    live = NS(symbol="AAPL", type=NS(value="stop"), status=NS(value="new"))
    parent = NS(symbol="AAPL", type=NS(value="limit"), status=NS(value="filled"), legs=[live, held, dead])
    assert trade._stop_orders([parent])["AAPL"] is live
    only_dead = NS(symbol="MSFT", type=NS(value="stop"), status=NS(value="replaced"), legs=None)
    assert "MSFT" not in trade._stop_orders([only_dead])


def test_failed_entry_waits_for_open_and_unwind_is_idempotent():
    import pairs_trade
    from alpaca.trading.enums import OrderSide
    monkey = []
    _tmp_state(monkey)
    pairs_trade.time.sleep = lambda s: None
    try:
        br = _Broker()
        br.add("ea", 100, "filled")
        br.add("eb", 0, "expired")                    # leg B's DAY order expired at the close
        mk = lambda: {"KO|PEP": {"beta": 1.0, "qa": 100, "qb": 100, "oa": "ea", "ob": "eb",  # noqa: E731
                                 "sa": OrderSide.BUY.value, "sb": OrderSide.SELL.value,
                                 "entry_pending": True}}
        pairs = mk()
        pairs_trade.reconcile_entries(br, pairs, can_trade=False)   # after hours
        assert br.calls == [] and "KO|PEP" in pairs                  # no orders while closed
        pairs_trade.reconcile_entries(br, pairs, can_trade=True)    # at the open
        assert "KO|PEP" not in pairs
        n_rev = len([c for c in br.calls if c[0] == "submit"])
        assert n_rev == 1, br.calls
        # a repeated unwind of the same leg (e.g. an interrupted scan) must NOT
        # forget the reverse order already working and send a second one
        pairs_trade.unwind_leg(br, "ea", "KO", OrderSide.BUY)
        assert len([c for c in br.calls if c[0] == "submit"]) == 1, br.calls
    finally:
        _restore(monkey)


def test_orphan_tracked_in_memory_when_file_cannot_be_saved():
    import pairs_trade
    from alpaca.trading.enums import OrderSide
    monkey = []
    _tmp_state(monkey)
    pairs_trade.time.sleep = lambda s: None
    orig_save = botstate.save
    try:
        pairs_trade._orphans_mem.clear()
        botstate.save = lambda path, data: False if path == botstate.PAIRS_ORPHANS else orig_save(path, data)
        br = _Broker()
        br.add("ea", 100, "filled")
        br.add("eb", 0, "rejected")
        pairs = {"KO|PEP": {"beta": 1.0, "qa": 100, "qb": 100, "oa": "ea", "ob": "eb",
                            "sa": OrderSide.BUY.value, "sb": OrderSide.SELL.value,
                            "entry_pending": True}}
        pairs_trade.reconcile_entries(br, pairs)
        assert "KO|PEP" in pairs                       # kept until the orphan record is saved
        assert "ea" in pairs_trade._orphans_mem        # but tracked (and reversed) in memory
        assert ("submit", "KO", 100.0, OrderSide.SELL) in br.calls
        botstate.save = orig_save                      # file writable again
        pairs_trade.reconcile_entries(br, pairs)       # idempotent: no duplicate reverse
        assert "KO|PEP" not in pairs
        assert len([c for c in br.calls if c[0] == "submit"]) == 1, br.calls
    finally:
        botstate.save = orig_save
        pairs_trade._orphans_mem.clear()
        _restore(monkey)


def test_flatten_fallback_closes_only_own_tracked_positions():
    import trade
    monkey = []
    _tmp_state(monkey)
    orig_legs = botstate.pairs_leg_symbols
    try:
        botstate.pairs_leg_symbols = lambda: (_ for _ in ()).throw(botstate.StateUnavailable("corrupt"))
        trade._trail.clear()
        trade._trail["AAPL"] = {"dist": 1.0, "hwm": 100.0}       # the stock bot's own trade
        closed = []

        class C:
            def get_orders(self, req):
                return []

            def get_all_positions(self):
                return [NS(symbol=s) for s in ("AAPL", "KO", "BTCUSD")]

            def close_position(self, s):
                closed.append(s)
        trade.time.sleep = lambda s: None
        assert trade.flatten_all(C()) is False and closed == []   # first: refuse to guess
        assert trade.flatten_all(C(), fallback=True) is True
        assert closed == ["AAPL"], closed                          # never KO (maybe a pair leg)
    finally:
        botstate.pairs_leg_symbols = orig_legs
        trade._trail.clear()
        _restore(monkey)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)


def test_gate_trades_end_at_session_close():
    import numpy as np
    from data import _trailing_returns
    close = np.array([100, 100, 100, 150, 150, 150.])
    rl, _ = _trailing_returns(close + .1, close - .1, close, np.full(6, .01), 3, 1., 1., 3.,
                              session=np.array([1, 1, 1, 2, 2, 2]))
    assert rl[1] == 0.0          # old code rode the overnight gap to +3%


def test_manipulation_flags_are_causal_and_use_pdh():
    import numpy as np
    from manipulation import bar_flags
    rng = np.random.default_rng(1)
    n = 400
    c = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    h, l = c + rng.random(n), c - rng.random(n)
    o = c.shift(1).fillna(100)
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "vol_z": rng.normal(0, 1, n),
                       "atr_pct": 0.01, "pdh_dist": c / c.rolling(7).max().shift(7).bfill() - 1,
                       "pdl_dist": c / c.rolling(7).min().shift(7).bfill() - 1,
                       "upper_wick": (h - np.maximum(o, c)) / c, "lower_wick": (np.minimum(o, c) - l) / c})
    full, part = bar_flags(df), bar_flags(df.iloc[:300])
    assert int((full.iloc[:300] - part).abs().values.max()) == 0     # adding future bars changes nothing
    # stop-hunt above yesterday's high (pdh=110): poke to 111, close back at 109 on a long wick
    one = pd.DataFrame({"open": [108.0, 108.5], "high": [109.0, 111.0], "low": [107.5, 108.4],
                        "close": [108.5, 109.0], "vol_z": [0.0, 0.0], "atr_pct": [0.01, 0.01],
                        "pdh_dist": [108.5 / 110 - 1, 109 / 110 - 1], "pdl_dist": [0.05, 0.05]})
    one["upper_wick"] = (one["high"] - one[["open", "close"]].max(axis=1)) / one["close"]
    one["lower_wick"] = (one[["open", "close"]].min(axis=1) - one["low"]) / one["close"]
    assert bar_flags(one)["stophunt_high"].iloc[-1] == 1


if __name__ == "__main__":
    test_gate_trades_end_at_session_close()
    test_manipulation_flags_are_causal_and_use_pdh()
    print("ok  test_manipulation_flags_are_causal_and_use_pdh")
    print("ok  test_gate_trades_end_at_session_close")
