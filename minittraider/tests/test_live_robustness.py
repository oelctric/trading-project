"""Live-loop robustness: failed downloads, restarts, late signals.

    python tests/test_live_robustness.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

import config  # noqa: E402
import history  # noqa: E402
import swing_live  # noqa: E402
from history import load  # noqa: E402


def _daily(n=500):
    return load("BTC-USD", "1d", 3650).iloc[-n:]


def test_failed_download_is_not_cached_and_falls_back():
    good = _daily()
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        raise RuntimeError("network down")

    orig = swing_live.fetch_coinbase
    try:
        swing_live._cache.clear()
        # first ever fetch fails -> error surfaces (nothing to fall back to)
        swing_live.fetch_coinbase = flaky
        try:
            swing_live.closed_bars("BTC-USD", True)
            raise AssertionError("expected an error")
        except RuntimeError:
            pass
        assert "BTC-USD" not in swing_live._cache          # failure NOT cached
        # last good bars from an older bar period, then the network dies
        swing_live._cache["BTC-USD"] = (0, good)
        out = swing_live.closed_bars("BTC-USD", True)
        assert out is good                                  # fell back to last good data
        assert swing_live._cache["BTC-USD"][0] == 0         # still marked stale -> retried
    finally:
        swing_live.fetch_coinbase = orig
        swing_live._cache.clear()


def test_partial_history_raises():
    orig = history._session.get

    class Boom:
        def __call__(self, *a, **k):
            raise history.requests.ConnectionError("down")

    history._session.get = Boom()
    orig_sleep = history.time.sleep
    history.time.sleep = lambda s: None
    try:
        try:
            history.fetch_coinbase("BTC-USD", "1d", 30)
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "failed after 5 attempts" in str(e)
    finally:
        history._session.get = orig
        history.time.sleep = orig_sleep


def test_acted_signal_survives_restart():
    d = tempfile.mkdtemp()
    orig = swing_live.SIGNALS_FILE
    try:
        swing_live.SIGNALS_FILE = os.path.join(d, "sig.json")
        bars = _daily()
        swing_live._acted_bar.pop("ETH-USD", None)
        swing_live.new_exit_state("ETH-USD", "long", 100.0, 5.0, bars)
        assert "ETH-USD" not in swing_live._acted_bar        # not traded yet -> not marked
        swing_live.mark_acted("ETH-USD", bars)                 # position opened
        # simulate a restart: reload the module-level memory from disk
        swing_live._acted_bar = swing_live._load_acted()
        assert swing_live._acted_bar["ETH-USD"] == str(bars.index[-1])
    finally:
        swing_live.SIGNALS_FILE = orig
        swing_live._acted_bar = swing_live._load_acted()


def test_late_signal_is_refused():
    df = load("BTC-USD", "1d", 3650)
    ind = swing_live.indicators(df, swing_live.PARAMS)
    # find a historical bar that produced a real entry signal
    k = next(i for i in range(len(df) - 1, 400, -1)
             if swing_live.entry_signal(df["close"].iloc[i], ind.iloc[i], swing_live.PARAMS,
                                        swing_live.round_trip_fee("BTC-USD"))[0] != "none")
    bars = df.iloc[:k + 1]
    swing_live._acted_bar.pop("BTC-USD", None)
    today = pd.Timestamp.now(tz="UTC").floor("D")
    fresh = bars.copy()                      # signal bar = yesterday -> closed < 1 day ago
    fresh.index = fresh.index + (today - pd.Timedelta(days=1) - bars.index[-1])
    age_ok = (pd.Timestamp.now(tz="UTC") - today).total_seconds() / 3600 <= config.SWING_MAX_SIGNAL_AGE_HOURS
    d_fresh = swing_live.decide_swing("BTC-USD", fresh)
    if age_ok:
        assert d_fresh.action != "none", d_fresh.reasons[-1]
    else:
        assert "too late" in d_fresh.reasons[-1], d_fresh.reasons[-1]
    old = bars.copy()                        # signal bar = 3 days ago -> always too late
    old.index = old.index + (today - pd.Timedelta(days=3) - bars.index[-1])
    d = swing_live.decide_swing("BTC-USD", old)
    assert d.action == "none" and "too late" in d.reasons[-1], d.reasons[-1]


def test_no_signal_says_so_not_too_late():
    df = load("BTC-USD", "1d", 3650)
    ind = swing_live.indicators(df, swing_live.PARAMS)
    k = next(i for i in range(len(df) - 1, 400, -1)
             if swing_live.entry_signal(df["close"].iloc[i], ind.iloc[i], swing_live.PARAMS,
                                        swing_live.round_trip_fee("BTC-USD"))[0] == "none")
    bars = df.iloc[:k + 1].copy()
    bars.index = bars.index + (pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=3) - bars.index[-1])
    d = swing_live.decide_swing("BTC-USD", bars)
    assert "too late" not in d.reasons[-1], d.reasons[-1]


def test_live_fee_matches_account():
    expect = config.FUTURES_FEE_PER_SIDE if config.USE_LEVERAGE else config.FEE_PER_SIDE["crypto"]
    assert swing_live.fee_per_side("BTC-USD") == expect
    import backtest_swing
    assert backtest_swing.FEE == expect


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
