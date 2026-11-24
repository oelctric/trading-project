"""Is the fake-breakout filter worth it? (causal detector, same-day trailing exits)

    python measure_fake_breakout.py

For every breakout-family setup that fired in the last 5 years, compares the
trailing-exit return (after COST_PER_TRADE) when the bar is flagged as a fake
breakout vs when it isn't. The live bot skips flagged ones when
config.FAKE_BREAKOUT_FILTER is True.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from data import compute_features, download_bars
from manipulation import bar_flags
from strategies import is_short

rows = []
for t in config.TICKERS:
    try:
        f = compute_features(download_bars(t))
    except Exception as e:
        print(f"  {t}: {e}")
        continue
    fake = bar_flags(f)["fake_breakout"].astype(bool)
    for s in config.BREAKOUT_FAMILY:
        col = f"setup_{s}"
        if col not in f:
            continue
        rcol = "trail_ret_short" if is_short(s) else "trail_ret"
        fired = (f[col] > 0) & f[rcol].notna()
        for ts in f.index[fired]:
            rows.append((t, s, ts, bool(fake.loc[ts]), float(f.at[ts, rcol]) - config.COST_PER_TRADE))
d = pd.DataFrame(rows, columns=["ticker", "setup", "ts", "fake", "ret"])
d["year"] = pd.to_datetime(d["ts"], utc=True).dt.year
print(f"{len(d):,} breakout-family signals, {d.fake.mean():.1%} flagged fake\n")
g = d.groupby("fake")["ret"].agg(["count", "mean"])
g["mean_bps"] = g["mean"] * 1e4
print(g[["count", "mean_bps"]].round(2).to_string())
diff = (d[d.fake].ret.mean() - d[~d.fake].ret.mean()) * 1e4
se = np.sqrt(d[d.fake].ret.var() / max(d.fake.sum(), 1) + d[~d.fake].ret.var() / max((~d.fake).sum(), 1)) * 1e4
print(f"\nflagged minus unflagged: {diff:+.2f} bps (t = {diff / se:+.1f})")
print("per year (bps, flagged vs unflagged):")
print((d.groupby(["year", "fake"]).ret.mean().unstack() * 1e4).round(1).to_string())
