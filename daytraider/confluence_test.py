"""Does CONFLUENCE (multiple setups agreeing at once) predict a better
trade? Tests whether trades where N active setups fire together have a
higher out-of-sample edge than single-setup trades — on BOTH unseen sets.

If higher confluence -> higher edge robustly, we can prioritize or size up
multi-setup entries. Ungated (raw), net of cost, so the signal is clean.

Usage: python confluence_test.py
"""
from __future__ import annotations

import pandas as pd

import config
from data import compute_features, download_bars
from diagnose import OOS_UNIVERSE, OOS2_UNIVERSE
from strategies import LONG_STRATEGIES, SHORT_STRATEGIES

COST = 5 / 1e4
LONGS = [s for s in LONG_STRATEGIES if s in config.ACTIVE_STRATEGIES]
SHORTS = [s for s in SHORT_STRATEGIES if s in config.ACTIVE_STRATEGIES]


def collect(universe):
    rows = []
    for t in universe:
        try:
            f = compute_features(download_bars(t))
        except Exception as e:
            print(f"  skip {t}: {e}"); continue
        long_fire = sum(f[f"setup_{s}"] for s in LONGS)     # count per bar
        short_fire = sum(f[f"setup_{s}"] for s in SHORTS)
        for side, cnt, rcol in [("long", long_fire, "trail_ret"),
                                ("short", short_fire, "trail_ret_short")]:
            m = (cnt > 0) & f[rcol].notna()
            for c, r in zip(cnt[m].astype(int), f.loc[m, rcol]):
                rows.append((min(c, 4), r - COST))          # cap conf at 4+
    return pd.DataFrame(rows, columns=["conf", "net"])


def main():
    for label, uni in [("OOS set #1", OOS_UNIVERSE),
                       ("OOS set #2", OOS2_UNIVERSE)]:
        print(f"\n=== {label} ({len(uni)} tickers) ===")
        d = collect(uni)
        print(f"  {'setups firing':14s} {'trades':>8s} {'avg bps':>9s} "
              f"{'hit':>6s}")
        for c in sorted(d["conf"].unique()):
            r = d.loc[d["conf"] == c, "net"]
            tag = f"{c}+" if c >= 4 else str(c)
            print(f"  {tag:14s} {len(r):8d} {r.mean()*1e4:+8.1f} "
                  f"{(r>0).mean():6.3f}")
        # single vs multi summary
        single = d.loc[d["conf"] == 1, "net"]
        multi = d.loc[d["conf"] >= 2, "net"]
        print(f"  --> single-setup {single.mean()*1e4:+.1f} bps  vs  "
              f"multi-setup {multi.mean()*1e4:+.1f} bps")


if __name__ == "__main__":
    main()
