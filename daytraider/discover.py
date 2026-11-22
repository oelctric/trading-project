"""Machine strategy discovery with honest train/validate/test discipline.

Generates thousands of random candidate rules of the form
    (feature_a {>|<} threshold_a) AND (feature_b {>|<} threshold_b) -> long/short
scores them on the TRAIN period, keeps the best, then requires them to
stay profitable on the later VALIDATION period. Only rules that survive
both are reported, together with their final TEST-period result — which is
the only number that matters (train/val numbers are selection-biased by
construction).

Discovered rules are saved to artifacts/discovered.json for review. They
are deliberately NOT auto-wired into live trading: promote one by hand
(add it to strategies.py) only if its test row convinces you.

Usage: python discover.py [n_candidates]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import config
from data import FEAR_COLUMNS, PRO_COLUMNS, compute_features, download_bars

RULE_FEATURES = [
    "log_ret", "range_pct", "body_pct", "vol_z", "rsi", "macd_hist",
    "sma20_dist", "sma50_dist", "bb_pos", "atr_pct",
] + FEAR_COLUMNS + PRO_COLUMNS
MIN_TRADES = 500  # per period; rare rules are unmeasurable noise


def build_panel() -> pd.DataFrame:
    frames = []
    for ticker in config.TICKERS:
        try:
            f = compute_features(download_bars(ticker))
        except Exception as e:
            print(f"  skipping {ticker}: {e}")
            continue
        print(f"  {ticker}: {len(f)} bars")
        cols = RULE_FEATURES + ["trade_ret", "trade_ret_short"]
        frames.append(f[cols].dropna())
    return pd.concat(frames).sort_index()


def main():
    n_candidates = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    rng = np.random.default_rng(7)

    print("Building feature panel for all tickers...")
    panel = build_panel()
    n = len(panel)
    train = panel.iloc[: int(n * 0.6)]
    val = panel.iloc[int(n * 0.6): int(n * 0.8)]
    test = panel.iloc[int(n * 0.8):]
    print(f"\nPanel rows: train={len(train)}  val={len(val)}  test={len(test)}")

    # Thresholds drawn from TRAIN quantiles only (no peeking).
    qs = {f: train[f].quantile([0.2, 0.35, 0.5, 0.65, 0.8]).to_numpy()
          for f in RULE_FEATURES}

    def apply(rule, df):
        f1, op1, t1, f2, op2, t2, side = rule
        m = (df[f1] > t1) if op1 == ">" else (df[f1] < t1)
        m &= (df[f2] > t2) if op2 == ">" else (df[f2] < t2)
        rets = df["trade_ret_short"] if side == "short" else df["trade_ret"]
        r = rets[m] - config.COST_PER_TRADE
        return r

    def score(rule, df):
        r = apply(rule, df)
        if len(r) < MIN_TRADES:
            return None
        return {"trades": int(len(r)), "avg_bps": float(r.mean() * 1e4),
                "total": float(r.sum())}

    print(f"Scoring {n_candidates} random 2-condition rules on train...")
    scored = []
    for _ in range(n_candidates):
        f1, f2 = rng.choice(RULE_FEATURES, size=2, replace=False)
        rule = (f1, rng.choice([">", "<"]), float(rng.choice(qs[f1])),
                f2, rng.choice([">", "<"]), float(rng.choice(qs[f2])),
                rng.choice(["long", "short"]))
        s = score(rule, train)
        if s and s["avg_bps"] > 2.0:
            scored.append((rule, s))
    scored.sort(key=lambda x: -x[1]["avg_bps"])
    top = scored[:30]
    print(f"  {len(scored)} rules cleared +2 bps on train; "
          f"validating top {len(top)}...")

    survivors = []
    for rule, s_train in top:
        s_val = score(rule, val)
        if s_val and s_val["avg_bps"] > 1.0:
            s_test = score(rule, test)
            survivors.append({"rule": list(rule), "train": s_train,
                              "val": s_val, "test": s_test})

    print(f"\n{len(survivors)} rules survived validation. "
          f"TEST results (the only honest column):")
    for s in survivors:
        f1, op1, t1, f2, op2, t2, side = s["rule"]
        desc = f"{side.upper():5s} when {f1}{op1}{t1:.4f} & {f2}{op2}{t2:.4f}"
        t = s["test"]
        t_str = (f"test: {t['trades']} trades, {t['avg_bps']:+.1f} bps"
                 if t else "test: too few trades")
        print(f"  {desc}")
        print(f"      train {s['train']['avg_bps']:+.1f} bps | "
              f"val {s['val']['avg_bps']:+.1f} bps | {t_str}")

    out_path = config.MODEL_DIR / "discovered.json"
    out_path.write_text(json.dumps(survivors, indent=2))
    print(f"\nSaved to {out_path} -- promote a rule by hand only if its "
          f"TEST row convinces you.")


if __name__ == "__main__":
    main()
