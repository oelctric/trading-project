"""Train the DQN scalper on real 1-minute bars and report the learned policy.

The honest experiment: with an 8 bps friction tax and passive-fill adverse
selection, does the agent discover positive-EV trades, or learn to Hold?
Reports the greedy action distribution and the average realized reward of the
trades it chooses vs. the always-Hold baseline (0).

Usage: python -m hft_rl.train [months] [episodes]
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("hft_rl", 1)[0])
from alpaca.data.requests import StockBarsRequest          # noqa: E402
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # noqa: E402

from data import _client                                    # noqa: E402
from hft_rl.context import ContextEngine                    # noqa: E402
from hft_rl.dqn import DQNAgent                              # noqa: E402
from hft_rl.env import STATE_DIM, N_ACTIONS, ScalperEnv, build_frame  # noqa: E402

TICKERS = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"]


def minute_bars(t, days):
    end = datetime.now(timezone.utc) - timedelta(minutes=16)
    req = StockBarsRequest(symbol_or_symbols=t,
                           timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                           start=end - timedelta(days=days), end=end,
                           adjustment="all")
    df = _client().get_stock_bars(req).df.droplevel(0)
    df.index = df.index.tz_convert("America/New_York")
    return df.between_time("09:30", "16:00")


def main():
    months = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    ctx = ContextEngine()
    print(f"Context engine live-LLM: {ctx._live} "
          f"(mock context in use if False)\n")

    # OUT-OF-SAMPLE split: train on the first 60% of each ticker's history,
    # evaluate the greedy policy ONLY on the held-out last 40%.
    train_frames, test_frames = [], []
    for t in TICKERS:
        try:
            fr = build_frame(minute_bars(t, months * 31))
            k = int(len(fr) * 0.6)
            train_frames.append(fr.iloc[:k])
            test_frames.append(fr.iloc[k:])
        except Exception as e:
            print(f"  skip {t}: {e}")
    print(f"Loaded {len(train_frames)} tickers "
          f"(train {sum(len(f) for f in train_frames)} / "
          f"test {sum(len(f) for f in test_frames)} bars)")

    agent = DQNAgent(STATE_DIM, N_ACTIONS)
    step = 0
    for ep in range(episodes):
        env = ScalperEnv(train_frames[ep % len(train_frames)], ctx=ctx, seed=ep)
        s = env.reset()
        done = False
        while not done:
            a = agent.act(s)
            s2, r, done, info = env.step(a)
            agent.replay.push(s, a, r, s2, float(done))
            agent.learn()
            s = s2
            step += 1
            if step % 1000 == 0:
                agent.sync_target()
        agent.eps = max(0.05, agent.eps * 0.92)   # anneal exploration

    # Evaluate the GREEDY policy on HELD-OUT test data only
    agent.eps = 0.0
    actions, rewards_when_traded = [], []
    for fr in test_frames:
        env = ScalperEnv(fr, ctx=ctx, seed=999)
        s = env.reset()
        done = False
        while not done:
            a = agent.act(s)
            s, r, done, info = env.step(a)
            actions.append(a)
            if info.get("traded"):
                rewards_when_traded.append(info["realized_bps"])
    actions = np.array(actions)
    dist = {i: float((actions == i).mean()) for i in range(N_ACTIONS)}
    print("\n=== Learned greedy policy (OUT-OF-SAMPLE test data) ===")
    print(f"  action distribution  Hold={dist[0]:.1%}  "
          f"Buy={dist[1]:.1%}  Sell={dist[2]:.1%}")
    if rewards_when_traded:
        rt = np.array(rewards_when_traded)
        print(f"  trades taken: {len(rt)}  "
              f"avg realized (pre-friction) {rt.mean():+.1f} bps  "
              f"net of 8bps friction {rt.mean()-8:+.1f} bps")
        print(f"  => trading EV vs always-Hold(0): "
              f"{'POSITIVE' if rt.mean()-8 > 0 else 'NEGATIVE'}")
    else:
        print("  agent learned to HOLD — took no trades under greedy policy")
        print("  => confirms: no positive-EV pocket survived 8bps friction")


if __name__ == "__main__":
    main()
