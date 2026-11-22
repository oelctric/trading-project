"""ScalperEnv: a Gym-convention 1-minute scalping environment.

Design goals from the spec:
  * State  = 1-min features + Regime Context Vector + microstructure
             (order-flow imbalance proxy, high-frequency volatility).
  * Action = {0 Hold, 1 Buy, 2 Sell} (discrete).
  * Reward = the BRUTAL reward: 0 for Hold; for a trade, a flat 8 bps
             friction tax plus the realized outcome of a 15-min-ATR trailing
             exit, with adverse-selection PASSIVE FILL (a long fills only if
             the next bar's low trades STRICTLY below the limit).

Single position at a time: on a trade the env simulates the position forward
to its exit and returns the realized reward, then the agent is flat again.
This isolates the only question that matters — does any state carry positive
expectancy AFTER 8 bps friction — which is exactly what an RL policy will or
won't discover.

True L2 quote imbalance is unavailable on the free feed, so the imbalance
state uses a close-location-value order-flow proxy (flagged, not real depth).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .context import ContextEngine, RegimeContext

FRICTION = 8.0 / 1e4         # 8 bps per trade (spread + fees + slippage)
TRAIL_TF = 15                # trailing stop anchored to 15-min ATR
TRAIL_MULT = 1.0
HOLD_CAP = 60                # force exit after this many 1-min bars


def build_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """1-min feature frame: returns, HF vol, order-flow proxy, 15-min ATR."""
    o = bars.copy()
    c, h, l, v = o["close"], o["high"], o["low"], o["volume"]
    o["ret1"] = np.log(c / c.shift(1))
    o["hf_vol"] = o["ret1"].rolling(15).std()                 # HF volatility
    rng = (h - l).replace(0, np.nan)
    o["ofi"] = ((2 * c - h - l) / rng).fillna(0)              # order-flow proxy
    o["vol_z"] = (v - v.rolling(30).mean()) / v.rolling(30).std()
    o["mom5"] = np.log(c / c.shift(5))
    b15 = bars.resample("15min").agg(high=("high", "max"), low=("low", "min"),
                                     close=("close", "last")).dropna()
    tr = pd.concat([b15["high"] - b15["low"],
                    (b15["high"] - b15["close"].shift()).abs(),
                    (b15["low"] - b15["close"].shift()).abs()], axis=1).max(axis=1)
    atr = (tr.ewm(alpha=1/14, adjust=False).mean() / b15["close"])
    o["atr15"] = atr.reindex(o.index, method="ffill")
    return o.dropna()


FEATURES = ["ret1", "hf_vol", "ofi", "vol_z", "mom5", "atr15"]
STATE_DIM = len(FEATURES) + 3   # + regime context vector (3)
N_ACTIONS = 3


def _trailing_outcome(entry, atr, hi, lo, cl, short=False):
    """Realized fractional return of a 15-min-ATR trailing exit."""
    sgn = -1.0 if short else 1.0
    best = 0.0
    for k in range(min(HOLD_CAP, len(hi))):
        favhi = sgn * (hi[k] - entry) / entry if not short else (entry - lo[k]) / entry
        favlo = sgn * (lo[k] - entry) / entry if not short else (entry - hi[k]) / entry
        stop = max(-TRAIL_MULT * atr, best - TRAIL_MULT * atr)
        if favlo <= stop:
            return stop
        best = max(best, favhi)
    j = min(HOLD_CAP, len(cl)) - 1
    if j < 0:
        return 0.0
    return sgn * (cl[j] - entry) / entry if not short else (entry - cl[j]) / entry


class ScalperEnv:
    """Gym-convention env (reset/step). One decision per 1-min bar."""

    def __init__(self, frame: pd.DataFrame, ctx: ContextEngine | None = None,
                 seed: int = 0):
        self.f = frame.reset_index(drop=False)
        self.H = self.f["high"].to_numpy()
        self.L = self.f["low"].to_numpy()
        self.C = self.f["close"].to_numpy()
        self.A = self.f["atr15"].to_numpy()
        self.X = self.f[FEATURES].to_numpy(np.float32)
        # normalize features once (z-score) for stable RL inputs
        self.X = ((self.X - self.X.mean(0)) / (self.X.std(0) + 1e-8)).astype(np.float32)
        self.ctx_engine = ctx or ContextEngine()
        self.rng = np.random.default_rng(seed)
        self.i = 0
        self.ctx = RegimeContext()

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.X[self.i], self.ctx.as_vector()])

    def reset(self) -> np.ndarray:
        self.i = 30
        self.ctx = self.ctx_engine.vibe_check([])   # mock/live context
        return self._obs()

    def step(self, action: int):
        """Returns (next_obs, reward, done, info)."""
        reward = 0.0
        info = {"action": action, "traded": False}
        atr = self.A[self.i]
        if action in (1, 2) and np.isfinite(atr) and atr > 0:
            # News veto: block mean-reversion-style entries in headline regimes
            if self.ctx.news_driven == 1:
                info["vetoed"] = True
            else:
                entry = self.C[self.i]
                fwd = slice(self.i + 1, self.i + 1 + HOLD_CAP)
                # PASSIVE fill both sides (symmetric adverse selection): a
                # long fills only if next low trades strictly BELOW the
                # limit; a short only if next high trades strictly ABOVE it.
                nxt_ok = self.i + 1 < len(self.L)
                if action == 1:
                    filled = nxt_ok and self.L[self.i + 1] < entry
                else:
                    filled = nxt_ok and self.H[self.i + 1] > entry
                if filled:
                    realized = _trailing_outcome(
                        entry, atr, self.H[fwd], self.L[fwd], self.C[fwd],
                        short=(action == 2))
                    reward = (realized - FRICTION) * 1e4   # reward in bps
                    info.update(traded=True, realized_bps=realized * 1e4)
                else:
                    info["no_fill"] = True
        # advance one bar; refresh context each minute
        self.i += 1
        self.ctx = self.ctx_engine.vibe_check([])
        done = self.i >= len(self.C) - HOLD_CAP - 1
        return self._obs(), reward, done, info
