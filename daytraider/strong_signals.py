"""Test genuinely STRONGER inputs: cross-sectional / market-structure signals.

Single-stock TA is weak because it ignores the rest of the market. Real
equity models use cross-sectional signals — a stock's behavior RELATIVE to
its peers. We add two established ones and measure whether they carry real
signal (permutation importance) and lift out-of-sample discrimination:

  breadth    : fraction of the universe with positive 5-bar return (regime)
  xs_rev_rank: this stock's 5-bar return percentile among all tickers at t
               (low = biggest relative loser = strongest bounce candidate)

Honest test: if these rank as important as VIX did (+62 bps), they're strong
inputs. If ~0, single-name hourly direction just has no more signal to give.

Usage: python strong_signals.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from data import FEATURE_COLUMNS, compute_features, download_bars
from model import DayTraderNet

NEW = ["breadth", "xs_rev_rank"]
COLS = FEATURE_COLUMNS + NEW
SEQ = config.SEQ_LEN


def build():
    frames = {}
    for t in config.TICKERS:
        try:
            frames[t] = compute_features(download_bars(t))
        except Exception as e:
            print(f"  skip {t}: {e}")
    print(f"  loaded {len(frames)} tickers")

    # cross-sectional panel of 5-bar returns aligned on common timestamps
    r5 = pd.DataFrame({t: f["close"].pct_change(5) for t, f in frames.items()})
    breadth = (r5 > 0).mean(axis=1)                       # fraction up
    xs_rank = r5.rank(axis=1, pct=True)                   # percentile per row

    for t, f in frames.items():
        f["breadth"] = breadth.reindex(f.index)
        f["xs_rev_rank"] = xs_rank[t].reindex(f.index)
    return frames


def seqs(frames, split):
    Xs, ys = [], []
    for t, f in frames.items():
        f = f.dropna(subset=COLS + ["trail_ret"])
        n = len(f)
        i_tr, i_va = int(n * 0.70), int(n * 0.85)
        part = {"train": f.iloc[:i_tr], "val": f.iloc[i_tr:i_va],
                "test": f.iloc[i_va:]}[split]
        vals = part[COLS].to_numpy(np.float32)
        tgt = part["target"].to_numpy(np.float32)
        if len(part) < SEQ:
            continue
        W = np.lib.stride_tricks.sliding_window_view(vals, SEQ, axis=0)
        Xs.append(np.ascontiguousarray(W.transpose(0, 2, 1)))
        ys.append(tgt[SEQ - 1:])
    return np.concatenate(Xs), np.concatenate(ys)


def train_eval(Xtr, ytr, Xva, yva, Xte, yte, device):
    sc_mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sc_std = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-8
    norm = lambda X: ((X - sc_mean) / sc_std).astype(np.float32)
    Xtr, Xva, Xte = norm(Xtr), norm(Xva), norm(Xte)
    model = DayTraderNet(n_features=len(COLS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr),
                                      torch.from_numpy(ytr)),
                        batch_size=256, shuffle=True)
    def predict(X, bs=4096):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = torch.from_numpy(X[i:i+bs]).to(device)
                out.append(torch.sigmoid(model(xb)).cpu().numpy())
        return np.concatenate(out)

    best, state, stale = 1e9, None, 0
    for ep in range(40):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        model.eval()
        pv = predict(Xva)
        vl = nn.functional.binary_cross_entropy(
            torch.from_numpy(pv), torch.from_numpy(yva)).item()
        if vl < best:
            best, state, stale = vl, {k: v.cpu().clone()
                                      for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= 6:
            break
    model.load_state_dict(state); model.eval()

    def acc(X):
        return predict(X)
    p = acc(Xte)
    base = max(yte.mean(), 1 - yte.mean())
    test_acc = ((p > 0.5) == (yte > 0.5)).mean()
    print(f"\n  test acc {test_acc:.4f}  (base {base:.4f}, "
          f"baseline-model ~0.556)")

    # permutation importance of the NEW features
    rng = np.random.default_rng(0)
    print("  permutation importance (accuracy drop when shuffled):")
    for name in NEW + ["vix_z", "rsi"]:
        j = COLS.index(name)
        Xp = Xte.copy()
        drop = 0.0
        for _ in range(3):
            Xp[:, :, j] = Xte[rng.permutation(len(Xte))][:, :, j]
            drop += test_acc - ((acc(Xp) > 0.5) == (yte > 0.5)).mean()
        Xp[:, :, j] = Xte[:, :, j]
        print(f"    {name:14s} {drop/3*1e4:+6.1f} bps")


def main():
    if torch.cuda.is_available():
        best = max(range(torch.cuda.device_count()),
                   key=lambda i: torch.cuda.get_device_properties(i).total_memory)
        device = torch.device(f"cuda:{best}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print("Building cross-sectional features...")
    frames = build()
    Xtr, ytr = seqs(frames, "train")
    Xva, yva = seqs(frames, "val")
    Xte, yte = seqs(frames, "test")
    print(f"  sequences: train {len(Xtr):,}  test {len(Xte):,}  "
          f"({len(COLS)} features incl {NEW})")
    train_eval(Xtr, ytr, Xva, yva, Xte, yte, device)


if __name__ == "__main__":
    main()
