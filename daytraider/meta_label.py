"""Meta-labeling: make the ML genuinely better at its ACTUAL job.

Standalone direction prediction has a hard data ceiling (pure_ml.py /
pure_ml_slim.py both fail). But the model is only ever used on setup-fired
bars, so training it on every bar is a train/serve mismatch. Meta-labeling
(Lopez de Prado) fixes this: train ONLY on setup-fired bars, predicting
whether THAT trade (long for long setups, short for short setups) wins.

If the meta-model's confidence sorts winners from losers, filtering to its
higher-confidence trades should raise the per-trade edge above taking all
setup trades. That is a real, deployable improvement to the ML's role.

Usage: python meta_label.py
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from data import load_dataset
from model import DayTraderNet, Scaler
from strategies import STRATEGY_NAMES, is_short


def build_meta(split):
    """From (X, y, ret_long, ret_short, setups) build per-fired-setup
    samples: input window, win label, and the trade's own return."""
    X, _, ret_long, ret_short, setups = split
    xs, labels, rets = [], [], []
    for i, name in enumerate(STRATEGY_NAMES):
        if name not in config.ACTIVE_STRATEGIES:
            continue
        mask = setups[:, i] > 0
        if not mask.any():
            continue
        r = (ret_short if is_short(name) else ret_long)[mask]
        xs.append(X[mask])
        rets.append(r)
        labels.append((r > 0).astype(np.float32))
    return (np.concatenate(xs), np.concatenate(labels).astype(np.float32),
            np.concatenate(rets))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset()
    Xtr, ytr, _ = build_meta(ds["train"])
    Xva, yva, _ = build_meta(ds["val"])
    Xte, yte, rte = build_meta(ds["test"])
    print(f"Meta samples (setup-fired only): train={len(Xtr)}  "
          f"val={len(Xva)}  test={len(Xte)}")
    print(f"Win base rate: train={ytr.mean():.3f}  test={yte.mean():.3f}")

    scaler = Scaler().fit(Xtr)
    Xtr, Xva, Xte = (scaler.transform(Xtr), scaler.transform(Xva),
                     scaler.transform(Xte))

    def train_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = DayTraderNet(dropout=0.4).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=3e-4)
        crit = nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(torch.from_numpy(Xtr),
                                          torch.from_numpy(ytr)),
                            batch_size=config.BATCH_SIZE, shuffle=True)
        Xva_t = torch.from_numpy(Xva).to(device)
        yva_t = torch.from_numpy(yva).to(device)
        best, state, stale = 1e9, None, 0
        for epoch in range(40):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = crit(model(xb.to(device)), yb.to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                vloss = crit(model(Xva_t), yva_t).item()
            if vloss < best:
                best, state, stale = vloss, {k: v.cpu().clone() for k, v
                                             in model.state_dict().items()}, 0
            else:
                stale += 1
            if stale >= 6:
                break
        model.load_state_dict(state)
        model.eval()
        print(f"  seed {seed}: val_loss={best:.4f}")
        return model

    print("\nTraining meta-model ensemble on setup-fired bars...")
    models = [train_seed(s) for s in (0, 1, 2, 3, 4)]
    with torch.no_grad():
        p = np.mean([torch.sigmoid(m(torch.from_numpy(Xte).to(device)))
                     .cpu().numpy() for m in models], axis=0)

    cost = config.COST_PER_TRADE
    base = rte - cost
    print(f"\nTake ALL setup trades: {len(base)} trades  "
          f"avg={base.mean()*1e4:+.1f} bps  hit={(base>0).mean():.3f}")
    print("Filter to meta-model's higher-confidence trades:")
    for keep in (75, 50, 25, 10):
        thr = np.percentile(p, 100 - keep)
        r = rte[p >= thr] - cost
        print(f"  keep top {keep:2d}% (p>={thr:.3f})  trades={len(r):6d}  "
              f"avg={r.mean()*1e4:+.1f} bps  hit={(r>0).mean():.3f}")


if __name__ == "__main__":
    main()
