"""Does the meta-filter STACK with the regime gate, or overlap it?

The regime gate improves results by benching bad STRATEGIES. The meta-filter
improves results by culling bad individual TRADES. If the meta-filter only
helps within strategies the gate already benches, it is redundant. If it
also improves the strategies the gate KEEPS (the profitable ones), it is
orthogonal and worth deploying as a second gate.

Test: train the meta-model on setup-fired train bars, then on the test set
measure, per strategy, the effect of keeping only the meta-model's better
half of trades. Watch especially the strategies that are already profitable.

Usage: python meta_stack.py
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


def build(split, keep_strategy=False):
    X, _, ret_long, ret_short, setups = split
    xs, labels, rets, strat = [], [], [], []
    for i, name in enumerate(STRATEGY_NAMES):
        if name not in config.ACTIVE_STRATEGIES:
            continue
        mask = setups[:, i] > 0
        if not mask.any():
            continue
        r = (ret_short if is_short(name) else ret_long)[mask]
        xs.append(X[mask]); rets.append(r)
        labels.append((r > 0).astype(np.float32))
        strat += [name] * int(mask.sum())
    out = [np.concatenate(xs), np.concatenate(labels), np.concatenate(rets)]
    if keep_strategy:
        out.append(np.array(strat))
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset()
    Xtr, ytr, _ = build(ds["train"])
    Xva, yva, _ = build(ds["val"])
    Xte, yte, rte, ste = build(ds["test"], keep_strategy=True)

    scaler = Scaler().fit(Xtr)
    Xtr, Xva, Xte = (scaler.transform(Xtr), scaler.transform(Xva),
                     scaler.transform(Xte))

    def train_seed(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        model = DayTraderNet(dropout=0.4).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=3e-4)
        crit = nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(torch.from_numpy(Xtr),
                                          torch.from_numpy(ytr)),
                            batch_size=config.BATCH_SIZE, shuffle=True)
        Xva_t, yva_t = torch.from_numpy(Xva).to(device), torch.from_numpy(yva).to(device)
        best, state, stale = 1e9, None, 0
        for epoch in range(40):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                nn.BCEWithLogitsLoss()(model(xb.to(device)), yb.to(device)).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            model.eval()
            with torch.no_grad():
                vl = crit(model(Xva_t), yva_t).item()
            if vl < best: best, state, stale = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
            else: stale += 1
            if stale >= 6: break
        model.load_state_dict(state); model.eval()
        return model

    print("Training meta-model...")
    models = [train_seed(s) for s in (0, 1, 2, 3, 4)]
    with torch.no_grad():
        p = np.mean([torch.sigmoid(m(torch.from_numpy(Xte).to(device))).cpu().numpy()
                     for m in models], axis=0)

    # Pick the keep-threshold on VALIDATION (not test) to avoid bias.
    with torch.no_grad():
        pv = np.mean([torch.sigmoid(m(torch.from_numpy(Xva).to(device))).cpu().numpy()
                      for m in models], axis=0)
    thr = np.median(pv)  # keep the better half
    cost = config.COST_PER_TRADE
    print(f"Keep-threshold (val median): {thr:.3f}\n")

    print(f"{'strategy':20s} {'all avg':>9s} {'kept avg':>9s} {'all n':>7s} "
          f"{'kept n':>7s}  verdict")
    for name in config.ACTIVE_STRATEGIES:
        m = ste == name
        if m.sum() < 50:
            continue
        allr = rte[m] - cost
        kept = rte[m & (p >= thr)] - cost
        a, k = allr.mean() * 1e4, (kept.mean() * 1e4 if len(kept) else float("nan"))
        verdict = ("improves" if k > a + 0.5 else
                   "hurts" if k < a - 0.5 else "flat")
        print(f"{name:20s} {a:+8.1f} {k:+8.1f} {len(allr):7d} {len(kept):7d}"
              f"  {verdict}")

    prof = ["momentum", "orb", "disc_orb_wide", "gap_fade", "breakout"]
    for label, names in [("PROFITABLE strategies (gate keeps)", prof),
                         ("ALL active", config.ACTIVE_STRATEGIES)]:
        m = np.isin(ste, names)
        allr = rte[m] - cost
        kept = rte[m & (p >= thr)] - cost
        print(f"\n{label}: all {allr.mean()*1e4:+.1f} bps ({len(allr)}) -> "
              f"meta-kept {kept.mean()*1e4:+.1f} bps ({len(kept)})")


if __name__ == "__main__":
    main()
