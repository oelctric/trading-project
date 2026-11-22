"""Improve the PURE-ML system: slim + regularized, re-raced.

Attribution (explain.py) showed the full 54-feature model is overfit — about
half its features had negative importance (shuffling them *raised* accuracy).
This trains a fresh ensemble on ONLY the positive-importance features, with
heavier regularization, then re-runs the pure-ML entry race to see whether
the network alone can finally clear break-even.

Self-contained: slices the cached dataset's feature axis (no re-download),
saves to artifacts/slim_* so the live model is untouched.

Usage: python pure_ml_slim.py
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from data import FEATURE_COLUMNS, load_dataset
from model import DayTraderNet, Scaler

# Features with clearly positive importance in explain.py (the ones the
# network actually reasons from). Everything else was noise or harmful.
SLIM = ["vix_trend", "vix_z", "macd_hist", "pdl_dist", "down_streak",
        "vix_chg", "mkt_above_vwap", "body_pct", "vol_z", "pdc_dist",
        "pdh_dist", "clv", "effort_result", "doji", "upper_wick"]
IDX = [FEATURE_COLUMNS.index(f) for f in SLIM]
SEEDS = [0, 1, 2, 3, 4]


def train_seed(seed, Xtr, ytr, Xva, yva, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Smaller + more regularized than the production net: fight overfitting.
    model = DayTraderNet(n_features=len(SLIM), hidden_size=32, num_layers=2,
                         dropout=0.5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    crit = nn.BCEWithLogitsLoss()
    tds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    loader = DataLoader(tds, batch_size=config.BATCH_SIZE, shuffle=True)
    Xva_t = torch.from_numpy(Xva).to(device)
    best, best_state, stale = 1e9, None, 0
    for epoch in range(40):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = crit(model(Xva_t), torch.from_numpy(yva).to(device)).item()
        if vloss < best:
            best, best_state, stale = vloss, {k: v.cpu().clone()
                                              for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= 6:
            break
    model.load_state_dict(best_state)
    model.eval()
    print(f"  seed {seed}: best val_loss={best:.4f} (stopped epoch {epoch})")
    return model


def probs(models, X, device):
    with torch.no_grad():
        out = [torch.sigmoid(m(torch.from_numpy(X).to(device))).cpu().numpy()
               for m in models]
    return np.mean(out, axis=0)


def summarize(label, rets):
    rets = np.asarray(rets)
    if len(rets) == 0:
        print(f"  {label:30s} no trades"); return
    print(f"  {label:30s} trades={len(rets):6d}  avg={rets.mean()*1e4:+.1f} "
          f"bps  hit={(rets>0).mean():.3f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Slim feature set ({len(SLIM)}): {SLIM}\nDevice: {device}\n")

    ds = load_dataset()
    Xtr, ytr = ds["train"][0][:, :, IDX], ds["train"][1]
    Xva, yva = ds["val"][0][:, :, IDX], ds["val"][1]
    Xte, yte = ds["test"][0][:, :, IDX], ds["test"][1]
    ret_long, ret_short = ds["test"][2], ds["test"][3]

    scaler = Scaler().fit(Xtr)
    Xtr, Xva, Xte = (scaler.transform(Xtr), scaler.transform(Xva),
                     scaler.transform(Xte))

    print("Training slim regularized ensemble...")
    models = [train_seed(s, Xtr, ytr, Xva, yva, device) for s in SEEDS]
    p = probs(models, Xte, device)
    acc = ((p > 0.5) == (yte > 0.5)).mean()
    print(f"\nSlim ensemble test accuracy: {acc:.4f} "
          f"(base {max(yte.mean(), 1-yte.mean()):.4f})")

    cost = config.COST_PER_TRADE
    print("\n=== PURE-ML (slim model) entry race ===")
    for pct in (30, 20, 10, 5, 2):
        hi = np.percentile(p, 100 - pct)
        lo = np.percentile(p, pct)
        both = np.concatenate([ret_long[p >= hi] - cost,
                               ret_short[p <= lo] - cost])
        summarize(f"top/bottom {pct}% long+short", both)
    print("  (compare: full 54-feat pure-ML was -8 to -10 bps at these bands)")


if __name__ == "__main__":
    main()
