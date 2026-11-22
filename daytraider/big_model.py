"""Demonstration: does a ~200M-parameter 'brain' beat the tiny one?

Builds a large LSTM (hidden ~4096, ~200M params vs the production 66k),
trains it on the same data with the same discipline, and compares held-out
test accuracy. Hypothesis (from this session's overfitting evidence): the
big model MEMORIZES the train set and generalizes WORSE out-of-sample.

Kept separate from the live model files on purpose — the live bot is NOT
switched to this.

Usage: python big_model.py [hidden]
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from data import (build_sequences, compute_features, download_bars,
                  load_dataset)
from model import DayTraderNet, Scaler

# ~100 liquid US names — 2.5x the live 40-ticker universe, for "more data".
EXPANDED = [
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
    "XLI", "XLU", "SMH", "SOXX", "IYR", "GLD", "SLV", "TLT", "HYG", "EEM",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AMD", "AVGO",
    "QCOM", "MU", "INTC", "TXN", "CRM", "ORCL", "ADBE", "NFLX", "CSCO", "IBM",
    "NOW", "INTU", "AMAT", "LRCX", "ADI", "PANW", "SNPS",
    "JPM", "BAC", "GS", "WFC", "C", "USB", "PNC", "SCHW", "AXP", "BLK",
    "SPGI", "V", "MA", "PYPL", "MS",
    "XOM", "CVX", "COP", "SLB", "PSX", "MPC", "OXY", "EOG",
    "BA", "CAT", "DE", "HON", "GE", "MMM", "UPS", "FDX", "RTX", "LMT",
    "WMT", "COST", "TGT", "HD", "LOW", "DIS", "KO", "PEP", "PG", "CL",
    "MO", "PM", "MCD", "SBUX", "NKE",
    "UNH", "LLY", "PFE", "MRK", "ABBV", "TMO", "GILD", "BMY", "AMGN",
    "T", "VZ", "NEE", "DUK", "SO", "UBER", "PLTR", "COIN", "SHOP", "ABNB",
]


def build_expanded_dataset():
    """Fetch the expanded universe, build sequences, chronological split.
    Cached separately (dataset_big.npz) so it never touches the live cache."""
    cache = config.MODEL_DIR / "dataset_big.npz"
    if cache.exists():
        import time
        if (time.time() - cache.stat().st_mtime) / 3600 < 12:
            print("  using cached expanded dataset")
            z = np.load(cache)
            return {n: tuple(z[f"{n}_{i}"] for i in range(5))
                    for n in ("train", "val", "test")}
    splits = {"train": [], "val": [], "test": []}
    for t in EXPANDED:
        try:
            f = compute_features(download_bars(t))
        except Exception as e:
            print(f"  skip {t}: {e}"); continue
        n = len(f)
        i_tr, i_va = int(n * config.TRAIN_FRAC), int(n * (config.TRAIN_FRAC + config.VAL_FRAC))
        for name, part in {"train": f.iloc[:i_tr], "val": f.iloc[i_tr:i_va],
                           "test": f.iloc[i_va:]}.items():
            splits[name].append(build_sequences(part))
    out = {n: tuple(np.concatenate([c[i] for c in chunks]) for i in range(5))
           for n, chunks in splits.items()}
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{f"{n}_{i}": out[n][i]
                       for n in out for i in range(5)})
    print(f"  built expanded dataset: {len(EXPANDED)} tickers")
    return out


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def evaluate(model, X, y, device, bs=512):
    model.eval()
    correct, tot, loss = 0, 0, 0.0
    crit = nn.BCEWithLogitsLoss(reduction="sum")
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i+bs]).to(device)
            yb = torch.from_numpy(y[i:i+bs]).to(device)
            logit = model(xb)
            loss += crit(logit, yb).item()
            correct += ((torch.sigmoid(logit) > 0.5) == yb.bool()).sum().item()
            tot += len(yb)
            del xb, yb, logit
    torch.cuda.empty_cache()
    return loss / tot, correct / tot


def main():
    hidden = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    # Use the GPU with the most memory (the RTX 4090, 25.8 GB) so the 211M
    # model fits with a large batch.
    if torch.cuda.is_available():
        best = max(range(torch.cuda.device_count()),
                   key=lambda i: torch.cuda.get_device_properties(i).total_memory)
        device = torch.device(f"cuda:{best}")
        print(f"Device: {device} ({torch.cuda.get_device_name(best)})")
    else:
        device = torch.device("cpu")
        print("Device: cpu")

    big = DayTraderNet(hidden_size=hidden, num_layers=2, dropout=0.3).to(device)
    print(f"BIG model: hidden={hidden}  params = {n_params(big):,}")
    small = DayTraderNet()
    print(f"(production small model params = {n_params(small):,})\n")

    more = "--more-data" in sys.argv
    ds = build_expanded_dataset() if more else load_dataset()
    Xtr, ytr = ds["train"][0], ds["train"][1]
    Xva, yva = ds["val"][0], ds["val"][1]
    Xte, yte = ds["test"][0], ds["test"][1]
    if more:  # fit a fresh scaler on the expanded train set
        from model import Scaler as _S
        scaler = _S().fit(Xtr)
    else:
        scaler = Scaler.load()
    print(f"  {'EXPANDED' if more else 'standard'} universe")
    Xtr, Xva, Xte = (scaler.transform(Xtr), scaler.transform(Xva),
                     scaler.transform(Xte))
    base = max(yte.mean(), 1 - yte.mean())
    print(f"train {len(Xtr):,} seqs  |  params/sample ratio ~ "
          f"{n_params(big)/len(Xtr):.0f}:1  (overfit danger zone)\n")

    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr),
                                      torch.from_numpy(ytr)),
                        batch_size=512, shuffle=True)   # 4090 has room
    opt = torch.optim.AdamW(big.parameters(), lr=3e-4, weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss()
    best_val, stale = 1e9, 0
    for epoch in range(1, 21):
        big.train()
        run, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(big(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(big.parameters(), 1.0)
            opt.step()
            run += loss.item() * len(yb); n += len(yb)
        tr_loss, tr_acc = evaluate(big, Xtr[:20000], ytr[:20000], device)
        va_loss, va_acc = evaluate(big, Xva, yva, device)
        marker = ""
        if va_loss < best_val:
            best_val, stale = va_loss, 0
            torch.save(big.state_dict(), config.MODEL_DIR / "big_model.pt")
            marker = " <-saved"
        else:
            stale += 1
        print(f"epoch {epoch:2d}  train_loss={run/n:.4f}  "
              f"train_acc={tr_acc:.3f}  val_loss={va_loss:.4f}  "
              f"val_acc={va_acc:.3f}{marker}")
        if stale >= 4:
            print("early stop"); break

    big.load_state_dict(torch.load(config.MODEL_DIR / "big_model.pt",
                                   weights_only=True))
    _, te_acc = evaluate(big, Xte, yte, device)
    _, tr_acc = evaluate(big, Xtr[:20000], ytr[:20000], device)
    print(f"\n=== RESULT ===")
    print(f"BIG model ({n_params(big):,} params):")
    print(f"  train acc {tr_acc:.3f}  vs  test acc {te_acc:.3f}  "
          f"(base rate {base:.3f})")
    gap = tr_acc - te_acc
    print(f"  train-minus-test gap: {gap:+.3f}  "
          f"({'OVERFIT' if gap > 0.03 else 'ok'})")
    print(f"  production small ensemble test acc was ~0.556")
    print(f"  => bigger brain {'LOST to' if te_acc < 0.556 else 'matched'} "
          f"the tiny one out-of-sample")


if __name__ == "__main__":
    main()
