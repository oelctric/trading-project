"""Train an ensemble of LSTMs on historical intraday bars.

Usage: python train.py [--seeds 0 1 2]
Downloads data once (cached on disk), trains one net per requested seed
(default: all of config.ENSEMBLE_SEEDS), saves each to
artifacts/model_seed{k}.pt, then calibrates the diagnostic threshold and
reports test metrics for whatever seed models exist on disk.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from data import load_dataset
from model import (DayTraderNet, Scaler, ensemble_probs, load_ensemble,
                   save_threshold, seed_path)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            correct += ((torch.sigmoid(logits) > 0.5) == yb.bool()).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


def train_one(seed: int, loaders, device) -> DayTraderNet:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DayTraderNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2)

    best_val, stale = float("inf"), 0
    path = seed_path(seed)
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val, stale = val_loss, 0
            torch.save(model.state_dict(), path)
        else:
            stale += 1
        if stale >= config.EARLY_STOP_PATIENCE:
            break
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    print(f"  seed {seed}: stopped epoch {epoch}, "
          f"best val_loss={best_val:.4f}, val_acc={val_acc:.3f}")
    return model


def ensemble_metrics(models, X: torch.Tensor, y: np.ndarray, device):
    probs = []
    for i in range(0, len(X), 4096):
        probs.append(ensemble_probs(models, X[i:i + 4096].to(device)).cpu())
    probs = torch.cat(probs).numpy()
    acc = ((probs > 0.5) == (y > 0.5)).mean()
    return probs, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=config.ENSEMBLE_SEEDS)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Downloading data...")
    ds = load_dataset()
    X_train, y_train = ds["train"][0], ds["train"][1]
    X_val, y_val = ds["val"][0], ds["val"][1]
    X_test, y_test = ds["test"][0], ds["test"][1]
    print(f"Sequences  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
    print(f"Up-bar base rate  train={y_train.mean():.3f}  test={y_test.mean():.3f}")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    scaler = Scaler().fit(X_train)
    scaler.save()

    loaders = {}
    for name, (X, y) in {"train": (X_train, y_train),
                         "val": (X_val, y_val)}.items():
        tds = TensorDataset(torch.from_numpy(scaler.transform(X)),
                            torch.from_numpy(y))
        loaders[name] = DataLoader(tds, batch_size=config.BATCH_SIZE,
                                   shuffle=(name == "train"))

    print(f"Training seeds {args.seeds}...")
    for seed in args.seeds:
        train_one(seed, loaders, device)
    models = [m.to(device) for m in load_ensemble()]
    print(f"Calibrating on ensemble of {len(models)} models")

    Xv = torch.from_numpy(scaler.transform(X_val))
    val_probs, val_acc = ensemble_metrics(models, Xv, y_val, device)
    threshold = float(np.percentile(val_probs, config.THRESHOLD_PERCENTILE))
    save_threshold(threshold)

    Xt = torch.from_numpy(scaler.transform(X_test))
    _, test_acc = ensemble_metrics(models, Xt, y_test, device)
    print(f"\nEnsemble val_acc={val_acc:.3f}  test_acc={test_acc:.3f}  "
          f"(base rate {max(y_test.mean(), 1 - y_test.mean()):.3f})")
    print(f"Calibrated prob threshold (p{config.THRESHOLD_PERCENTILE} of val): "
          f"{threshold:.4f}")
    print(f"Saved {len(models)} models to {config.MODEL_DIR}")
    # Free GPU memory promptly so a lingering process holds no VRAM.
    del models, loaders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
