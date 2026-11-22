"""Alternative ranker: does XGBoost rank setup trades better than the LSTM?

The neural net's only validated job is ranking trades WITHIN setups (top
half vs bottom half of its confidence). Gradient-boosted trees often beat
small nets on tabular data, so this trains XGBoost on the same features/
target and compares the two rankers head-to-head, per profitable strategy,
on the held-out test set.

Uses the last bar of each sequence window as the tabular feature vector.

Usage: python gbm_ranker.py
"""
from __future__ import annotations

import numpy as np
import torch
import xgboost as xgb

import config
from data import FEATURE_COLUMNS, load_dataset
from model import Scaler, ensemble_probs, load_ensemble
from strategies import STRATEGY_NAMES, is_short

PROFITABLE = ["momentum", "orb", "disc_orb_wide", "gap_fade", "breakout",
              "gap_go", "pullback"]


def main():
    ds = load_dataset()
    Xtr, ytr = ds["train"][0][:, -1, :], ds["train"][1]   # last-bar features
    Xva, yva = ds["val"][0][:, -1, :], ds["val"][1]
    Xte = ds["test"][0]
    yte = ds["test"][1]
    ret_long, ret_short, setups = ds["test"][2], ds["test"][3], ds["test"][4]

    print("Training XGBoost ranker...")
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=FEATURE_COLUMNS)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=FEATURE_COLUMNS)
    params = {"objective": "binary:logistic", "eval_metric": "logloss",
              "max_depth": 4, "eta": 0.05, "subsample": 0.8,
              "colsample_bytree": 0.8, "min_child_weight": 20}
    booster = xgb.train(params, dtr, num_boost_round=400,
                        evals=[(dva, "val")], early_stopping_rounds=30,
                        verbose_eval=False)
    xgb_p = booster.predict(xgb.DMatrix(Xte[:, -1, :],
                                        feature_names=FEATURE_COLUMNS))

    # LSTM ensemble probs on the same test set
    scaler = Scaler.load()
    models = load_ensemble()
    Xt = torch.from_numpy(scaler.transform(Xte))
    lstm_p = torch.cat([ensemble_probs(models, Xt[i:i + 4096])
                        for i in range(0, len(Xt), 4096)]).numpy()

    def split(score, mask, short):
        s = (1 - score) if short else score
        med = np.median(s[mask])
        r = (ret_short if short else ret_long)[mask] - config.COST_PER_TRADE
        top = r[s[mask] >= med]; bot = r[s[mask] < med]
        return top.mean() * 1e4, bot.mean() * 1e4

    print(f"\n{'strategy':16s} {'LSTM top/bot':>16s} {'XGB top/bot':>16s}")
    lstm_edge, xgb_edge = [], []
    for i, name in enumerate(STRATEGY_NAMES):
        if name not in PROFITABLE:
            continue
        m = setups[:, i] > 0
        if m.sum() < 50:
            continue
        short = is_short(name)
        lt, lb = split(lstm_p, m, short)
        xt, xb = split(xgb_p, m, short)
        lstm_edge.append(lt - lb); xgb_edge.append(xt - xb)
        print(f"{name:16s}   {lt:+5.1f}/{lb:+5.1f} bps    {xt:+5.1f}/{xb:+5.1f} bps")

    print(f"\nMean top-minus-bottom edge (higher = better ranking):")
    print(f"  LSTM: {np.mean(lstm_edge):+.1f} bps")
    print(f"  XGB:  {np.mean(xgb_edge):+.1f} bps")
    print(f"\n{'XGB ranks better' if np.mean(xgb_edge) > np.mean(lstm_edge) else 'LSTM ranks better (keep it)'}")


if __name__ == "__main__":
    main()
