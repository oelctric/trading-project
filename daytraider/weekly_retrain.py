"""Automated weekly retrain so the model tracks the current regime.

Runs the full pipeline end to end: fresh data (cache expires), retrain all
ensemble seeds, recalibrate the threshold, and log a one-line summary with
timestamp. Intended to be triggered weekly by the OS scheduler (Windows
Task Scheduler) — e.g. Sunday evening before the trading week.

It does NOT restart the live bot; the bot reloads models from disk each
time it is (re)started. Restart the bot after a retrain to pick up the new
weights, or let your process manager do it.

Usage (manual):     python weekly_retrain.py
Scheduler command:  python C:\\...\\daytraider\\weekly_retrain.py

To register the weekly schedule on Windows (run once, in PowerShell):
  $py = (Get-Command python).Source
  $dir = "<the folder this file is in, e.g. ...\trading project\daytraider>"
  schtasks /Create /SC WEEKLY /D SUN /TN "DaytraderRetrain" /ST 18:00 ^
    /TR "$py $dir\\weekly_retrain.py" /F
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import config

LOG = config.PROJECT_ROOT / "retrain_log.txt"


def log(msg: str):
    line = f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    log("weekly retrain starting")
    # Force fresh data by expiring the cache.
    cache = config.MODEL_DIR / "dataset_v3.npz"
    if cache.exists():
        cache.unlink()

    # Back up the live ensemble + scaler + threshold. A retrain that failed
    # halfway used to leave NEW seeds next to OLD seeds with a NEW scaler the
    # old seeds were never trained with - a silently broken ensemble.
    backup = config.MODEL_DIR / "_pre_retrain_backup"
    shutil.rmtree(backup, ignore_errors=True)    # never restore a stale old backup
    backup.mkdir(parents=True, exist_ok=True)
    live_files = [config.SCALER_PATH, config.THRESHOLD_PATH] + [
        config.MODEL_DIR / f"model_seed{s}.pt" for s in config.ENSEMBLE_SEEDS]
    backed_up = set()
    for f in live_files:
        if f.exists():
            shutil.copy2(f, backup / f.name)
            backed_up.add(f.name)

    start = time.time()
    # All seeds in ONE run so scaler, models and threshold always match.
    r = subprocess.run([sys.executable, str(config.PROJECT_ROOT / "train.py"),
                        "--seeds", *map(str, config.ENSEMBLE_SEEDS)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        for f in live_files:
            if f.name in backed_up:
                shutil.copy2(backup / f.name, f)
            elif f.exists():
                f.unlink()        # created by the failed run, had no previous version
        log(f"retrain FAILED - previous models restored: {r.stderr[-500:]}")
        return
    for line in r.stdout.splitlines():
        if "test_acc" in line or "threshold" in line:
            log("  " + line.strip())

    mins = (time.time() - start) / 60
    log(f"weekly retrain complete in {mins:.0f} min -- restart the bot to "
        f"load new weights")


if __name__ == "__main__":
    main()
