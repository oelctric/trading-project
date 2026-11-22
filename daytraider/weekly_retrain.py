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
  $dir = "C:\\Users\\adamh\\OneDrive\\Desktop\\daytraider"
  schtasks /Create /SC WEEKLY /D SUN /TN "DaytraderRetrain" /ST 18:00 ^
    /TR "$py $dir\\weekly_retrain.py" /F
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
    cache = config.MODEL_DIR / "dataset_v2.npz"
    if cache.exists():
        cache.unlink()

    start = time.time()
    # Train in two chunks so each stays within typical time limits.
    for seeds in (["0", "1", "2"], ["3", "4"]):
        r = subprocess.run([sys.executable, str(config.PROJECT_ROOT / "train.py"),
                            "--seeds", *seeds], capture_output=True, text=True)
        if r.returncode != 0:
            log(f"retrain FAILED on seeds {seeds}: {r.stderr[-500:]}")
            return
        # Echo the final metrics line from each chunk.
        tail = [l for l in r.stdout.splitlines() if "test_acc" in l
                or "threshold" in l]
        for l in tail:
            log("  " + l.strip())

    mins = (time.time() - start) / 60
    log(f"weekly retrain complete in {mins:.0f} min -- restart the bot to "
        f"load new weights")


if __name__ == "__main__":
    main()
