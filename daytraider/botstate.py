"""Tiny JSON state files so the bots remember things across restarts and can
see which positions belong to which bot (they share one Alpaca account)."""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAIRS_STATE = ROOT / "pairs_state.json"      # pairs bot: open pairs (+ their legs)
STOCK_TRAIL = ROOT / "stock_trail.json"      # stock bot: trailing-stop state
CRYPTO_TRAIL = ROOT / "crypto_trail.json"    # crypto bot: trailing-stop state
PAIRS_ORPHANS = ROOT / "pairs_orphans.json"  # pairs bot: unwinds not yet confirmed


class StateUnavailable(RuntimeError):
    """A state file exists but couldn't be read (locked / mid-write). Callers
    must NOT treat this as "no state" - e.g. "no pairs legs" would let the stock
    bot close the pairs bot's positions."""


def load(path: Path, default):
    """Lenient: for state that can be rebuilt (trailing stops) - any problem
    returns `default`."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def load_strict(path: Path, default):
    """For state where "empty" is dangerous: `default` ONLY if the file doesn't
    exist; transient read errors are retried, and a persistent one raises
    StateUnavailable instead of silently returning `default`."""
    if not path.exists():
        return default
    last = None
    for attempt in range(10):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as e:   # locked by OneDrive / caught mid-replace
            last = e
            time.sleep(0.2 * (attempt + 1))
            if not path.exists():
                return default
    raise StateUnavailable(f"{path.name} unreadable after retries ({last})")


def save(path: Path, data) -> bool:
    """Atomic write (never a half-written file). On Windows a file another
    process has open - e.g. OneDrive syncing, or another bot reading it - makes
    the rename fail, so retry briefly; never raise (a crash here used to abort a
    scan right after orders were placed). Returns False if it couldn't save."""
    tmp = path.with_suffix(".tmp")
    for attempt in range(10):
        try:
            tmp.write_text(json.dumps(data, indent=1))
            tmp.replace(path)
            return True
        except OSError as e:
            last = e
            time.sleep(0.2 * (attempt + 1))
    print(f"  WARNING: could not save {path.name} after retries ({last}) - "
          f"state kept in memory, will be saved on the next change")
    return False


def pairs_leg_symbols() -> set[str]:
    """Symbols currently held by the pairs bot (open pairs AND unwinds still being
    resolved) - other bots must not touch them."""
    legs = {leg for k in load_strict(PAIRS_STATE, {}) for leg in k.split("|")}
    return legs | {o["symbol"] for o in load_strict(PAIRS_ORPHANS, {}).values()}
