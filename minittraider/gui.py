"""Native Windows desktop app for minittraider (tkinter - no browser, no web server).

Run it:  python gui.py     (or double-click run_gui.bat)

Layout:
  top    - account bar: equity, cash, realized P&L, win rate, status
  left   - watchlist: live price, regime, confidence, and decision per symbol
  right  - open positions with live unrealized P&L and age
  bottom - activity log with the bot's reasoning (click a watchlist row to
           see that symbol's full thinking from the latest scan)

The market scanner runs on a background thread and hands the GUI immutable
snapshots through a queue, so the window never freezes during data fetches.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk

import config
from bounce import decide_bounce
from brain import decide, fmt_price
from cognitive import decide_cognitive, record_result
from data import get_candles, fetch_micro, order_book_walls, tape_imbalance
from portfolio import Portfolio
from logger import get_daily_log_file, setup_logger

setup_logger(level=config.LOG_LEVEL, log_file=get_daily_log_file() if config.LOG_TO_FILE else None)
import swing_live

# ---------------------------------------------------------------- theme

BG = "#12151c"          # window background
PANEL = "#1a1e28"       # panel background
ROW_ALT = "#1f2430"
FG = "#d6dae3"          # main text
DIM = "#7a8194"         # secondary text
GREEN = "#4caf7d"
RED = "#e05c5c"
YELLOW = "#d9a441"
BLUE = "#5b8dd9"
FONT = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_BIG = ("Segoe UI", 14, "bold")
MONO = ("Consolas", 10)

WATCHLIST: list[tuple[str, bool]] = (
    [(s, True) for s in config.CRYPTO_SYMBOLS] + [(s, False) for s in config.STOCK_SYMBOLS]
)


# ---------------------------------------------------------------- worker

def _age(minutes: float) -> str:
    """Position age that stays readable for multi-day swing trades."""
    if minutes >= 1440:
        return f"{minutes / 1440:.1f}d"
    if minutes >= 60:
        return f"{minutes / 60:.1f}h"
    return f"{minutes:.1f}m"


class Scanner(threading.Thread):
    """Background thread: fetches data, runs the brain, trades the paper account."""

    def __init__(self, out: queue.Queue) -> None:
        super().__init__(daemon=True)
        self.out = out
        self.portfolio = Portfolio()
        self.running = threading.Event()
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()   # held for a whole scan
        self.reset_requested = threading.Event()   # handled by THIS thread between scans

    def log(self, text: str, tag: str = "info") -> None:
        self.out.put(("log", datetime.now().strftime("%H:%M:%S"), text, tag))

    def _do_reset(self) -> None:
        import os
        from portfolio import STATE_FILE
        with self.lock:
            if os.path.exists(STATE_FILE):
                os.remove(STATE_FILE)
            self.portfolio = Portfolio()
        self.reset_requested.clear()
        self.log(f"account reset to ${config.STARTING_CASH:,.0f}")

    def run(self) -> None:
        while not self.stop_flag.is_set():
            if self.reset_requested.is_set():
                self._do_reset()
            if self.running.is_set():
                try:
                    with self.lock:
                        self.scan()
                except Exception as e:
                    self.log(f"scan failed: {e}", "bad")
            # sleep in small steps so pause/quit reacts quickly
            for _ in range(config.SCAN_INTERVAL_SECONDS * 4):
                if self.stop_flag.is_set() or not self.running.is_set() \
                        or self.reset_requested.is_set():
                    break
                time.sleep(0.25)
            if not self.running.is_set():
                time.sleep(0.25)

    def scan(self) -> None:
        rows = []
        prices: dict[str, float] = {}
        reasons: dict[str, str] = {}
        dfs: dict = {}
        p = self.portfolio
        swing_mode = config.STRATEGY == config.Strategy.SWING
        if swing_mode:
            swing_live.prefetch([s for s, c in WATCHLIST if c or not config.SWING_CRYPTO_ONLY])

        for symbol, is_crypto in WATCHLIST:
            if swing_mode and config.SWING_CRYPTO_ONLY and not is_crypto and symbol not in p.positions:
                rows.append((symbol, "-", "-", "-", "skipped (swing trades crypto only)"))
                continue
            try:
                df = get_candles(symbol, is_crypto)
            except Exception as e:
                rows.append((symbol, "-", "data error", "-", str(e)[:40]))
                continue
            dfs[symbol] = df

            stale = df is None or df.empty or (
                (time.time() - df.index[-1].timestamp()) / 60 > config.STALE_DATA_MINUTES
            )
            if stale and symbol in p.positions and p.positions[symbol].exit_state:
                rows.append((symbol, "-", "-", "-", "no fresh data - holding swing position"))
                continue
            if stale:
                if symbol in p.positions:
                    price = float(df["close"].iloc[-1]) if df is not None and not df.empty \
                        else p.positions[symbol].entry
                    pnl = p.close(symbol, price, "stale data / market closed")
                    self.log(f"closed {symbol} (market closed) for {pnl:+.2f}",
                             "good" if pnl >= 0 else "bad")
                rows.append((symbol, "-", "closed", "-", "market closed / no fresh data"))
                continue

            price = float(df["close"].iloc[-1])
            prices[symbol] = price

            if symbol in p.positions and p.positions[symbol].exit_state:
                pos = p.positions[symbol]
                try:
                    bars = swing_live.closed_bars(symbol, is_crypto)
                except Exception:
                    bars = None
                why, pos.exit_state, fill = swing_live.manage(pos.exit_state, bars, price)
                if fill is not None and p.check_liquidation(symbol, fill):
                    why = "LIQUIDATED (gapped through the stop and the liquidation level)"
                elif fill is None and p.check_liquidation(symbol, price):  # an earlier stop fill wins
                    why, fill = "LIQUIDATED (price hit the liquidation level)", None
                p.save()
                if why:
                    exit_px = fill if fill is not None else price
                    pnl = p.close(symbol, exit_px, why)
                    record_result(pnl)
                    self.log(f"CLOSE {pos.side} {symbol} @ {fmt_price(exit_px)} -> {pnl:+.2f} ({why})",
                             "good" if pnl >= 0 else "bad")
                    # like trader.py: no re-entry in the same scan (it would act on a
                    # signal bar from while we were still in the trade)
                    rows.append((symbol, fmt_price(price), "-", "-", f"CLOSED {pos.side} - {why}"[:70]))
                    continue
                else:
                    note = swing_live.describe(pos.exit_state, price)
                    reasons[symbol] = f"holding {pos.side} {symbol}: {note}"
                    rows.append((symbol, fmt_price(price), "-", "-", f"HOLD {pos.side} - {note}"[:70]))
                    continue
            elif symbol in p.positions:
                why = p.check_exit(symbol, price)
                if why:
                    side = p.positions[symbol].side
                    pnl = p.close(symbol, price, why)
                    record_result(pnl)
                    self.log(f"CLOSE {side} {symbol} @ {fmt_price(price)} -> {pnl:+.2f} ({why})",
                             "good" if pnl >= 0 else "bad")

            exit_state = None
            if config.STRATEGY == config.Strategy.SWING:
                if config.SWING_CRYPTO_ONLY and not is_crypto:
                    rows.append((symbol, fmt_price(price), "-", "-", "skipped (swing trades crypto only)"))
                    continue
                try:
                    bars = swing_live.closed_bars(symbol, is_crypto)
                except Exception as e:
                    rows.append((symbol, fmt_price(price), "-", "-", f"daily bars unavailable: {e}"[:70]))
                    continue
                decision = swing_live.decide_swing(symbol, bars)
                decision.price = price
                if decision.action != "none" and p.can_open(symbol):
                    exit_state = swing_live.new_exit_state(symbol, decision.action, price,
                                                           decision.atr, bars)
                    decision.sl = exit_state["stop"]
                max_hold = 0
            elif config.STRATEGY == config.Strategy.COGNITIVE:
                walls = order_book_walls(symbol) if is_crypto else None
                tape = tape_imbalance(symbol) if is_crypto else None
                decision = decide_cognitive(symbol, df, walls, tape,
                                            equity=p.equity(prices))
                max_hold = config.COGNITIVE_MAX_HOLD_MINUTES
            elif config.STRATEGY == config.Strategy.BOUNCE:
                walls = order_book_walls(symbol) if is_crypto else None
                tape = tape_imbalance(symbol) if is_crypto else None
                decision = decide_bounce(symbol, df, walls, tape)
                max_hold = config.BOUNCE_MAX_HOLD_MINUTES
            else:
                leader = "BTC-USD" if is_crypto else "SPY"
                leader_df = dfs.get(leader) if symbol != leader else None
                micro = fetch_micro(symbol, is_crypto, leader_df, leader)
                decision = decide(symbol, df, micro)
                max_hold = 0
            reasons[symbol] = decision.explain()

            if decision.action != "none" and p.can_open(symbol):
                pos = p.open(symbol, decision.action, decision.price,
                             decision.atr, decision.confidence, decision.reasons[-1],
                             tp=decision.tp, sl=decision.sl, max_hold=max_hold,
                             exit_state=exit_state)
                if pos:
                    if exit_state is not None:
                        swing_live.mark_acted(symbol, bars)   # only once it really opened
                    tp_txt = "dynamic" if pos.exit_state else fmt_price(pos.take_profit)
                    self.log(f"OPEN {pos.side.upper()} {symbol} @ {fmt_price(price)} "
                             f"(confidence {decision.confidence:.0%}) "
                             f"tp {tp_txt} sl {fmt_price(pos.stop_loss)}", "trade")
                    self.log("  " + decision.reasons[-1], "dim")

            verdict = decision.action.upper() if decision.action != "none" else "wait"
            note = decision.reasons[-1].removeprefix("=> no trade: ").removeprefix("=> ")
            rows.append((symbol, f"{fmt_price(price)}", decision.regime or "-",
                         f"{decision.confidence:.0%}", verdict + " - " + note[:60]))

        positions = [
            (pos.symbol, pos.side, f"{pos.qty:.6f}", f"{fmt_price(pos.entry)}",
             f"{fmt_price(prices.get(sym, pos.entry))}",
             pos.pnl(prices.get(sym, pos.entry)),
             _age(pos.age_minutes()),
             "dynamic" if pos.exit_state else f"{fmt_price(pos.take_profit)}",
             f"{fmt_price(pos.exit_state['stop'] if pos.exit_state else pos.stop_loss)}")
            for sym, pos in p.positions.items()
        ]
        total = p.wins + p.losses
        stats = {
            "equity": p.equity(prices),
            "cash": p.cash,
            "pnl": p.realized_pnl,
            "trades": total,
            "winrate": (p.wins / total) if total else None,
            "open": len(p.positions),
        }
        self.out.put(("snapshot", rows, positions, stats, reasons))


# ---------------------------------------------------------------- GUI

class App:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("minittraider - paper trading")
        self.root.configure(bg=BG)
        self.root.geometry("1150x700")
        self.root.minsize(900, 560)

        self.queue: queue.Queue = queue.Queue()
        self.scanner = Scanner(self.queue)
        self.reasons: dict[str, str] = {}

        self._style()
        self._build()
        self.scanner.start()
        self.root.after(200, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    def _style(self) -> None:
        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                    foreground=FG, rowheight=26, font=FONT, borderwidth=0)
        s.configure("Treeview.Heading", background="#232837", foreground=DIM,
                    font=FONT_BOLD, borderwidth=0, relief="flat")
        s.map("Treeview", background=[("selected", "#2d3550")])
        s.map("Treeview.Heading", background=[("active", "#232837")])
        s.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])

    # ------------------------------------------------------------ layout
    def _build(self) -> None:
        # --- top account bar ---
        bar = tk.Frame(self.root, bg=PANEL, padx=14, pady=10)
        bar.pack(fill="x", padx=8, pady=(8, 4))

        self.stat_labels: dict[str, tk.Label] = {}
        for key, title in [("equity", "EQUITY"), ("cash", "CASH"), ("pnl", "REALIZED P&L"),
                           ("trades", "TRADES"), ("winrate", "WIN RATE"), ("open", "OPEN")]:
            cell = tk.Frame(bar, bg=PANEL)
            cell.pack(side="left", padx=(0, 28))
            tk.Label(cell, text=title, bg=PANEL, fg=DIM, font=("Segoe UI", 8, "bold")).pack(anchor="w")
            lbl = tk.Label(cell, text="-", bg=PANEL, fg=FG, font=FONT_BIG)
            lbl.pack(anchor="w")
            self.stat_labels[key] = lbl

        btns = tk.Frame(bar, bg=PANEL)
        btns.pack(side="right")
        self.start_btn = tk.Button(btns, text="▶  Start", command=self._toggle,
                                   bg=GREEN, fg="#0b0e14", font=FONT_BOLD, relief="flat",
                                   padx=18, pady=6, cursor="hand2", activebackground="#5fc290")
        self.start_btn.pack(side="left", padx=(0, 8))
        tk.Button(btns, text="Reset account", command=self._reset,
                  bg="#2a3040", fg=FG, font=FONT, relief="flat",
                  padx=12, pady=6, cursor="hand2", activebackground="#39415a").pack(side="left")

        self.status = tk.Label(bar, text="paused", bg=PANEL, fg=YELLOW, font=FONT)
        self.status.pack(side="right", padx=14)

        # --- middle: watchlist | positions ---
        mid = tk.Frame(self.root, bg=BG)
        mid.pack(fill="both", expand=True, padx=8, pady=4)
        mid.columnconfigure(0, weight=5)
        mid.columnconfigure(1, weight=4)
        mid.rowconfigure(1, weight=1)

        tk.Label(mid, text="WATCHLIST  (click a row to see its reasoning)",
                 bg=BG, fg=DIM, font=("Segoe UI", 8, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 3))
        cols = ("symbol", "price", "regime", "conf", "decision")
        self.watch = ttk.Treeview(mid, columns=cols, show="headings", selectmode="browse")
        for c, w, anchor in [("symbol", 90, "w"), ("price", 90, "e"), ("regime", 70, "center"),
                             ("conf", 55, "e"), ("decision", 380, "w")]:
            self.watch.heading(c, text=c.upper())
            self.watch.column(c, width=w, anchor=anchor, stretch=(c == "decision"))
        self.watch.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        self.watch.bind("<<TreeviewSelect>>", self._show_reasoning)
        for sym, _ in WATCHLIST:
            self.watch.insert("", "end", iid=sym, values=(sym, "-", "-", "-", "starting..."))

        tk.Label(mid, text="OPEN POSITIONS", bg=BG, fg=DIM,
                 font=("Segoe UI", 8, "bold")).grid(row=0, column=1, sticky="w", pady=(0, 3))
        pcols = ("symbol", "side", "qty", "entry", "now", "pnl", "age", "tp", "sl")
        self.pos = ttk.Treeview(mid, columns=pcols, show="headings", selectmode="none")
        for c, w in [("symbol", 80), ("side", 50), ("qty", 80), ("entry", 80),
                     ("now", 80), ("pnl", 70), ("age", 50), ("tp", 80), ("sl", 80)]:
            self.pos.heading(c, text=c.upper())
            self.pos.column(c, width=w, anchor="e" if c not in ("symbol", "side") else "w")
        self.pos.grid(row=1, column=1, sticky="nsew", padx=(4, 0))

        for tree in (self.watch, self.pos):
            tree.tag_configure("alt", background=ROW_ALT)
            tree.tag_configure("long", foreground=GREEN)
            tree.tag_configure("short", foreground=RED)
            tree.tag_configure("wait", foreground=DIM)
            tree.tag_configure("win", foreground=GREEN)
            tree.tag_configure("loss", foreground=RED)

        # --- bottom: activity log ---
        tk.Label(self.root, text="ACTIVITY & REASONING", bg=BG, fg=DIM,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=10)
        logframe = tk.Frame(self.root, bg=PANEL)
        logframe.pack(fill="both", padx=8, pady=(2, 8), ipady=2)
        self.log = tk.Text(logframe, height=11, bg=PANEL, fg=FG, font=MONO,
                           relief="flat", wrap="word", state="disabled",
                           insertbackground=FG, padx=10, pady=6)
        sb = ttk.Scrollbar(logframe, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)
        for tag, color in [("info", FG), ("dim", DIM), ("good", GREEN),
                           ("bad", RED), ("trade", BLUE), ("time", DIM)]:
            self.log.tag_configure(tag, foreground=color)

        self._log_line(datetime.now().strftime("%H:%M:%S"),
                       "ready - paper account loaded, press Start to begin scanning", "info")

    # ------------------------------------------------------------ actions
    def _toggle(self) -> None:
        if self.scanner.running.is_set():
            self.scanner.running.clear()
            self.start_btn.configure(text="▶  Start", bg=GREEN)
            self.status.configure(text="paused", fg=YELLOW)
        else:
            self.scanner.running.set()
            self.start_btn.configure(text="⏸  Pause", bg=YELLOW)
            self.status.configure(text=f"scanning every {config.SCAN_INTERVAL_SECONDS}s", fg=GREEN)

    def _reset(self) -> None:
        # The scanner thread performs the reset between scans: taking its lock
        # here (UI thread) froze the window until a slow scan finished.
        self.scanner.reset_requested.set()
        self._log_line(datetime.now().strftime("%H:%M:%S"),
                       "reset requested - applying after the current scan", "info")

    def _show_reasoning(self, _event=None) -> None:
        sel = self.watch.selection()
        if not sel:
            return
        text = self.reasons.get(sel[0])
        if text:
            self._log_line(datetime.now().strftime("%H:%M:%S"),
                           "reasoning for " + sel[0] + ":\n" + text, "dim")

    def _quit(self) -> None:
        self.scanner.stop_flag.set()
        self.scanner.running.clear()
        self.root.destroy()

    # ------------------------------------------------------------ updates
    def _log_line(self, ts: str, text: str, tag: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", ts + "  ", "time")
        self.log.insert("end", text + "\n", tag)
        if int(self.log.index("end-1c").split(".")[0]) > 400:
            self.log.delete("1.0", "50.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _poll(self) -> None:
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg[0] == "log":
                    self._log_line(msg[1], msg[2], msg[3])
                elif msg[0] == "snapshot":
                    self._apply_snapshot(*msg[1:])
        except queue.Empty:
            pass
        self.root.after(200, self._poll)

    def _apply_snapshot(self, rows, positions, stats, reasons) -> None:
        self.reasons = reasons
        for i, (symbol, price, regime, conf, decision) in enumerate(rows):
            tag = "wait"
            if decision.startswith("LONG"):
                tag = "long"
            elif decision.startswith("SHORT"):
                tag = "short"
            tags = (tag, "alt") if i % 2 else (tag,)
            if self.watch.exists(symbol):
                self.watch.item(symbol, values=(symbol, price, regime, conf, decision), tags=tags)

        self.pos.delete(*self.pos.get_children())
        for i, row in enumerate(positions):
            pnl = row[5]
            vals = row[:5] + (f"{pnl:+.2f}",) + row[6:]
            tags = ("win" if pnl >= 0 else "loss",) + (("alt",) if i % 2 else ())
            self.pos.insert("", "end", values=vals, tags=tags)

        self.stat_labels["equity"].configure(text=f"${stats['equity']:,.2f}")
        self.stat_labels["cash"].configure(text=f"${stats['cash']:,.2f}")
        pnl = stats["pnl"]
        self.stat_labels["pnl"].configure(text=f"{pnl:+,.2f}",
                                          fg=GREEN if pnl >= 0 else RED)
        self.stat_labels["trades"].configure(text=str(stats["trades"]))
        wr = stats["winrate"]
        self.stat_labels["winrate"].configure(text=f"{wr:.0%}" if wr is not None else "-")
        self.stat_labels["open"].configure(
            text=f"{stats['open']}/{config.MAX_OPEN_POSITIONS}")


if __name__ == "__main__":
    import sys
    app = App()
    if "--autostart" in sys.argv:
        app._toggle()
    app.root.mainloop()
