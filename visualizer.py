"""
visualizer.py — Live trading dashboard (AI EA v4)

Runs in a background daemon thread.
Updates every N seconds with:
  - Live candlestick price chart with entry/SL/TP markers
  - Equity curve
  - Trade history table
  - Risk status panel

Uses matplotlib (offline, no server needed).
"""

import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless operation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_DIR    = "logs/charts"
REFRESH_SECS  = 30
MAX_CANDLES   = 100
MAX_TRADES    = 200


class TradingVisualizer:
    """
    Thread-safe live visualiser.

    Usage
    -----
    viz = TradingVisualizer()
    viz.start()                              # starts background thread
    viz.update_price_data(symbol, df)        # call from main loop
    viz.add_trade(trade_record)              # call on every trade
    viz.update_equity(equity_value)          # call every cycle
    viz.update_risk_status(risk_dict)        # call every cycle
    viz.stop()
    """

    def __init__(self, refresh_secs: int = REFRESH_SECS):
        self.refresh_secs = refresh_secs
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Shared state (protected by lock)
        self._price_data:  Dict[str, pd.DataFrame] = {}
        self._trades:      deque = deque(maxlen=MAX_TRADES)
        self._equity:      deque = deque(maxlen=5000)
        self._risk_status: Dict  = {}
        self._active_symbol: str = ""

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Thread control
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        logger.info("TradingVisualizer started.")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("TradingVisualizer stopped.")

    # ------------------------------------------------------------------
    # Data feeds (called from main trading loop)
    # ------------------------------------------------------------------

    def update_price_data(self, symbol: str, df: pd.DataFrame) -> None:
        with self._lock:
            self._price_data[symbol] = df.tail(MAX_CANDLES).copy()
            self._active_symbol = symbol

    def add_trade(self, trade: Dict) -> None:
        with self._lock:
            self._trades.append({**trade, "_added": datetime.now().isoformat()})

    def update_equity(self, equity: float) -> None:
        with self._lock:
            self._equity.append({"time": datetime.now().isoformat(), "equity": equity})

    def update_risk_status(self, risk: Dict) -> None:
        with self._lock:
            self._risk_status = dict(risk)

    # ------------------------------------------------------------------
    # Render loop
    # ------------------------------------------------------------------

    def _render_loop(self) -> None:
        while self._running:
            try:
                self._render()
            except Exception as e:
                logger.error(f"Visualizer render error: {e}", exc_info=True)
            time.sleep(self.refresh_secs)

    def _render(self) -> None:
        with self._lock:
            price_data  = dict(self._price_data)
            trades      = list(self._trades)
            equity_data = list(self._equity)
            risk        = dict(self._risk_status)
            active_sym  = self._active_symbol

        if not price_data:
            return

        # Pick the most recently updated symbol for the price chart
        symbol = active_sym if active_sym in price_data else next(iter(price_data))
        df = price_data[symbol]

        fig = plt.figure(figsize=(18, 12), facecolor="#0d1117")
        fig.suptitle(
            f"AI EA v4 — {symbol}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            color="white", fontsize=13, fontweight="bold",
        )

        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        # ── 1. Candlestick price chart ────────────────────────────────
        ax_price = fig.add_subplot(gs[0:2, 0:2])
        self._draw_candlesticks(ax_price, df, symbol, trades)

        # ── 2. Equity curve ───────────────────────────────────────────
        ax_eq = fig.add_subplot(gs[2, 0:2])
        self._draw_equity_curve(ax_eq, equity_data)

        # ── 3. RSI sub-chart ──────────────────────────────────────────
        ax_rsi = fig.add_subplot(gs[0, 2])
        self._draw_rsi(ax_rsi, df)

        # ── 4. Risk status panel ──────────────────────────────────────
        ax_risk = fig.add_subplot(gs[1, 2])
        self._draw_risk_panel(ax_risk, risk)

        # ── 5. Recent trades table ────────────────────────────────────
        ax_trades = fig.add_subplot(gs[2, 2])
        self._draw_trade_table(ax_trades, trades[-10:])

        # Save
        fname = os.path.join(OUTPUT_DIR, f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(fname, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        # Keep only the last 5 charts to save disk space
        self._prune_charts()
        logger.debug(f"Chart saved: {fname}")

    # ------------------------------------------------------------------
    # Sub-plot renderers
    # ------------------------------------------------------------------

    def _draw_candlesticks(self, ax, df: pd.DataFrame, symbol: str, trades: List[Dict]) -> None:
        ax.set_facecolor("#161b22")
        ax.set_title(f"{symbol} Price", color="white", fontsize=10)

        x = np.arange(len(df))
        o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values

        for i in range(len(df)):
            colour = "#26a641" if c[i] >= o[i] else "#f85149"
            # Body
            ax.bar(x[i], abs(c[i] - o[i]), bottom=min(o[i], c[i]),
                   color=colour, width=0.7, linewidth=0)
            # Wick
            ax.plot([x[i], x[i]], [l[i], h[i]], color=colour, linewidth=0.7)

        # Trade markers
        idx_map = {str(t): i for i, t in enumerate(df.index)}
        for trade in trades[-20:]:
            ts = trade.get("timestamp", "")
            # Find closest bar index
            if "price" in trade and "type" in trade:
                ix = len(df) - 5  # approximate if no timestamp match
                colour = "#58a6ff" if trade["type"] == "buy" else "#f78166"
                marker = "^" if trade["type"] == "buy" else "v"
                ax.scatter(ix, trade["price"], color=colour, marker=marker,
                           s=80, zorder=5)
                if "sl" in trade:
                    ax.axhline(trade["sl"], color="#f85149", linewidth=0.5,
                               linestyle="--", alpha=0.6)
                if "tp" in trade:
                    ax.axhline(trade["tp"], color="#26a641", linewidth=0.5,
                               linestyle="--", alpha=0.6)

        ax.tick_params(colors="gray"); ax.spines[:].set_color("#30363d")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color("gray")

    def _draw_equity_curve(self, ax, equity_data: List[Dict]) -> None:
        ax.set_facecolor("#161b22")
        ax.set_title("Equity Curve", color="white", fontsize=10)
        if not equity_data:
            ax.text(0.5, 0.5, "No equity data", color="gray",
                    ha="center", transform=ax.transAxes)
            return
        eq = [d["equity"] for d in equity_data]
        x  = range(len(eq))
        colour = "#26a641" if eq[-1] >= eq[0] else "#f85149"
        ax.plot(x, eq, color=colour, linewidth=1.5)
        ax.fill_between(x, eq[0], eq, alpha=0.15, color=colour)
        ax.axhline(eq[0], color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Equity ($)", color="gray", fontsize=8)
        ax.tick_params(colors="gray"); ax.spines[:].set_color("#30363d")

    def _draw_rsi(self, ax, df: pd.DataFrame) -> None:
        ax.set_facecolor("#161b22")
        ax.set_title("RSI (14)", color="white", fontsize=10)
        if "rsi" not in df.columns:
            ax.text(0.5, 0.5, "RSI N/A", color="gray", ha="center", transform=ax.transAxes)
            return
        rsi = df["rsi"].values
        x   = np.arange(len(rsi))
        ax.plot(x, rsi, color="#58a6ff", linewidth=1.2)
        ax.axhline(70, color="#f85149", linewidth=0.7, linestyle="--")
        ax.axhline(30, color="#26a641", linewidth=0.7, linestyle="--")
        ax.fill_between(x, 70, rsi, where=(rsi > 70), alpha=0.2, color="#f85149")
        ax.fill_between(x, 30, rsi, where=(rsi < 30), alpha=0.2, color="#26a641")
        ax.set_ylim(0, 100)
        ax.tick_params(colors="gray"); ax.spines[:].set_color("#30363d")

    def _draw_risk_panel(self, ax, risk: Dict) -> None:
        ax.set_facecolor("#161b22")
        ax.set_title("Risk Status", color="white", fontsize=10)
        ax.axis("off")
        if not risk:
            ax.text(0.5, 0.5, "No risk data", color="gray",
                    ha="center", transform=ax.transAxes)
            return

        lines = [
            ("Emergency Stop", str(risk.get("emergency_stop", "?")),
             "#f85149" if risk.get("emergency_stop") else "#26a641"),
            ("Daily Trades",
             f"{risk.get('daily_trades',0)}/{risk.get('max_trades_day',10)}", "white"),
            ("Daily P&L",
             f"${risk.get('daily_pnl',0):.2f}  ({risk.get('daily_loss_pct',0):.2f}%)",
             "#f85149" if risk.get("daily_pnl", 0) < 0 else "#26a641"),
            ("Drawdown",
             f"{risk.get('drawdown_pct',0):.2f}% / {risk.get('max_drawdown_pct',8):.0f}%",
             "#f85149" if risk.get("drawdown_pct", 0) > 5 else "white"),
            ("Prop Mode", str(risk.get("prop_mode", False)),
             "#58a6ff" if risk.get("prop_mode") else "gray"),
            ("Open Positions", str(risk.get("open_positions", 0)), "white"),
        ]
        y = 0.92
        for label, value, colour in lines:
            ax.text(0.05, y, f"{label}:", color="gray", fontsize=8, transform=ax.transAxes)
            ax.text(0.65, y, value, color=colour, fontsize=8,
                    fontweight="bold", transform=ax.transAxes)
            y -= 0.15

    def _draw_trade_table(self, ax, trades: List[Dict]) -> None:
        ax.set_facecolor("#161b22")
        ax.set_title("Recent Trades", color="white", fontsize=10)
        ax.axis("off")
        if not trades:
            ax.text(0.5, 0.5, "No trades yet", color="gray",
                    ha="center", transform=ax.transAxes)
            return
        headers = ["Symbol", "Type", "P&L"]
        rows = []
        for t in reversed(trades[-8:]):
            sym  = t.get("symbol", "—")
            typ  = t.get("type", "—").upper()
            pnl  = t.get("profit", "—")
            pnl_s = f"${pnl:.2f}" if isinstance(pnl, (int, float)) else "open"
            rows.append([sym, typ, pnl_s])

        tbl = ax.table(
            cellText=rows, colLabels=headers,
            cellLoc="center", loc="center",
            bbox=[0, 0, 1, 1],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor("#0d1117")
            cell.set_edgecolor("#30363d")
            txt = cell.get_text().get_text()
            colour = "#f85149" if txt.startswith("$-") else \
                     "#26a641" if txt.startswith("$") else "white"
            cell.get_text().set_color(colour)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def _prune_charts(self, keep: int = 5) -> None:
        """Keep only the N most recent chart files."""
        try:
            files = sorted(
                [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR)
                 if f.endswith(".png")],
                key=os.path.getmtime,
            )
            for old in files[:-keep]:
                os.remove(old)
        except Exception:
            pass
