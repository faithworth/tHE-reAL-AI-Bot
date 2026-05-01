"""
performance_monitor.py — Live Sharpe/Sortino/Drawdown Monitor (AI EA v6 PRO)
=============================================================================
Tracks per-symbol trade P&L and computes rolling Sharpe, Sortino,
win rate, and max drawdown. Automatically reduces lot sizes when
a symbol's metrics degrade, and escalates to emergency stop if needed.

Usage:
    pm = PerformanceMonitor()
    pm.record_trade("EURUSD", pnl=25.50)
    metrics = pm.get_metrics("EURUSD")
    size_mult = pm.get_size_multiplier("EURUSD")
"""

import logging
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

PERF_LOG     = "data/performance_monitor.json"
ROLLING_N    = 50     # rolling window for metrics
MIN_TRADES   = 15     # minimum trades before applying degradation controls
SHARPE_FLOOR = -0.5   # below this -> reduce size to 50%
SHARPE_WARN  = 0.3    # below this -> reduce size to 75%
SORTINO_FLOOR = -0.3
WIN_RATE_FLOOR = 0.38  # below 38% win rate -> reduce size
MAX_DD_HALT   = 0.15   # halt symbol if rolling DD > 15% of starting equity


class SymbolMetrics:
    def __init__(self, symbol: str):
        self.symbol  = symbol
        self.pnl:    deque = deque(maxlen=ROLLING_N)
        self.equity: deque = deque(maxlen=500)
        self.wins:   deque = deque(maxlen=ROLLING_N)
        self.peak_equity = 0.0
        self.total_trades = 0

    def add_trade(self, pnl: float, equity: float = 0.0) -> None:
        self.pnl.append(pnl)
        self.wins.append(1 if pnl > 0 else 0)
        self.total_trades += 1
        if equity > 0:
            self.equity.append(equity)
            if equity > self.peak_equity:
                self.peak_equity = equity

    def sharpe(self) -> float:
        if len(self.pnl) < 5:
            return 0.0
        arr = np.array(self.pnl)
        std = arr.std()
        if std == 0:
            return 0.0
        return float(arr.mean() / std * np.sqrt(252 / len(arr)))

    def sortino(self) -> float:
        if len(self.pnl) < 5:
            return 0.0
        arr = np.array(self.pnl)
        down = arr[arr < 0]
        if len(down) == 0:
            return 2.0
        downstd = down.std()
        if downstd == 0:
            return 0.0
        return float(arr.mean() / downstd * np.sqrt(252 / len(arr)))

    def win_rate(self) -> float:
        if not self.wins:
            return 0.5
        return float(np.mean(self.wins))

    def rolling_drawdown(self) -> float:
        """Max drawdown over rolling equity window."""
        if len(self.equity) < 5:
            return 0.0
        eq = np.array(self.equity)
        peak = np.maximum.accumulate(eq)
        dd   = (peak - eq) / (peak + 1e-10)
        return float(dd.max())

    def to_dict(self) -> Dict:
        return {
            "symbol":       self.symbol,
            "sharpe":       round(self.sharpe(), 3),
            "sortino":      round(self.sortino(), 3),
            "win_rate":     round(self.win_rate(), 3),
            "rolling_dd":   round(self.rolling_drawdown(), 3),
            "total_trades": self.total_trades,
            "n_rolling":    len(self.pnl),
        }


class PerformanceMonitor:
    """
    Per-symbol rolling performance tracker with automated size reduction.
    """

    def __init__(self):
        os.makedirs("data", exist_ok=True)
        self._symbols: Dict[str, SymbolMetrics] = {}
        self._load()

    def record_trade(
        self,
        symbol:  str,
        pnl:     float,
        equity:  float = 0.0,
    ) -> None:
        if symbol not in self._symbols:
            self._symbols[symbol] = SymbolMetrics(symbol)
        self._symbols[symbol].add_trade(pnl, equity)
        self._save()

        m = self._symbols[symbol]
        if m.total_trades >= MIN_TRADES:
            sharpe = m.sharpe()
            wr     = m.win_rate()
            dd     = m.rolling_drawdown()
            if sharpe < SHARPE_FLOOR:
                logger.warning(f"[PerfMonitor] {symbol}: Sharpe={sharpe:.3f} below floor "
                               f"{SHARPE_FLOOR} -- size reduced to 50%")
            if wr < WIN_RATE_FLOOR:
                logger.warning(f"[PerfMonitor] {symbol}: WinRate={wr:.1%} below floor "
                               f"{WIN_RATE_FLOOR:.0%} -- size reduced")
            if dd > MAX_DD_HALT:
                logger.error(f"[PerfMonitor] {symbol}: Rolling DD={dd:.1%} exceeded "
                             f"MAX_DD_HALT={MAX_DD_HALT:.0%} -- HALTING SYMBOL")

    def get_metrics(self, symbol: str) -> Dict:
        if symbol not in self._symbols:
            return {"symbol": symbol, "sharpe": 0.0, "sortino": 0.0,
                    "win_rate": 0.5, "rolling_dd": 0.0, "total_trades": 0}
        return self._symbols[symbol].to_dict()

    def get_size_multiplier(self, symbol: str) -> float:
        """
        Returns 0.0-1.0 position size multiplier based on recent performance.
          1.0 = full size, 0.0 = halt trading this symbol.
        """
        if symbol not in self._symbols:
            return 1.0
        m = self._symbols[symbol]
        if m.total_trades < MIN_TRADES:
            return 1.0

        sharpe = m.sharpe()
        sortino = m.sortino()
        wr      = m.win_rate()
        dd      = m.rolling_drawdown()

        # Hard halt
        if dd > MAX_DD_HALT:
            return 0.0

        mult = 1.0

        if sharpe < SHARPE_FLOOR:
            mult = min(mult, 0.50)
        elif sharpe < SHARPE_WARN:
            mult = min(mult, 0.75)

        if sortino < SORTINO_FLOOR:
            mult = min(mult, 0.60)

        if wr < WIN_RATE_FLOOR:
            mult = min(mult, 0.70)

        return round(mult, 2)

    def is_symbol_halted(self, symbol: str) -> bool:
        return self.get_size_multiplier(symbol) == 0.0

    def get_all_metrics(self) -> Dict[str, Dict]:
        return {sym: m.to_dict() for sym, m in self._symbols.items()}

    def _save(self) -> None:
        try:
            data = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "symbols": {sym: {
                    "pnl": list(m.pnl),
                    "equity": list(m.equity),
                    "wins": list(m.wins),
                    "total_trades": m.total_trades,
                    "peak_equity": m.peak_equity,
                } for sym, m in self._symbols.items()}
            }
            with open(PERF_LOG, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"PerfMonitor save failed: {e}")

    def _load(self) -> None:
        try:
            if not os.path.exists(PERF_LOG):
                return
            with open(PERF_LOG) as f:
                data = json.load(f)
            for sym, state in data.get("symbols", {}).items():
                m = SymbolMetrics(sym)
                for p in state.get("pnl", []):
                    m.pnl.append(p)
                for e in state.get("equity", []):
                    m.equity.append(e)
                for w in state.get("wins", []):
                    m.wins.append(w)
                m.total_trades = state.get("total_trades", len(m.pnl))
                m.peak_equity  = state.get("peak_equity", 0.0)
                self._symbols[sym] = m
            logger.info(f"PerfMonitor: loaded data for {len(self._symbols)} symbols")
        except Exception as e:
            logger.debug(f"PerfMonitor load failed: {e}")
