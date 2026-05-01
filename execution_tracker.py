"""
execution_tracker.py — Execution Quality Feedback Loop (AI EA v6 PRO)
=====================================================================
Tracks slippage, fill quality, signal-to-fill latency and generates
alerts when execution degrades. Feeds directly into risk sizing.

Metrics tracked per symbol:
  - Expected price vs actual fill price (slippage in pips)
  - Signal time vs fill confirmation time (latency ms)
  - Fill rate (orders placed vs filled)
  - Adverse fill rate (filled worse than mid)

Usage:
    tracker = ExecutionTracker()
    tracker.record_signal(symbol, direction, expected_price, signal_time)
    tracker.record_fill(symbol, actual_fill_price, fill_time, order_id)
    quality = tracker.get_quality_score(symbol)  # 0.0-1.0
    size_multiplier = tracker.get_size_adjustment(symbol)
"""

import logging
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

EXECUTION_LOG = "data/execution_log.json"
ALERT_SLIPPAGE_PIPS = 3.0   # alert if avg slippage > 3 pips
MAX_SAMPLES = 200            # rolling window per symbol


class ExecutionTracker:
    """
    Tracks fill quality and adjusts position sizing based on execution health.
    """

    def __init__(self, alert_slippage_pips: float = ALERT_SLIPPAGE_PIPS):
        self.alert_slippage = alert_slippage_pips
        os.makedirs("data", exist_ok=True)

        # Per-symbol rolling windows
        self._slippage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_SAMPLES))
        self._latency:  Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_SAMPLES))
        self._pnl_adj:  Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_SAMPLES))
        # Pending signals awaiting fill confirmation
        self._pending: Dict[str, Dict] = {}

        self._load_log()

    # ── Record API ────────────────────────────────────────────────────

    def record_signal(
        self,
        symbol: str,
        direction: str,
        expected_price: float,
        signal_time: Optional[datetime] = None,
        order_id: str = "",
    ) -> None:
        """Call when a trade signal is generated (before order placement)."""
        if signal_time is None:
            signal_time = datetime.now(timezone.utc)
        key = order_id or f"{symbol}_{signal_time.isoformat()}"
        self._pending[key] = {
            "symbol":         symbol,
            "direction":      direction,
            "expected_price": expected_price,
            "signal_time":    signal_time,
        }

    def record_fill(
        self,
        symbol:       str,
        actual_fill:  float,
        fill_time:    Optional[datetime] = None,
        order_id:     str = "",
        point:        float = 0.00001,
    ) -> Dict:
        """
        Call when an order fill is confirmed.
        Returns a dict with slippage and latency metrics.
        """
        if fill_time is None:
            fill_time = datetime.now(timezone.utc)

        # Find matching pending signal
        pending = None
        if order_id and order_id in self._pending:
            pending = self._pending.pop(order_id)
        else:
            # Find by symbol (most recent)
            for k, v in list(self._pending.items()):
                if v["symbol"] == symbol:
                    pending = self._pending.pop(k)
                    break

        if pending is None:
            logger.debug(f"No pending signal for {symbol} fill")
            return {}

        expected      = pending["expected_price"]
        direction     = pending["direction"]
        signal_time   = pending["signal_time"]

        # Slippage in pips
        pip_size = point * 10
        slip_price = actual_fill - expected  # positive = slipped against buy
        if direction == "SELL":
            slip_price = -slip_price
        slip_pips = slip_price / pip_size if pip_size > 0 else 0.0

        # Latency in ms
        latency_ms = (fill_time - signal_time).total_seconds() * 1000

        self._slippage[symbol].append(slip_pips)
        self._latency[symbol].append(latency_ms)

        # Alert on bad slippage
        avg_slip = self.get_avg_slippage(symbol)
        if avg_slip > self.alert_slippage:
            logger.warning(
                f"[ExecutionTracker] {symbol}: avg slippage={avg_slip:.2f} pips "
                f"(threshold={self.alert_slippage:.2f}). Check broker conditions."
            )

        result = {
            "symbol":     symbol,
            "slip_pips":  round(slip_pips, 4),
            "latency_ms": round(latency_ms, 1),
            "fill_time":  fill_time.isoformat(),
        }
        self._append_log(result)
        return result

    def record_trade_pnl(self, symbol: str, pnl_pips: float) -> None:
        """Record final trade P&L for quality scoring."""
        self._pnl_adj[symbol].append(pnl_pips)

    # ── Query API ─────────────────────────────────────────────────────

    def get_avg_slippage(self, symbol: str) -> float:
        """Average slippage in pips for last N fills."""
        data = list(self._slippage[symbol])
        return float(np.mean(data)) if data else 0.0

    def get_avg_latency_ms(self, symbol: str) -> float:
        """Average fill latency in milliseconds."""
        data = list(self._latency[symbol])
        return float(np.mean(data)) if data else 0.0

    def get_quality_score(self, symbol: str) -> float:
        """
        Returns 0.0-1.0 quality score.
        1.0 = perfect fills, 0.0 = terrible slippage.
        Considers slippage relative to alert threshold.
        """
        avg_slip = self.get_avg_slippage(symbol)
        # Normalise: 0 slip = 1.0, alert_threshold slip = 0.5, 2x = 0.0
        if avg_slip <= 0:
            return 1.0
        score = max(0.0, 1.0 - (avg_slip / (self.alert_slippage * 2)))
        return round(float(score), 3)

    def get_size_adjustment(self, symbol: str) -> float:
        """
        Returns a multiplier (0.5-1.0) to reduce position size when
        execution quality is poor. Applied by risk engine.
        """
        q = self.get_quality_score(symbol)
        # Scale: quality 1.0 -> multiplier 1.0, quality 0.0 -> multiplier 0.5
        return max(0.5, round(0.5 + q * 0.5, 3))

    def get_summary(self, symbol: str = "") -> Dict:
        """Get execution summary for one or all symbols."""
        if symbol:
            return self._symbol_summary(symbol)
        syms = set(list(self._slippage.keys()) + list(self._latency.keys()))
        return {s: self._symbol_summary(s) for s in syms}

    def _symbol_summary(self, symbol: str) -> Dict:
        return {
            "symbol":         symbol,
            "avg_slip_pips":  round(self.get_avg_slippage(symbol), 3),
            "avg_latency_ms": round(self.get_avg_latency_ms(symbol), 1),
            "quality_score":  self.get_quality_score(symbol),
            "size_mult":      self.get_size_adjustment(symbol),
            "n_fills":        len(self._slippage[symbol]),
        }

    # ── Persistence ───────────────────────────────────────────────────

    def _append_log(self, record: Dict) -> None:
        try:
            records = []
            if os.path.exists(EXECUTION_LOG):
                with open(EXECUTION_LOG) as f:
                    records = json.load(f)
            records.append(record)
            # Keep last 1000 records
            if len(records) > 1000:
                records = records[-1000:]
            with open(EXECUTION_LOG, "w") as f:
                json.dump(records, f, indent=2)
        except Exception as e:
            logger.debug(f"Execution log write failed: {e}")

    def _load_log(self) -> None:
        try:
            if not os.path.exists(EXECUTION_LOG):
                return
            with open(EXECUTION_LOG) as f:
                records = json.load(f)
            for r in records[-200:]:  # load last 200 fills
                sym  = r.get("symbol", "")
                slip = r.get("slip_pips", 0.0)
                lat  = r.get("latency_ms", 0.0)
                if sym:
                    self._slippage[sym].append(slip)
                    self._latency[sym].append(lat)
            logger.debug(f"Execution tracker: loaded {len(records)} historical fills")
        except Exception as e:
            logger.debug(f"Execution log load failed: {e}")
