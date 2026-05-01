"""
logger_system.py — Structured logging system (AI EA v4)
--------------------------------------------------------
Logs all system events to:
  - logs/trades.jsonl      (newline-delimited JSON, one record per trade)
  - logs/signals.jsonl     (ML signal predictions)
  - logs/errors.jsonl      (errors & exceptions)
  - logs/risk_events.jsonl (risk limit breaches)
  - logs/trades.csv        (trade summary in CSV for spreadsheet import)
  - logs/predictions.csv   (signal history CSV)

All writes are thread-safe.
"""

import csv
import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Directory & file paths ────────────────────────────────────────────────────
LOG_DIR = "logs"
TRADE_JSONL      = os.path.join(LOG_DIR, "trades.jsonl")
SIGNAL_JSONL     = os.path.join(LOG_DIR, "signals.jsonl")
ERROR_JSONL      = os.path.join(LOG_DIR, "errors.jsonl")
RISK_JSONL       = os.path.join(LOG_DIR, "risk_events.jsonl")
TRADE_CSV        = os.path.join(LOG_DIR, "trades.csv")
PREDICTION_CSV   = os.path.join(LOG_DIR, "predictions.csv")

TRADE_CSV_FIELDS = [
    "timestamp", "symbol", "type", "lot", "price", "sl", "tp",
    "spread_pips", "signal_prob", "ticket", "atr", "profit", "equity_after",
]
PREDICTION_CSV_FIELDS = [
    "timestamp", "symbol", "signal", "probability",
    "buy_prob", "sell_prob", "no_trade_prob",
    "structure_trend", "structure_score", "score_total",
]


class StructuredLogger:
    """
    Central structured logging hub.

    Usage
    -----
    slog = StructuredLogger()
    slog.log_trade(record_dict)
    slog.log_signal(signal_dict)
    slog.log_error(context, exception)
    slog.log_risk_event(event_dict)
    """

    def __init__(self):
        self._lock = threading.Lock()
        os.makedirs(LOG_DIR, exist_ok=True)
        self._ensure_csv_headers()

    # ── Public API ────────────────────────────────────────────────────────────

    def log_trade(self, record: Dict[str, Any]) -> None:
        """Log a trade open/close event."""
        record = self._stamp(record)
        self._append_jsonl(TRADE_JSONL, record)
        self._append_csv(TRADE_CSV, TRADE_CSV_FIELDS, record)
        logger.info(
            f"[TRADE] {record.get('type','?').upper()} {record.get('symbol','?')} "
            f"lot={record.get('lot','?')} price={record.get('price','?')} "
            f"ticket={record.get('ticket','?')}"
        )

    def log_signal(self, record: Dict[str, Any]) -> None:
        """Log an ML signal prediction."""
        record = self._stamp(record)
        self._append_jsonl(SIGNAL_JSONL, record)
        self._append_csv(PREDICTION_CSV, PREDICTION_CSV_FIELDS, record)
        logger.debug(
            f"[SIGNAL] {record.get('symbol','?')} → {record.get('signal','?')} "
            f"p={record.get('probability', 0):.3f}"
        )

    def log_error(self, context: str, exc: Optional[Exception] = None,
                  extra: Optional[Dict] = None) -> None:
        """Log an error or exception."""
        record: Dict[str, Any] = {"context": context}
        if exc:
            record["exception"] = str(exc)
            record["exc_type"] = type(exc).__name__
        if extra:
            record.update(extra)
        record = self._stamp(record)
        self._append_jsonl(ERROR_JSONL, record)
        logger.error(f"[ERROR] {context}: {exc}")

    def log_risk_event(self, record: Dict[str, Any]) -> None:
        """Log a risk limit event (breach, emergency stop, cooldown, etc.)."""
        record = self._stamp(record)
        self._append_jsonl(RISK_JSONL, record)
        logger.warning(
            f"[RISK] {record.get('event','?')} — {record.get('reason','')}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _stamp(record: Dict) -> Dict:
        if "timestamp" not in record:
            record["timestamp"] = datetime.now().isoformat()
        return record

    def _append_jsonl(self, path: str, record: Dict) -> None:
        try:
            with self._lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.error(f"JSONL write failed ({path}): {e}")

    def _append_csv(self, path: str, fields: list, record: Dict) -> None:
        try:
            with self._lock:
                file_exists = os.path.exists(path)
                with open(path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow({k: record.get(k, "") for k in fields})
        except Exception as e:
            logger.error(f"CSV write failed ({path}): {e}")

    def _ensure_csv_headers(self) -> None:
        """Write CSV headers if files don't exist yet."""
        for path, fields in [(TRADE_CSV, TRADE_CSV_FIELDS),
                              (PREDICTION_CSV, PREDICTION_CSV_FIELDS)]:
            if not os.path.exists(path):
                try:
                    with open(path, "w", newline="", encoding="utf-8") as f:
                        csv.DictWriter(f, fieldnames=fields).writeheader()
                except Exception as e:
                    logger.error(f"CSV header init failed ({path}): {e}")


# Module-level singleton
_slog: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    """Return the module-level singleton StructuredLogger."""
    global _slog
    if _slog is None:
        _slog = StructuredLogger()
    return _slog
