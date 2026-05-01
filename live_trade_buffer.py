"""
live_trade_buffer.py — Persistent live-trade learning buffer (AI EA v17+)
=========================================================================
Purpose
-------
Every real trade the EA takes produces one ground-truth labelled sample that
is far more valuable than a simulated historical bar because:
  - it reflects CURRENT regime, spread, and slippage
  - the outcome (pnl) is unambiguous
  - the model *actually chose* to take that trade (selection-bias aware)

This module provides:
  LiveTradeBuffer   — thread-safe persistent store (jsonlines on disk)
  record_live_trade — one-call capture at position close
  load_as_dataframe — reload all past trades as a weighted DataFrame ready
                      for blending into the next retrain

Design
------
Storage   : {MODEL_DIR}/live_trades_{symbol}.jsonl  (one JSON obj per line)
Format    : { "ts": ISO, "symbol": str, "direction": int (0=sell,1=buy),
              "features": {col: val, ...}, "label": int (0/1/2),
              "pnl": float, "prob": float, "score": float }
Thread-safe: a per-symbol file-lock (threading.Lock) protects concurrent writes.
Durability : append-only — no record is ever overwritten; old file survives crashes.
Max size   : LIVE_BUFFER_MAX_ROWS per symbol (oldest rows trimmed when exceeded).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR           = "models"
LIVE_BUFFER_MAX_ROWS = 2_000   # per symbol — ~6 months of daily trading
LIVE_SAMPLE_WEIGHT   = 5.0    # weight multiplier vs a historical bar
MIN_LIVE_FOR_BLEND   = 20     # don't blend until we have this many real trades

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Per-symbol threading locks ────────────────────────────────────────────────
_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)


def _file_path(symbol: str) -> str:
    safe = symbol.replace("/", "_").replace(".", "_")
    return os.path.join(MODEL_DIR, f"live_trades_{safe}.jsonl")


# ── Core API ──────────────────────────────────────────────────────────────────

def record_live_trade(
    symbol: str,
    direction: str,        # "BUY" or "SELL"
    features: Dict[str, float],   # raw feature dict from _build_features last row
    pnl: float,
    prob: float = 0.0,
    score: float = 0.0,
) -> None:
    """
    Append one completed live trade to the on-disk buffer.

    Parameters
    ----------
    symbol    : trading symbol, e.g. "EURUSD"
    direction : "BUY" or "SELL"
    features  : feature dict captured at entry (from SignalEngine._build_features)
    pnl       : realised P&L in account currency (positive = win)
    prob      : model confidence at entry
    score     : composite score at entry
    """
    # Derive label from direction (not from pnl — we want the entered signal class)
    # Label encoding must match signal_engine: 0=NO_TRADE, 1=BUY, 2=SELL
    label = 1 if direction.upper() == "BUY" else 2

    record = {
        "ts":        datetime.utcnow().isoformat(),
        "symbol":    symbol,
        "direction": direction.upper(),
        "label":     label,
        "pnl":       float(pnl),
        "prob":      float(prob),
        "score":     float(score),
        "features":  {k: float(v) for k, v in features.items()
                      if v is not None and np.isfinite(float(v))},
    }

    path = _file_path(symbol)
    with _locks[symbol]:
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            logger.info(
                f"[LiveBuffer] {symbol}: recorded {direction} pnl=${pnl:.2f} "
                f"prob={prob:.3f} → {path}"
            )
            _trim_if_needed(symbol, path)
        except Exception as exc:
            logger.error(f"[LiveBuffer] write error for {symbol}: {exc}")


def _trim_if_needed(symbol: str, path: str) -> None:
    """Keep the buffer within LIVE_BUFFER_MAX_ROWS (called under lock)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        if len(lines) > LIVE_BUFFER_MAX_ROWS:
            keep = lines[-LIVE_BUFFER_MAX_ROWS:]
            with open(path, "w", encoding="utf-8") as fh:
                fh.writelines(keep)
            logger.debug(
                f"[LiveBuffer] {symbol}: trimmed to {len(keep)} rows"
            )
    except Exception as exc:
        logger.warning(f"[LiveBuffer] trim error for {symbol}: {exc}")


def count_live_trades(symbol: str) -> int:
    """Return how many live trade records exist for this symbol."""
    path = _file_path(symbol)
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except Exception:
        return 0


def load_as_dataframe(
    symbol: str,
    feature_columns: Optional[List[str]] = None,
    min_records: int = MIN_LIVE_FOR_BLEND,
) -> Optional[pd.DataFrame]:
    """
    Load the live trade buffer and return a DataFrame ready for blending.

    Columns: all feature columns present in the buffer PLUS 'label', 'weight'.
    The 'weight' column = LIVE_SAMPLE_WEIGHT (higher than historical 1.0).

    Returns None if fewer than min_records are available.
    """
    path = _file_path(symbol)
    if not os.path.exists(path):
        return None

    rows: List[dict] = []
    with _locks[symbol]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as exc:
            logger.error(f"[LiveBuffer] load error for {symbol}: {exc}")
            return None

    if len(rows) < min_records:
        logger.debug(
            f"[LiveBuffer] {symbol}: only {len(rows)} trades "
            f"(need {min_records}) — skipping blend"
        )
        return None

    # Flatten features into columns
    records = []
    for row in rows:
        flat = {"label": row["label"], "weight": LIVE_SAMPLE_WEIGHT}
        flat.update(row.get("features", {}))
        records.append(flat)

    df = pd.DataFrame(records)

    # Align to the expected feature columns if provided
    if feature_columns:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0.0           # fill missing features with zero
        df = df[feature_columns + ["label", "weight"]]

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    logger.info(
        f"[LiveBuffer] {symbol}: loaded {len(df)} live trades for blending "
        f"(weight={LIVE_SAMPLE_WEIGHT}x)"
    )
    return df


class LiveTradeBuffer:
    """
    Convenience wrapper so SignalEngine can hold a single buffer object
    and not deal with the module-level API directly.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def record(
        self,
        direction: str,
        features: Dict[str, float],
        pnl: float,
        prob: float = 0.0,
        score: float = 0.0,
    ) -> None:
        record_live_trade(
            symbol=self.symbol,
            direction=direction,
            features=features,
            pnl=pnl,
            prob=prob,
            score=score,
        )

    def count(self) -> int:
        return count_live_trades(self.symbol)

    def as_dataframe(
        self,
        feature_columns: Optional[List[str]] = None,
        min_records: int = MIN_LIVE_FOR_BLEND,
    ) -> Optional[pd.DataFrame]:
        return load_as_dataframe(
            self.symbol,
            feature_columns=feature_columns,
            min_records=min_records,
        )
