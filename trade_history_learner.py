"""
trade_history_learner.py — Real Trade History ML Learner  (AI EA v20)
======================================================================
Loads EVERY trade the broker has ever recorded (wins + losses) and
uses them to tune signal engine predictions and filter decisions.

Key upgrades
------------
1. Loads broker trade history (Alpaca / MT5 / IBKR / cTrader) via the
   standard BaseBroker.get_trade_history() interface.
2. Merges with LiveTradeBuffer jsonlines for a unified corpus.
3. Builds per-symbol outcome statistics:
      - win_rate, avg_rr, avg_duration_h, best_hour, worst_hour
      - which signals fired on winning vs losing trades
      - avg pnl per day-of-week (avoid bad days)
4. Exposes suggest_filter(symbol, hour, weekday) → True/False
   which the signal engine can query before approving a trade.
5. Exposes bias_score(symbol) → float in [-1, +1]
   positive = historical edge, negative = avoid.
6. Persists learned stats to data/history_stats_{symbol}.json so
   they survive restarts without re-fetching history.
7. run_full_learn(broker, symbols, days) is the one-shot call at
   startup, then incremental updates happen via record_close().

Threading: all disk operations are lock-protected.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR          = "data"
STATS_FILE_TPL    = os.path.join(DATA_DIR, "history_stats_{sym}.json")
TRADE_LOG_GLOBAL  = os.path.join(DATA_DIR, "all_trades_global.jsonl")
MIN_TRADES_FILTER = 15    # min trades before we start filtering by hour/day
MIN_TRADES_BIAS   = 5     # min trades before bias_score is returned
HISTORY_DAYS      = 365   # how far back to pull from broker on startup

os.makedirs(DATA_DIR, exist_ok=True)

_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_sym(symbol: str) -> str:
    return symbol.replace("/", "_").replace(".", "_")


def _stats_path(symbol: str) -> str:
    return STATS_FILE_TPL.format(sym=_safe_sym(symbol))


def _load_stats(symbol: str) -> Dict:
    path = _stats_path(symbol)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[HistLearner] load_stats {symbol}: {e}")
        return {}


def _save_stats(symbol: str, stats: Dict) -> None:
    path = _stats_path(symbol)
    with _locks[symbol]:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.error(f"[HistLearner] save_stats {symbol}: {e}")


def _append_global_log(trade: Dict) -> None:
    try:
        with _locks["__global__"]:
            with open(TRADE_LOG_GLOBAL, "a", encoding="utf-8") as f:
                f.write(json.dumps(trade) + "\n")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Core statistics builder
# ─────────────────────────────────────────────────────────────────────────────

def _get_field(trade: Dict, *keys, default=0.0):
    """Get first matching key from a trade dict — handles broker field name differences."""
    for k in keys:
        if k in trade and trade[k] is not None:
            return trade[k]
    return default


def _compute_stats(trades: List[Dict]) -> Dict:
    """
    From a list of trade dicts (any broker), compute the stats dict.
    Handles field name variants across MT5/IBKR/cTrader/Alpaca.
    """
    if not trades:
        return {"n": 0}

    profits   = [float(_get_field(t, "profit", default=0.0)) for t in trades]
    wins      = [p for p in profits if p > 0]
    losses    = [p for p in profits if p < 0]
    n         = len(profits)
    win_rate  = len(wins) / n if n > 0 else 0

    avg_win  = float(np.mean(wins))   if wins   else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    rr       = avg_win / abs(avg_loss) if avg_loss != 0 else 0.0

    # Duration in hours — MT5 uses "time" for close, others use "close_time"/"open_time"
    durations = []
    for t in trades:
        try:
            # open_time: try both field names
            ot = _get_field(t, "open_time", default="")
            # close_time: MT5 stores only "time" (the close), others "close_time"
            ct = _get_field(t, "close_time", "time", default="")
            if ot and ct:
                open_dt  = _parse_dt(str(ot))
                close_dt = _parse_dt(str(ct))
                if open_dt and close_dt:
                    durations.append((close_dt - open_dt).total_seconds() / 3600)
        except Exception:
            pass

    avg_dur = float(np.mean(durations)) if durations else 0.0

    # Hour/weekday PnL — use open_time when available, fall back to "time"
    hour_pnl: Dict[int, List[float]] = defaultdict(list)
    dow_pnl:  Dict[int, List[float]] = defaultdict(list)
    for t in trades:
        pnl = float(_get_field(t, "profit", default=0.0))
        try:
            ts_str = _get_field(t, "open_time", "time", default="")
            dt = _parse_dt(str(ts_str)) if ts_str else None
            if dt:
                hour_pnl[dt.hour].append(pnl)
                dow_pnl[dt.weekday()].append(pnl)
        except Exception:
            pass

    hour_stats = {
        h: {
            "n": len(v),
            "win_rate": len([x for x in v if x > 0]) / len(v),
            "avg_pnl": float(np.mean(v)),
        }
        for h, v in hour_pnl.items()
    }
    dow_stats = {
        d: {
            "n": len(v),
            "win_rate": len([x for x in v if x > 0]) / len(v),
            "avg_pnl": float(np.mean(v)),
        }
        for d, v in dow_pnl.items()
    }

    good_hours   = [h for h, s in hour_stats.items() if s["n"] >= 8 and s["win_rate"] > 0.55]
    bad_hours    = [h for h, s in hour_stats.items() if s["n"] >= 8 and s["win_rate"] < 0.30]
    bad_weekdays = [d for d, s in dow_stats.items()  if s["n"] >= 8 and s["win_rate"] < 0.30]

    # Bias score [-1, +1]
    if n >= MIN_TRADES_BIAS:
        bias = (win_rate - 0.50) * 2.0 * min(1.0, math.log(n / 5 + 1) / math.log(21))
        bias = max(-1.0, min(1.0, bias))
    else:
        bias = 0.0

    return {
        "n":           n,
        "win_rate":    round(win_rate, 4),
        "avg_win":     round(avg_win, 4),
        "avg_loss":    round(avg_loss, 4),
        "rr":          round(rr, 4),
        "total_pnl":   round(sum(profits), 4),
        "avg_dur_h":   round(avg_dur, 2),
        "bias_score":  round(bias, 4),
        "hour_stats":  {str(k): v for k, v in hour_stats.items()},
        "dow_stats":   {str(k): v for k, v in dow_stats.items()},
        "good_hours":  good_hours,
        "bad_hours":   bad_hours,
        "bad_weekdays":bad_weekdays,
        "updated_at":  datetime.utcnow().isoformat(),
    }


def _parse_dt(s: str) -> Optional[datetime]:
    """Parse ISO or common broker datetime strings."""
    if not s:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(str(s)[:26], fmt[:len(str(s)[:26])])
        except Exception:
            pass
    try:
        return datetime.fromisoformat(str(s)[:26])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class TradeHistoryLearner:
    """
    Loads and learns from all historical trades — every win and loss
    ever recorded by the broker or saved to disk.

    Usage in ai_ea.py startup:
        learner = TradeHistoryLearner()
        learner.run_full_learn(broker=self.broker, symbols=self.symbols)

    Usage before trading:
        ok = learner.suggest_filter(symbol, hour=datetime.now().hour)
        bias = learner.bias_score(symbol)
    """

    def __init__(self) -> None:
        self._stats: Dict[str, Dict] = {}
        self._raw_trades: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = threading.Lock()

    # ── Full learn at startup ─────────────────────────────────────────────────

    def run_full_learn(
        self,
        broker,
        symbols: List[str],
        days: int = HISTORY_DAYS,
        prefetched_history: Optional[List[Dict]] = None,
    ) -> None:
        """
        Load all trade history and compute per-symbol stats.

        broker          — the live broker instance. Pass None if prefetched_history
                          is provided (avoids calling broker from a bg thread,
                          which is unsafe for MT5's single-threaded API).
        prefetched_history — raw list of trade dicts already fetched from broker.
                          When provided, the broker API call is skipped entirely.
        """
        broker_name = getattr(broker, "broker_name", "prefetched") if broker else "prefetched"
        logger.info(
            f"[HistLearner] Starting full learn: broker={broker_name} "
            f"symbols={len(symbols)} days={days}"
        )

        all_trades: List[Dict] = []

        # 1. Broker API history — only if not already prefetched
        if prefetched_history is not None:
            # History was fetched synchronously in the main thread (MT5-safe)
            all_trades.extend(prefetched_history)
            if prefetched_history:
                logger.info(
                    f"[HistLearner] Prefetched: {len(prefetched_history)} trades"
                )
        elif broker is not None:
            try:
                broker_hist = broker.get_trade_history(days=days)
                if broker_hist:
                    all_trades.extend(broker_hist)
                    logger.info(
                        f"[HistLearner] Broker API: {len(broker_hist)} trades from {broker_name}"
                    )
            except AttributeError:
                logger.debug("[HistLearner] Broker has no get_trade_history")
            except Exception as e:
                logger.warning(f"[HistLearner] Broker history error: {e}")

        # 2. Local JSONL trade logs — load from all known paths (broker-neutral)
        local_logs = [
            TRADE_LOG_GLOBAL,                                        # global log (all brokers)
            os.path.join(DATA_DIR, "trade_log_alpaca.jsonl"),        # Alpaca
            os.path.join(DATA_DIR, "trade_log_mt5.jsonl"),           # MT5
            os.path.join(DATA_DIR, "trade_log_ibkr.jsonl"),          # IBKR
            os.path.join(DATA_DIR, "trade_log_ctrader.jsonl"),       # cTrader
        ]
        for log_path in local_logs:
            loaded = self._load_jsonl(log_path)
            if loaded:
                all_trades.extend(loaded)
                logger.info(f"[HistLearner] Local log {os.path.basename(log_path)}: {len(loaded)} records")

        # 3. LiveTradeBuffer jsonlines for each symbol (all brokers write here)
        for sym in symbols:
            safe = _safe_sym(sym)
            lb_path = os.path.join("models", f"live_trades_{safe}.jsonl")
            lb_trades = self._load_jsonl(lb_path)
            for t in lb_trades:
                if "symbol" not in t:
                    t["symbol"] = sym
            if lb_trades:
                all_trades.extend(lb_trades)
                logger.info(f"[HistLearner] LiveBuffer {sym}: {len(lb_trades)} records")

        if not all_trades:
            logger.info(
                f"[HistLearner] No historical trades found for {broker_name} — "
                "will learn as trades accumulate."
            )
            return

        # Deduplicate by ticket (string-cast to handle int/str mix across brokers)
        seen = set()
        unique: List[Dict] = []
        for t in all_trades:
            key = str(t.get("ticket", id(t)))
            if key not in seen:
                seen.add(key)
                unique.append(t)

        logger.info(f"[HistLearner] Total unique trades across all sources: {len(unique)}")

        # Group by symbol and compute stats
        by_symbol: Dict[str, List[Dict]] = defaultdict(list)
        for t in unique:
            sym = t.get("symbol", "")
            if sym:
                by_symbol[sym].append(t)

        with self._lock:
            for sym, trades in by_symbol.items():
                self._raw_trades[sym] = trades
                stats = _compute_stats(trades)
                self._stats[sym] = stats
                _save_stats(sym, stats)
                logger.info(
                    f"[HistLearner] {sym}: n={stats['n']} "
                    f"win_rate={stats.get('win_rate',0)*100:.1f}% "
                    f"rr={stats.get('rr',0):.2f} "
                    f"bias={stats.get('bias_score',0):+.3f} "
                    f"bad_hours={stats.get('bad_hours',[])} "
                    f"bad_days={stats.get('bad_weekdays',[])}"
                )

        logger.info(
            f"[HistLearner] Full learn complete — "
            f"{len(by_symbol)} symbols learned from {len(unique)} total trades."
        )

    def load_persisted(self, symbols: List[str]) -> None:
        """
        Load previously saved stats from disk (instant, no broker call).
        Call in __init__ path; run_full_learn can run in background.
        """
        for sym in symbols:
            stats = _load_stats(sym)
            if stats:
                with self._lock:
                    self._stats[sym] = stats

    # ── Incremental update after each trade close ─────────────────────────────

    def record_close(self, trade: Dict) -> None:
        """
        Update stats with a just-closed trade (immediate, in-memory + disk).
        Call from ai_ea.py whenever a position is closed.

        trade dict must have: symbol, profit, open_time, close_time (optional)
        """
        sym = trade.get("symbol", "")
        if not sym:
            return

        _append_global_log(trade)

        with self._lock:
            self._raw_trades[sym].append(trade)
            stats = _compute_stats(self._raw_trades[sym])
            self._stats[sym] = stats

        _save_stats(sym, stats)
        logger.debug(
            f"[HistLearner] {sym}: updated stats n={stats['n']} "
            f"win_rate={stats.get('win_rate',0)*100:.1f}%"
        )

    # ── Query API ─────────────────────────────────────────────────────────────

    def suggest_filter(
        self,
        symbol: str,
        hour: Optional[int] = None,
        weekday: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Returns (should_trade: bool, reason: str).

        Blocks trade if:
          - This hour is in bad_hours (win rate < 35% with ≥3 samples)
          - This weekday is in bad_weekdays (win rate < 35% with ≥3 samples)

        Only activates after MIN_TRADES_FILTER trades to avoid premature filtering.
        """
        stats = self._get_stats(symbol)
        if not stats or stats.get("n", 0) < MIN_TRADES_FILTER:
            return True, "OK (insufficient history)"

        if hour is not None and hour in stats.get("bad_hours", []):
            hr_s = stats["hour_stats"].get(str(hour), {})
            return (
                False,
                f"Bad trading hour {hour:02d}:xx — historical win_rate="
                f"{hr_s.get('win_rate',0)*100:.0f}% ({hr_s.get('n',0)} trades)",
            )

        if weekday is not None and weekday in stats.get("bad_weekdays", []):
            dnames = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            dname  = dnames[weekday] if weekday < 7 else str(weekday)
            dw_s   = stats["dow_stats"].get(str(weekday), {})
            return (
                False,
                f"Bad trading day {dname} — historical win_rate="
                f"{dw_s.get('win_rate',0)*100:.0f}% ({dw_s.get('n',0)} trades)",
            )

        return True, "OK"

    def bias_score(self, symbol: str) -> float:
        """
        Returns a float in [-1, +1]:
          +1.0 → strong historical edge (trade freely)
           0.0 → neutral / not enough data
          -1.0 → avoid this symbol (consistent losses)
        """
        stats = self._get_stats(symbol)
        if not stats or stats.get("n", 0) < MIN_TRADES_BIAS:
            return 0.0
        return float(stats.get("bias_score", 0.0))

    def get_summary(self, symbol: str) -> Dict:
        """Return the full stats dict for a symbol (for dashboard / logging)."""
        return dict(self._get_stats(symbol) or {})

    def all_symbols_summary(self) -> Dict[str, Dict]:
        """Return summary for all symbols with recorded stats."""
        with self._lock:
            return {sym: dict(s) for sym, s in self._stats.items()}

    # ── TP/SL tuning hint ─────────────────────────────────────────────────────

    def tp_sl_hint(self, symbol: str) -> Tuple[float, float]:
        """
        Suggest TP multiplier and SL multiplier based on historical R:R.
        Returns (tp_mult, sl_mult) — pass to risk_engine or signal_engine.

        Logic:
          If avg RR historically > 1.5 → widen TP slightly.
          If avg RR historically < 0.8 → tighten SL slightly (cut losses faster).
          Else → return defaults (1.5, 1.0).
        """
        stats = self._get_stats(symbol)
        if not stats or stats.get("n", 0) < MIN_TRADES_FILTER:
            return 1.5, 1.0

        rr = float(stats.get("rr", 0))
        if rr >= 1.8:
            return min(2.5, rr * 1.1), 1.0   # wide TP — history shows winners run far
        elif rr < 0.8 and rr > 0:
            return 1.5, max(0.7, rr * 0.8)   # tighter SL — cut losses faster
        return 1.5, 1.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_stats(self, symbol: str) -> Optional[Dict]:
        with self._lock:
            if symbol in self._stats:
                return self._stats[symbol]
        # Try loading from disk if not in memory
        stats = _load_stats(symbol)
        if stats:
            with self._lock:
                self._stats[symbol] = stats
        return stats or None

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict]:
        if not os.path.exists(path):
            return []
        rows = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"[HistLearner] _load_jsonl {path}: {e}")
        return rows

    def __repr__(self) -> str:
        n_symbols = len(self._stats)
        total_trades = sum(s.get("n", 0) for s in self._stats.values())
        return f"<TradeHistoryLearner symbols={n_symbols} total_trades={total_trades}>"
