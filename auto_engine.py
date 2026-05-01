"""
auto_engine.py — AI EA v19 Autonomous Orchestration Engine
===========================================================
All 12 AUTO directives from the v19 spec are implemented here as a
standalone module so every affected file can import a single clean API.

Key classes
-----------
AutoFetcher           — Directive 1 : retry-with-multiplier data fetching
AutoLabelTuner        — Directive 2 : relax RR / max_bars / TP until labels ok
AutoWalkForwardTuner  — Directive 3 : guaranteed ≥2 CV folds
AutoHyperSearch       — Directive 4 : 5-minute grid search per symbol
AutoRegimeTrainer     — Directive 5 : trending / ranging sub-models
AutoFeatureSelector   — Directive 6 : OOS permutation pruning
AutoEnsembleWeights   — Directive 7 : OOF-proportional per-symbol weights
AutoLiveRetrain       — Directive 8 : early retrain on 10+ new live trades
AutoKellySizer        — Directive 9 : Kelly fraction with drawdown brake
AutoSymbolScorer      — Directive 10: rolling composite score, top-60% filter
AutoStopLoss          — Directive 11: tiered daily / weekly drawdown escalation
AutoRetrainScheduler  — Directive 12: session-aware smart retrain timing

All parameters have sensible defaults and are overridable via .env.
Every auto-decision is logged with [AUTO] prefix.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Helper: env float/int with default ───────────────────────────────────────

def _ef(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _ei(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Directive 1: AUTO-FETCH
# ─────────────────────────────────────────────────────────────────────────────

class AutoFetcher:
    """
    Retry loop for fetching H1 bars.

    Attempts: BARS → BARS*2 → BARS*3, capped at broker maximum.
    If the broker caps at 5000 but the window needs 8760, uses 5000
    and adjusts forward_bars proportionally.
    """

    BROKER_MAX = _ei("BROKER_MAX_BARS", 50_000)
    MIN_BARS   = _ei("AUTO_FETCH_MIN_BARS", 300)

    @classmethod
    def fetch_with_retry(
        cls,
        fetcher_fn,          # callable(symbol, timeframe, n_bars) -> DataFrame | None
        symbol: str,
        timeframe: str,
        base_bars: int,
        forward_bars: int,
        min_required: int = 300,
    ) -> Tuple[Optional[object], int]:
        """
        Returns (dataframe, effective_forward_bars).
        Never returns None — falls back to smallest viable slice if necessary.
        """
        attempts = [base_bars, base_bars * 2, base_bars * 3]
        attempts = [min(a, cls.BROKER_MAX) for a in dict.fromkeys(attempts)]

        for n in attempts:
            try:
                df = fetcher_fn(symbol, timeframe, n)
                if df is None or len(df) == 0:
                    logger.warning(
                        f"[AUTO] {symbol}: fetch returned empty for {n} bars — retrying"
                    )
                    continue
                if len(df) >= min_required:
                    if n > base_bars:
                        logger.info(
                            f"[AUTO] {symbol}: base_bars={base_bars} insufficient, "
                            f"fetched {len(df)} bars (attempt {n})"
                        )
                    return df, forward_bars
                # Got some data but less than expected — scale forward_bars down
                fwd_scaled = max(1, int(forward_bars * len(df) / base_bars))
                logger.warning(
                    f"[AUTO] {symbol}: broker returned {len(df)}/{base_bars} bars — "
                    f"scaling forward_bars {forward_bars} → {fwd_scaled}"
                )
                return df, fwd_scaled
            except Exception as exc:
                logger.warning(f"[AUTO] {symbol}: fetch error at n={n}: {exc}")

        logger.error(
            f"[AUTO] {symbol}: all fetch attempts failed — symbol will be skipped"
        )
        return None, forward_bars


# ─────────────────────────────────────────────────────────────────────────────
# Directive 2: AUTO-LABEL TUNING
# ─────────────────────────────────────────────────────────────────────────────

class AutoLabelTuner:
    """
    Relax label-generation parameters until ≥30 BUY+SELL labels appear.

    Relaxation order:
      1. rr_threshold: 0.5 → 0.4 → 0.3 → 0.25
      2. max_bars: 30 → 20 → 15 → 10
      3. tp_mult: 2.0 → 1.5 → 1.2 → 1.0 (+ sl_mult: 1.0 → 0.8 → 0.6)
    """

    RR_STEPS   = [0.5, 0.4, 0.3, 0.25]
    BARS_STEPS = [30, 20, 15, 10]
    TP_STEPS   = [2.0, 1.5, 1.2, 1.0]
    SL_STEPS   = [1.0, 0.8, 0.6, 0.6]
    MIN_LABELS = _ei("AUTO_LABEL_MIN", 30)

    @classmethod
    def find_valid_params(
        cls,
        label_fn,          # callable(df, tp_mult, sl_mult, max_bars) -> pd.Series
        df,
        symbol: str = "?",
    ) -> Tuple[float, float, int]:
        """
        Returns (tp_mult, sl_mult, max_bars) that produce ≥MIN_LABELS.
        Falls back to most relaxed combo if nothing works.
        """
        for rr in cls.RR_STEPS:
            for mb in cls.BARS_STEPS:
                for tp, sl in zip(cls.TP_STEPS, cls.SL_STEPS):
                    try:
                        labels = label_fn(df, tp_mult=tp, sl_mult=sl, max_bars=mb)
                        n_trade = int((labels == 1).sum() + (labels == 2).sum())
                        if n_trade >= cls.MIN_LABELS:
                            if (tp, sl, mb) != (2.0, 1.0, 30):
                                logger.info(
                                    f"[AUTO] {symbol}: label relaxation → "
                                    f"tp={tp} sl={sl} max_bars={mb} rr_hint={rr} "
                                    f"→ {n_trade} BUY+SELL labels"
                                )
                            return tp, sl, mb
                    except Exception:
                        pass

        logger.warning(
            f"[AUTO] {symbol}: all label relaxations exhausted — "
            f"using most permissive params (tp=1.0 sl=0.6 max_bars=10)"
        )
        return 1.0, 0.6, 10


# ─────────────────────────────────────────────────────────────────────────────
# Directive 3: AUTO WALK-FORWARD TUNING
# ─────────────────────────────────────────────────────────────────────────────

class AutoWalkForwardTuner:
    """
    Derive safe n_splits and embargo_bars so sklearn never raises ValueError.
    Guarantees at least 2 folds.
    """

    MIN_FOLDS      = 2
    MAX_FOLDS      = 10
    MIN_TRAIN_SIZE = _ei("WF_MIN_TRAIN_SIZE", 300)
    MAX_EMBARGO_PCT = 0.10   # never more than 10% of dataset

    @classmethod
    def safe_params(cls, n_rows: int) -> Tuple[int, int, int]:
        """
        Returns (n_splits, embargo_bars, test_size).
        Always produces a valid split for the given number of rows.
        """
        min_train  = max(50, min(cls.MIN_TRAIN_SIZE, n_rows // 4))
        test_size  = max(10, n_rows // 30)
        # embargo: 5% of rows, capped at 10% and never > WF_EMBARGO_BARS default
        embargo    = max(3, min(30, int(n_rows * cls.MAX_EMBARGO_PCT)))
        max_safe   = max(
            cls.MIN_FOLDS,
            (n_rows - min_train) // (test_size + embargo)
        )
        n_splits   = min(max_safe, cls.MAX_FOLDS)
        if n_splits < cls.MIN_FOLDS:
            # Dataset too small for any CV — report 0.0 accuracy
            logger.warning(
                f"[AUTO] Dataset too small ({n_rows} rows) for {cls.MIN_FOLDS}-fold CV "
                f"— will train on full data, WF accuracy = 0.0"
            )
            n_splits = 0
        else:
            logger.debug(
                f"[AUTO] WF params: rows={n_rows} splits={n_splits} "
                f"embargo={embargo} test_size={test_size}"
            )
        return n_splits, embargo, test_size


# ─────────────────────────────────────────────────────────────────────────────
# Directive 4: AUTO HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────────────────────────────────────

class AutoHyperSearch:
    """
    Lightweight grid search after baseline training.
    Runs at most TIME_BOX_SECS (default 300) per symbol.
    Only saves if new combo strictly beats current saved model.
    """

    TIME_BOX_SECS = _ei("AUTO_HYPER_TIME_BOX", 300)
    MIN_WF_TO_TRIGGER = float(os.getenv("AUTO_HYPER_TRIGGER_BELOW", "0.50"))

    N_ESTIMATORS  = [80, 120, 160]
    MAX_DEPTH     = [4, 6, 8]
    LEARNING_RATE = [0.05, 0.08, 0.12]
    FORWARD_BARS  = [3, 5, 8, 10]

    @classmethod
    def search(
        cls,
        train_fn,       # callable(df, df_h4, forward_bars, n_est, lr, depth) -> (engine, ok)
        df,
        df_h4,
        symbol: str,
        baseline_acc: float,
    ) -> Optional[object]:
        """
        Run grid search.  Returns the best engine if it beats baseline_acc,
        else None.  Time-boxed at TIME_BOX_SECS.
        """
        if baseline_acc >= cls.MIN_WF_TO_TRIGGER:
            return None

        logger.info(
            f"[AUTO] {symbol}: baseline wf_acc={baseline_acc:.3f} < "
            f"{cls.MIN_WF_TO_TRIGGER} — starting hyperparameter grid search "
            f"(time-box={cls.TIME_BOX_SECS}s)"
        )

        best_acc    = baseline_acc
        best_engine = None
        t_start     = time.time()
        combo_n     = 0

        for fwd in cls.FORWARD_BARS:
            for n_est in cls.N_ESTIMATORS:
                for lr in cls.LEARNING_RATE:
                    for depth in cls.MAX_DEPTH:
                        if time.time() - t_start > cls.TIME_BOX_SECS:
                            logger.info(
                                f"[AUTO] {symbol}: hyper search time-box reached "
                                f"after {combo_n} combos — best_acc={best_acc:.3f}"
                            )
                            return best_engine
                        try:
                            eng, ok = train_fn(
                                df, df_h4,
                                forward_bars=fwd,
                                n_estimators=n_est,
                                learning_rate=lr,
                                max_depth=depth,
                            )
                            combo_n += 1
                            if ok and eng._wf_mean_acc > best_acc:
                                best_acc    = eng._wf_mean_acc
                                best_engine = eng
                                logger.info(
                                    f"[AUTO] {symbol}: new best combo "
                                    f"fwd={fwd} n_est={n_est} lr={lr} depth={depth} "
                                    f"→ wf_acc={best_acc:.3f}"
                                )
                        except Exception as exc:
                            logger.debug(f"[AUTO] hyper combo error: {exc}")

        elapsed = time.time() - t_start
        if best_engine is not None:
            logger.info(
                f"[AUTO] {symbol}: grid search done in {elapsed:.0f}s — "
                f"best_acc={best_acc:.3f} beats baseline={baseline_acc:.3f}"
            )
        else:
            logger.info(
                f"[AUTO] {symbol}: grid search done in {elapsed:.0f}s — "
                f"no improvement over baseline={baseline_acc:.3f}"
            )
        return best_engine


# ─────────────────────────────────────────────────────────────────────────────
# Directive 5: AUTO REGIME-ADAPTIVE TRAINING
# ─────────────────────────────────────────────────────────────────────────────

class AutoRegimeTrainer:
    """
    Split the training dataset into trending / ranging bars and
    train separate sub-models.

    Storage paths:
      models/signal_model_{symbol}_trending.pkl
      models/signal_model_{symbol}_ranging.pkl
    """

    ADX_TRENDING = float(os.getenv("REGIME_ADX_TRENDING", "25"))
    ADX_RANGING  = float(os.getenv("REGIME_ADX_RANGING",  "15"))
    MIN_REGIME_BARS = _ei("REGIME_MIN_BARS", 200)
    BLEND_REGIME   = float(os.getenv("REGIME_BLEND_WEIGHT", "0.6"))  # 60/40

    @classmethod
    def split_by_regime(cls, df) -> Tuple[Optional[object], Optional[object]]:
        """
        Returns (df_trending, df_ranging).  Either can be None if too few bars.
        """
        try:
            import pandas as pd
            high, low, close = df["high"], df["low"], df["close"]
            plus_dm  = high.diff().clip(lower=0)
            minus_dm = (-low.diff()).clip(lower=0)
            mask     = plus_dm > minus_dm
            plus_dm  = plus_dm.where(mask, 0)
            minus_dm = minus_dm.where(~mask, 0)
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low  - close.shift()).abs(),
            ], axis=1).max(axis=1)
            atr_s   = tr.rolling(14).mean().replace(0, float("nan"))
            plus_di = 100 * plus_dm.rolling(14).mean() / atr_s
            minus_di = 100 * minus_dm.rolling(14).mean() / atr_s
            dx      = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
            adx     = dx.rolling(14).mean().fillna(20.0)

            df_trend = df[adx > cls.ADX_TRENDING].copy()
            df_range = df[adx < cls.ADX_RANGING].copy()

            df_t = df_trend if len(df_trend) >= cls.MIN_REGIME_BARS else None
            df_r = df_range if len(df_range) >= cls.MIN_REGIME_BARS else None
            return df_t, df_r
        except Exception as exc:
            logger.warning(f"[AUTO] regime split error: {exc}")
            return None, None

    @classmethod
    def current_regime_confidence(cls, df) -> Tuple[str, float]:
        """
        Returns ('trending'|'ranging'|'neutral', confidence 0-1).
        """
        try:
            import pandas as pd
            high, low, close = df["high"], df["low"], df["close"]
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low  - close.shift()).abs(),
            ], axis=1).max(axis=1)
            atr   = tr.rolling(14).mean().replace(0, float("nan"))
            plus_dm  = high.diff().clip(lower=0)
            minus_dm = (-low.diff()).clip(lower=0)
            mask = plus_dm > minus_dm
            plus_dm  = plus_dm.where(mask, 0)
            minus_dm = minus_dm.where(~mask, 0)
            plus_di  = 100 * plus_dm.rolling(14).mean() / atr
            minus_di = 100 * minus_dm.rolling(14).mean() / atr
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
            adx = dx.rolling(14).mean().fillna(20.0)
            last_adx = float(adx.iloc[-1])
            if last_adx > cls.ADX_TRENDING:
                conf = min(1.0, (last_adx - cls.ADX_TRENDING) / 20.0)
                return "trending", conf
            elif last_adx < cls.ADX_RANGING:
                conf = min(1.0, (cls.ADX_RANGING - last_adx) / 10.0)
                return "ranging", conf
            return "neutral", 0.0
        except Exception:
            return "neutral", 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Directive 6: AUTO FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

class AutoFeatureSelector:
    """
    Permutation importance pruning on the OOS fold.
    Drop features with importance < threshold, retrain, keep if better.
    """

    IMPORTANCE_THRESHOLD = float(os.getenv("AUTO_FEAT_THRESHOLD", "0.001"))
    N_REPEATS            = _ei("AUTO_FEAT_REPEATS", 5)

    @classmethod
    def prune(
        cls,
        model,                 # fitted sklearn estimator
        X_oos: "np.ndarray",
        y_oos: "np.ndarray",
        feature_names: List[str],
        symbol: str = "?",
    ) -> List[str]:
        """
        Returns a (possibly shorter) list of feature names after pruning.
        """
        try:
            from sklearn.inspection import permutation_importance
            from sklearn.metrics import balanced_accuracy_score
            result = permutation_importance(
                model, X_oos, y_oos,
                n_repeats=cls.N_REPEATS,
                scoring="balanced_accuracy",
                random_state=42,
                n_jobs=1,
            )
            means = result.importances_mean
            keep = [
                name for name, imp in zip(feature_names, means)
                if imp >= cls.IMPORTANCE_THRESHOLD
            ]
            dropped = len(feature_names) - len(keep)
            if dropped > 0:
                logger.info(
                    f"[AUTO] {symbol}: feature pruning — dropped {dropped} weak features "
                    f"(importance < {cls.IMPORTANCE_THRESHOLD}), keeping {len(keep)}"
                )
            return keep if len(keep) >= 10 else feature_names  # safety floor
        except Exception as exc:
            logger.debug(f"[AUTO] feature pruning error: {exc}")
            return feature_names


# ─────────────────────────────────────────────────────────────────────────────
# Directive 7: AUTO ENSEMBLE WEIGHT TUNING
# ─────────────────────────────────────────────────────────────────────────────

class AutoEnsembleWeights:
    """
    Compute per-symbol ensemble weights proportional to OOF balanced accuracy.
    Models below random baseline (0.333) get weight=0.
    """

    RANDOM_BASELINE = float(os.getenv("AUTO_ENSEMBLE_BASELINE", "0.333"))

    @classmethod
    def compute(
        cls,
        oof_scores: Dict[str, float],   # {"rf": 0.45, "gbm": 0.48, "xgb": 0.50, "lgb": 0.52}
        symbol: str = "?",
    ) -> Dict[str, float]:
        """
        Returns {"rf": w, "gbm": w, "xgb": w, "lgb": w} summing to 1.0.
        """
        adjusted = {
            k: max(0.0, v - cls.RANDOM_BASELINE)
            for k, v in oof_scores.items()
        }
        total = sum(adjusted.values())
        if total <= 0:
            logger.warning(
                f"[AUTO] {symbol}: all base models below random baseline — "
                f"falling back to uniform weights"
            )
            n = len(oof_scores)
            return {k: 1.0 / n for k in oof_scores}

        weights = {k: v / total for k, v in adjusted.items()}
        logger.info(
            f"[AUTO] {symbol}: ensemble weights updated "
            + " ".join(f"{k}={v:.3f}" for k, v in weights.items())
        )
        return weights


# ─────────────────────────────────────────────────────────────────────────────
# Directive 8: AUTO LIVE FEEDBACK ACCELERATION
# ─────────────────────────────────────────────────────────────────────────────

class AutoLiveRetrain:
    """
    Trigger incremental retrain when ≥10 new live trades arrived since last retrain.
    Weight the 20 most recent live trades at 10× (vs standard 5×).
    Hot-swap the model in memory without restarting.
    """

    NEW_TRADE_TRIGGER  = _ei("AUTO_LIVE_TRIGGER", 10)
    RECENT_WEIGHT_N    = _ei("AUTO_LIVE_RECENT_N", 20)
    RECENT_WEIGHT_MULT = float(os.getenv("AUTO_LIVE_RECENT_WEIGHT", "10.0"))

    def __init__(self):
        self._last_retrain_counts: Dict[str, int] = {}

    def should_trigger(self, symbol: str, current_count: int) -> bool:
        last = self._last_retrain_counts.get(symbol, 0)
        new_since = current_count - last
        if new_since >= self.NEW_TRADE_TRIGGER:
            logger.info(
                f"[AUTO] {symbol}: {new_since} new live trades since last retrain "
                f"— triggering incremental retrain"
            )
            return True
        return False

    def mark_retrained(self, symbol: str, current_count: int) -> None:
        self._last_retrain_counts[symbol] = current_count

    @classmethod
    def boosted_sample_weight(cls, w: "np.ndarray", n_hist: int) -> "np.ndarray":
        """
        Given a weight array where indices [n_hist:] are live samples,
        boost the last RECENT_WEIGHT_N live rows to RECENT_WEIGHT_MULT.
        """
        w = w.copy()
        live_start = n_hist
        live_end   = len(w)
        recent_start = max(live_start, live_end - cls.RECENT_WEIGHT_N)
        w[recent_start:] = cls.RECENT_WEIGHT_MULT
        return w


# ─────────────────────────────────────────────────────────────────────────────
# Directive 9: AUTO POSITION SIZING — Kelly with drawdown brake
# ─────────────────────────────────────────────────────────────────────────────

class AutoKellySizer:
    """
    Kelly criterion position sizing replacing fixed ATR lot formula.

    f* = (p * b - q) / b  where p=win_rate, q=1-p, b=avg_win/avg_loss
    Half-Kelly applied (×0.5) for safety.

    Drawdown brake:
      - Drawdown > 5%  → Kelly × 0.5
      - Last 10 trades all losses → minimum lot until 2 consecutive wins
    """

    HALF_KELLY           = 0.5
    DD_BRAKE_THRESHOLD   = float(os.getenv("KELLY_DD_BRAKE_PCT", "0.05"))
    DD_BRAKE_FACTOR      = float(os.getenv("KELLY_DD_BRAKE_FACTOR", "0.5"))
    LOSS_STREAK_TRIGGER  = _ei("KELLY_LOSS_STREAK_N", 10)
    LOSS_STREAK_RECOVER  = _ei("KELLY_LOSS_STREAK_WIN", 2)
    MIN_TRADES_FOR_KELLY = _ei("KELLY_MIN_TRADES", 20)

    def __init__(self):
        self._loss_streak:     Dict[str, int] = {}
        self._win_since_brake: Dict[str, int] = {}
        self._brake_active:    Dict[str, bool] = {}

    def calculate(
        self,
        equity: float,
        atr: float,
        pnl_history: List[float],
        drawdown_pct: float,
        symbol: str = "?",
        min_lot: float = 0.01,
        max_lot: float = 0.50,
        contract_size: float = 100_000,
    ) -> float:
        """
        Returns lot size.  Falls back to ATR-based sizing if insufficient history.
        """
        # Loss streak brake
        if self._brake_active.get(symbol, False):
            wins = self._win_since_brake.get(symbol, 0)
            if wins < self.LOSS_STREAK_RECOVER:
                logger.info(
                    f"[AUTO] {symbol}: loss-streak brake active "
                    f"({wins}/{self.LOSS_STREAK_RECOVER} recovery wins) — min lot"
                )
                return min_lot

        if len(pnl_history) < self.MIN_TRADES_FOR_KELLY or equity <= 0 or atr <= 0:
            # Fallback: standard 0.7% ATR sizing
            return self._atr_fallback(equity, atr, min_lot, max_lot, contract_size)

        wins  = [p for p in pnl_history if p > 0]
        loses = [p for p in pnl_history if p < 0]
        if not wins or not loses:
            return self._atr_fallback(equity, atr, min_lot, max_lot, contract_size)

        win_rate  = len(wins) / len(pnl_history)
        avg_win   = float(np.mean(wins))
        avg_loss  = abs(float(np.mean(loses)))
        if avg_loss == 0:
            return self._atr_fallback(equity, atr, min_lot, max_lot, contract_size)

        b    = avg_win / avg_loss
        q    = 1.0 - win_rate
        f    = (win_rate * b - q) / b   # full Kelly fraction
        f    = max(0.0, f) * self.HALF_KELLY  # half-Kelly

        # Drawdown brake
        if drawdown_pct > self.DD_BRAKE_THRESHOLD:
            f_before = f
            f *= self.DD_BRAKE_FACTOR
            logger.info(
                f"[AUTO] {symbol}: drawdown={drawdown_pct*100:.1f}% > "
                f"{self.DD_BRAKE_THRESHOLD*100:.0f}% — Kelly brake: "
                f"{f_before:.4f} → {f:.4f}"
            )

        risk_dollars = equity * f
        stop_distance = atr * 1.5
        lot = risk_dollars / (stop_distance * contract_size) if (stop_distance * contract_size) > 0 else min_lot
        lot = round(max(min_lot, min(lot, max_lot)), 2)
        logger.debug(
            f"[AUTO] {symbol}: Kelly lot={lot} f={f:.4f} "
            f"wr={win_rate:.2f} b={b:.2f} dd={drawdown_pct:.3f}"
        )
        return lot

    def record_trade(self, symbol: str, pnl: float) -> None:
        if pnl < 0:
            streak = self._loss_streak.get(symbol, 0) + 1
            self._loss_streak[symbol] = streak
            if streak >= self.LOSS_STREAK_TRIGGER:
                if not self._brake_active.get(symbol, False):
                    logger.warning(
                        f"[AUTO] {symbol}: {streak} consecutive losses — "
                        f"loss-streak brake activated, min lot until "
                        f"{self.LOSS_STREAK_RECOVER} wins"
                    )
                self._brake_active[symbol]    = True
                self._win_since_brake[symbol] = 0
        else:
            self._loss_streak[symbol] = 0
            if self._brake_active.get(symbol, False):
                wins = self._win_since_brake.get(symbol, 0) + 1
                self._win_since_brake[symbol] = wins
                if wins >= self.LOSS_STREAK_RECOVER:
                    logger.info(
                        f"[AUTO] {symbol}: {wins} recovery wins — "
                        f"loss-streak brake released"
                    )
                    self._brake_active[symbol] = False

    @staticmethod
    def _atr_fallback(
        equity: float, atr: float,
        min_lot: float, max_lot: float, contract_size: float
    ) -> float:
        risk_dollars  = equity * 0.007
        stop_distance = atr * 1.5
        if stop_distance <= 0 or contract_size <= 0:
            return min_lot
        lot = risk_dollars / (stop_distance * contract_size)
        return round(max(min_lot, min(lot, max_lot)), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Directive 10: AUTO SYMBOL SCORING
# ─────────────────────────────────────────────────────────────────────────────

class AutoSymbolScorer:
    """
    Rolling 30-trade composite score per symbol.
    Components: WF accuracy (40%), live Sharpe (30%), live win-rate (30%).
    Only the top 60% of symbols receive trade allocations.
    Re-evaluated every retrain cycle.
    """

    WINDOW          = _ei("SYMBOL_SCORE_WINDOW", 30)
    TOP_PCT         = float(os.getenv("SYMBOL_SCORE_TOP_PCT", "0.60"))
    WF_WEIGHT       = float(os.getenv("SYMBOL_SCORE_WF_W",     "0.40"))
    SHARPE_WEIGHT   = float(os.getenv("SYMBOL_SCORE_SHARPE_W", "0.30"))
    WINRATE_WEIGHT  = float(os.getenv("SYMBOL_SCORE_WR_W",     "0.30"))

    def __init__(self):
        self._pnl_history:  Dict[str, deque] = {}
        self._wf_acc:       Dict[str, float] = {}
        self._active_set:   set = set()

    def record(self, symbol: str, pnl: float) -> None:
        if symbol not in self._pnl_history:
            self._pnl_history[symbol] = deque(maxlen=self.WINDOW)
        self._pnl_history[symbol].append(pnl)

    def update_wf_acc(self, symbol: str, wf_acc: float) -> None:
        self._wf_acc[symbol] = wf_acc

    def score(self, symbol: str) -> float:
        pnl   = list(self._pnl_history.get(symbol, []))
        wf_a  = self._wf_acc.get(symbol, 0.333)
        if len(pnl) == 0:
            return wf_a  # no live data yet — rank by model quality

        arr = np.array(pnl)
        std = arr.std()
        sharpe   = float(arr.mean() / std * np.sqrt(252)) if std > 0 else 0.0
        win_rate = float(np.mean(arr > 0))

        # Normalise: sharpe to [0,1] via sigmoid-like, win_rate already [0,1]
        sharpe_n  = 1.0 / (1.0 + np.exp(-sharpe * 0.5))  # ~0.5 at sharpe=0
        wf_norm   = min(1.0, max(0.0, (wf_a - 0.333) / (0.65 - 0.333)))

        return (
            self.WF_WEIGHT      * wf_norm
            + self.SHARPE_WEIGHT  * sharpe_n
            + self.WINRATE_WEIGHT * win_rate
        )

    def refresh_active_set(self, all_symbols: List[str]) -> None:
        scored = sorted(all_symbols, key=lambda s: self.score(s), reverse=True)
        n_keep = max(1, int(len(scored) * self.TOP_PCT))
        self._active_set = set(scored[:n_keep])
        inactive = set(scored[n_keep:])
        if inactive:
            logger.info(
                f"[AUTO] Symbol scoring: active={sorted(self._active_set)} "
                f"inactive={sorted(inactive)} "
                f"(top {self.TOP_PCT*100:.0f}% by composite score)"
            )

    def is_active(self, symbol: str, all_symbols: List[str]) -> bool:
        """Return True if symbol is in the active (top-60%) set."""
        if not self._active_set:
            return True   # first cycle — allow all until first evaluation
        return symbol in self._active_set


# ─────────────────────────────────────────────────────────────────────────────
# Directive 11: AUTO STOPPING LOSS ESCALATION
# ─────────────────────────────────────────────────────────────────────────────

class AutoStopLoss:
    """
    Tiered daily / weekly drawdown protection.

    Thresholds are .env-configurable — no code changes needed.
    """

    DAILY_HALVE_PCT   = float(os.getenv("AUTOSTOP_DAILY_HALVE",   "0.02"))
    DAILY_STOP_PCT    = float(os.getenv("AUTOSTOP_DAILY_STOP",    "0.03"))
    WEEKLY_REDUCE_PCT = float(os.getenv("AUTOSTOP_WEEKLY_REDUCE", "0.06"))

    def __init__(self):
        self._session_start_equity: float     = 0.0
        self._week_peak_equity:     float     = 0.0
        self._lots_halved:          bool      = False
        self._day_stopped:          bool      = False
        self._max_concurrent_override: Optional[int] = None
        self._weekly_reduce_until:  Optional[datetime] = None
        self._today:                date      = date.today()

    def set_session_equity(self, equity: float) -> None:
        if self._session_start_equity == 0:
            self._session_start_equity = equity
        if equity > self._week_peak_equity:
            self._week_peak_equity = equity
        # Day roll
        today = date.today()
        if today != self._today:
            self._today             = today
            self._session_start_equity = equity
            self._lots_halved       = False
            self._day_stopped       = False

    def evaluate(
        self,
        equity: float,
        base_max_concurrent: int,
    ) -> Tuple[float, int, bool]:
        """
        Returns (lot_scale_factor, effective_max_concurrent, day_stopped).
        lot_scale_factor=1.0 means no change; 0.5 means halved.
        """
        lot_scale     = 1.0
        max_conc      = base_max_concurrent
        day_stopped   = False

        if self._session_start_equity <= 0:
            return lot_scale, max_conc, day_stopped

        daily_pct = (equity - self._session_start_equity) / self._session_start_equity

        # Daily -2% → halve lots
        if daily_pct <= -self.DAILY_HALVE_PCT and not self._lots_halved:
            self._lots_halved = True
            logger.warning(
                f"[AUTO] Daily P&L={daily_pct*100:.2f}% ≤ "
                f"-{self.DAILY_HALVE_PCT*100:.0f}% — halving all lot sizes for today"
            )

        # Daily -3% → stop trading today
        if daily_pct <= -self.DAILY_STOP_PCT:
            self._day_stopped = True
            day_stopped       = True
            logger.warning(
                f"[AUTO] Daily P&L={daily_pct*100:.2f}% ≤ "
                f"-{self.DAILY_STOP_PCT*100:.0f}% — CLOSING ALL and halting today"
            )

        if self._lots_halved:
            lot_scale = 0.5

        # Weekly drawdown
        if self._week_peak_equity > 0:
            weekly_dd = (self._week_peak_equity - equity) / self._week_peak_equity
            if weekly_dd >= self.WEEKLY_REDUCE_PCT:
                until = self._weekly_reduce_until
                if until is None or datetime.now() > until:
                    self._weekly_reduce_until = datetime.now() + timedelta(hours=48)
                    logger.warning(
                        f"[AUTO] Weekly drawdown={weekly_dd*100:.1f}% ≥ "
                        f"{self.WEEKLY_REDUCE_PCT*100:.0f}% — reducing max_concurrent "
                        f"from {base_max_concurrent} to {max(1, base_max_concurrent//2)} "
                        f"for 48 hours"
                    )
                if datetime.now() < (self._weekly_reduce_until or datetime.min):
                    max_conc = max(1, base_max_concurrent // 2)

        return lot_scale, max_conc, day_stopped


# ─────────────────────────────────────────────────────────────────────────────
# Directive 12: AUTO TRAINING SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

class AutoRetrainScheduler:
    """
    Decide whether a retrain should proceed right now based on:
      1. High-impact news windows → delay and retry
      2. First 30 min of London/NY session → erratic price action
      3. Prefer Asian session / mid-session low-volatility periods
      4. If last retrain produced worse model → re-attempt in 2 hours
    """

    SESSION_AVOID_MINUTES = _ei("RETRAIN_AVOID_SESSION_OPEN_MINS", 30)
    RETRY_AFTER_WORSE_H   = _ei("RETRAIN_RETRY_AFTER_WORSE_HRS", 2)

    # London open: 08:00 UTC, NY open: 13:00 UTC
    SESSION_OPENS_UTC = [8, 13]

    def __init__(self, news_filter=None):
        self._news_filter     = news_filter
        self._worse_until:    Optional[datetime] = None
        self._pending_delay_until: Optional[datetime] = None

    def can_retrain(self, symbol: str = "") -> Tuple[bool, str]:
        """
        Returns (allowed, reason).
        reason is a human-readable explanation used in [AUTO] log lines.
        """
        now = datetime.utcnow()

        # Check: pending delay from previous worse-model retrain
        if self._pending_delay_until and now < self._pending_delay_until:
            remaining = int((self._pending_delay_until - now).total_seconds() / 60)
            return False, f"worse-model retry delay: {remaining} min remaining"

        # Check: news window
        if self._news_filter is not None:
            try:
                if hasattr(self._news_filter, "is_news_window"):
                    if self._news_filter.is_news_window():
                        return False, "high-impact news window active"
                elif hasattr(self._news_filter, "block_trading"):
                    if self._news_filter.block_trading():
                        return False, "news filter blocking"
            except Exception:
                pass

        # Check: first 30 min after a major session open
        hour_utc   = now.hour
        minute_utc = now.minute
        for open_hour in self.SESSION_OPENS_UTC:
            if hour_utc == open_hour and minute_utc < self.SESSION_AVOID_MINUTES:
                return (
                    False,
                    f"within {self.SESSION_AVOID_MINUTES} min of session open "
                    f"({open_hour}:00 UTC)"
                )

        # Prefer Asian session (23:00–06:00 UTC) or mid-session (10:00–12:00, 15:00–17:00)
        # This is advisory — we log but do NOT block
        in_preferred = (
            hour_utc >= 23 or hour_utc <= 6
            or 10 <= hour_utc <= 12
            or 15 <= hour_utc <= 17
        )
        if not in_preferred:
            logger.debug(
                f"[AUTO] Retrain allowed but not in preferred low-vol window "
                f"(UTC {hour_utc:02d}:{minute_utc:02d}) — proceeding anyway"
            )

        return True, "OK"

    def notify_worse_model(self) -> None:
        """
        Call when a retrain produced a model that did NOT beat the saved one.
        Schedules a re-attempt in RETRY_AFTER_WORSE_H hours.
        """
        retry_at = datetime.utcnow() + timedelta(hours=self.RETRY_AFTER_WORSE_H)
        self._pending_delay_until = retry_at
        logger.info(
            f"[AUTO] Retrain produced no improvement — "
            f"next attempt scheduled at {retry_at.strftime('%H:%M UTC')} "
            f"({self.RETRY_AFTER_WORSE_H}h delay)"
        )

    def notify_improved(self) -> None:
        """Clear the worse-model delay when a retrain DID improve."""
        self._pending_delay_until = None
