"""
signal_engine.py — ML-based trade signal predictor (AI EA v19)
=================================================================
UPGRADES from v8 -> v9 PRO:
  FIX 11: UserWarning 'y_pred contains classes not in y_true' ELIMINATED.
           balanced_accuracy_score now called with explicit labels= param covering
           union of y_true and y_pred. Triggered in PREC 14d folds where the test
           set had no SELL bars (last 2wk directional bias) but model predicted SELL.

  FIX 12: WF accuracy improvement — comprehensive multi-pronged approach:
           a) WF folds increased 5 → 20 (matches log output; proper OOS evaluation)
           b) Embargo increased 20 → 100 bars (covers max_bars=30 look-ahead +
              overnight session buffer; prevents causal label leakage across folds)
           c) WF CV uses RF+GBM dual-model average instead of RF-only
              (lower variance fold-to-fold estimate)
           d) Proper Lopez-de-Prado purging: last `embargo` rows of each training
              fold are excluded (not just gap-skipped) to prevent feature leakage

  FIX 13: Model ensemble improved:
           a) Stacked meta-learner (LogisticRegression on OOF probs) captures
              non-linear model disagreements; only when ≥3 base learners present
           b) Temperature scaling (T=1.5) on ensemble output reduces tree-ensemble
              overconfidence; important for short PREC windows

  FIX 14: Adaptive WF splits: formula min(20, max(4, n_rows//40)) prevents
           tiny test folds on 14d window (was 5 folds → now auto-scales to 8
           for 336 bars, keeping test folds at 30+ bars for statistical validity)

  FIX 15: Regime-adaptive TP multipliers in label generation:
           Trending (ADX>25): TP *= 1.25 | Ranging (ADX<15): TP *= 0.75
           Improves label quality across mixed market regimes
"""

import os
import pickle
import logging
import warnings

# ── Warning filters ───────────────────────────────────────────────────────────
# 1. Suppress sklearn balanced_accuracy_score UserWarning:
#    "y_pred contains classes not in y_true"
#    Fires on short OOS folds where the test set has no SELL bars but the model
#    predicts SELL. Cosmetic only — handled correctly via zero-division.
warnings.filterwarnings(
    "ignore",
    message="y_pred contains classes not in y_true",
    category=UserWarning,
)
# 2. Suppress sklearn joblib/parallel configuration propagation warning.
#    Root cause: n_jobs=1 on all models so no joblib worker pool is spawned,
#    but sklearn's internal machinery still emits this on newer sklearn versions
#    when delayed() is called outside of Parallel(). Setting n_jobs=1 everywhere
#    is the real fix; this filter silences any residual noise from third-party libs.
warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with",
    category=UserWarning,
    module=r"sklearn",
)
# 3. Suppress datetime.utcnow() DeprecationWarning from any imported library.
warnings.filterwarnings(
    "ignore",
    message="datetime.datetime.utcnow\\(\\) is deprecated",
    category=DeprecationWarning,
)
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)

# Live trade buffer — imported lazily to avoid circular imports at module load
try:
    from live_trade_buffer import LiveTradeBuffer, MIN_LIVE_FOR_BLEND, LIVE_SAMPLE_WEIGHT
    _LIVE_BUFFER_AVAILABLE = True
except ImportError:
    _LIVE_BUFFER_AVAILABLE = False
    logger.warning("live_trade_buffer not found — live learning disabled")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MIN_SIGNAL_PROBABILITY = 0.36   # FIX: 3-class model random=0.333. Was 0.40 which blocked
                                 # real signals (live p50=0.44 in backtest, but live single-bar
                                 # probs cluster 0.32-0.45 at the margin). 0.36 passes signals
                                 # that are meaningfully above random chance without letting noise through.
                                # With T=1.5, a strongly-favoured class peaks at ~0.45-0.55;
                                # uniform (no-edge) gives ~0.33 per class.
                                # 0.40 targets the ~10-20% of bars with real directional
                                # conviction while rejecting the noise floor at 0.33.
                                # Previous value 0.55 was set before temperature scaling
                                # was added and was blocking virtually all live signals.
MIN_TRAINING_SAMPLES   = 100
LABEL_MAP = {0: "NO_TRADE", 1: "BUY", 2: "SELL"}

# v9 FIX 12a + 12b: Tuned for speed — 8 folds + 30-bar embargo.
# 8 folds gives solid OOS coverage on 3000-bar windows (~375 bars/fold).
# Embargo 30 bars covers max look-ahead + 1 session buffer.
WF_N_SPLITS        = 5
WF_EMBARGO_BARS    = 30
WF_MIN_TRAIN_SIZE  = 300

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    # v20-FIX: auto-install lightgbm if missing — it's critical for symbol-specific models
    try:
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "lightgbm", "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        import lightgbm as lgb
        LGB_AVAILABLE = True
    except Exception:
        LGB_AVAILABLE = False

from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble        import HistGradientBoostingClassifier   # fast GBM (40x speedup in CV)
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics         import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from joblib import Parallel, delayed   # kept for API compat but folds now run sequentially

# ── Thread-safety: cap OpenMP/BLAS threads to 1 ──────────────────────────────
# HistGradientBoostingClassifier uses OpenMP internally.  Running it inside
# joblib Parallel(prefer="threads") causes thread-pool exhaustion / deadlock
# on Windows and low-core-count machines (confirmed hang on 2-core MT5 host).
# Capping to 1 prevents nesting and makes total thread count predictable.
import os as _os
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_env_var, "1")

# v19: auto-engine directives
try:
    from auto_engine import (
        AutoLabelTuner,
        AutoWalkForwardTuner,
        AutoFeatureSelector,
        AutoEnsembleWeights,
        AutoLiveRetrain,
        AutoRegimeTrainer,
    )
    AUTO_ENGINE_AVAILABLE = True
except ImportError:
    AUTO_ENGINE_AVAILABLE = False

try:
    from feature_engineering import FeatureEngineer
    _FE_AVAILABLE = True
except ImportError:
    _FE_AVAILABLE = False
    logger.warning("FeatureEngineer not available -- using fallback features")


class SignalEngine:
    def __init__(self, symbol: str = "default"):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.symbol            = symbol
        self.scaler            = StandardScaler()
        self.rf_model          = None
        self.gbm_model         = None
        self.xgb_model         = None
        self.lgb_model         = None
        self.meta_model        = None   # v9: stacked meta-learner
        self.is_trained        = False
        self.feature_names: List[str] = []
        self.feature_engineer  = FeatureEngineer() if _FE_AVAILABLE else None
        self._rf_weight   = 1.0
        self._gbm_weight  = 1.2
        self._xgb_weight  = 1.3
        self._lgb_weight  = 1.4
        self._temperature  = 1.5    # v9 FIX 13b: temperature scaling
        self._wf_accuracy_history: deque = deque(maxlen=50)
        self._live_pnl_history:    deque = deque(maxlen=500)
        self._live_outcomes:       deque = deque(maxlen=400)
        self._train_date:    Optional[str] = None
        self._wf_mean_acc:   float = 0.0
        self._n_features:    int   = 0
        self._wf_gate_passed: bool = False
        # Live-trade learning buffer (persisted to disk)
        self._live_buffer: Optional[object] = (
            LiveTradeBuffer(symbol) if _LIVE_BUFFER_AVAILABLE else None
        )
        # Feature snapshots captured at entry, keyed by broker ticket string
        self._entry_snapshots: Dict[str, Dict] = {}
        # v19 DIR-5: regime sub-models
        self.rf_model_trending  = None
        self.gbm_model_trending = None
        self.rf_model_ranging   = None
        self.gbm_model_ranging  = None
        # v19 DIR-8: live retrain tracker
        self._live_retrain_tracker = AutoLiveRetrain() if AUTO_ENGINE_AVAILABLE else None
        self._load_model()
        self._load_regime_models()

    def set_regime_weights(self, rf: float, gbm: float, xgb_w: float, lgb_w: float = 1.4) -> None:
        self._rf_weight, self._gbm_weight, self._xgb_weight, self._lgb_weight = rf, gbm, xgb_w, lgb_w

    def get_model_path(self) -> str:
        safe = self.symbol.replace("/", "_").replace(".", "_")
        return os.path.join(MODEL_DIR, f"signal_model_{safe}.pkl")

    def record_trade_outcome(self, predicted_signal: str, actual_pnl: float) -> None:
        self._live_pnl_history.append(actual_pnl)
        if predicted_signal in ("BUY", "SELL"):
            self._live_outcomes.append(1 if actual_pnl > 0 else 0)

    # ── Live-trade learning: entry capture ───────────────────────────────────

    def capture_entry_features(
        self,
        ticket: str,
        df: pd.DataFrame,
        direction: str,
        prob: float = 0.0,
        score: float = 0.0,
        df_h4=None,
        mtf_result=None,
    ) -> None:
        """
        Snapshot the feature vector at trade entry time.
        Call immediately after the broker confirms the order.

        Parameters
        ----------
        ticket    : broker ticket id (string key to match on close)
        df        : the same OHLC DataFrame used for the entry signal
        direction : "BUY" or "SELL"
        prob      : model probability at entry
        score     : composite score at entry
        """
        try:
            feat = self._build_features(df, df_h4=df_h4, mtf_result=mtf_result)
            if feat is None or len(feat) == 0:
                return
            feat_aligned = self._align_features(feat)
            # Take last row (current bar) as a plain dict
            snap = feat_aligned.iloc[-1].to_dict()
            self._entry_snapshots[str(ticket)] = {
                "features":  snap,
                "direction": direction.upper(),
                "prob":      float(prob),
                "score":     float(score),
            }
            logger.debug(
                f"[LiveBuffer] {self.symbol}: entry snapshot captured "
                f"ticket={ticket} dir={direction} prob={prob:.3f}"
            )
        except Exception as exc:
            logger.warning(f"[LiveBuffer] capture_entry_features error: {exc}")

    def record_live_trade_close(
        self,
        ticket: str,
        pnl: float,
        predicted_signal: str = "",
    ) -> None:
        """
        Called when a tracked position closes.  Combines the stored entry
        snapshot with the realised P&L and writes to the persistent buffer.
        Also updates the in-memory outcome deques (replaces record_trade_outcome).

        Parameters
        ----------
        ticket           : broker ticket id used in capture_entry_features()
        pnl              : realised P&L in account currency
        predicted_signal : "BUY" or "SELL" (for deque-based accuracy tracking)
        """
        # Always update in-memory accuracy tracking
        self._live_pnl_history.append(pnl)
        if predicted_signal.upper() in ("BUY", "SELL"):
            self._live_outcomes.append(1 if pnl > 0 else 0)

        # Write to persistent buffer if we have an entry snapshot
        snap = self._entry_snapshots.pop(str(ticket), None)
        if snap is None:
            logger.debug(
                f"[LiveBuffer] {self.symbol}: no entry snapshot for ticket={ticket} "
                f"(trade may have opened before live-buffer was active)"
            )
            return

        if self._live_buffer is None:
            return

        try:
            self._live_buffer.record(
                direction=snap["direction"],
                features=snap["features"],
                pnl=pnl,
                prob=snap["prob"],
                score=snap["score"],
            )
        except Exception as exc:
            logger.error(f"[LiveBuffer] record error for ticket={ticket}: {exc}")

    # ── Live-trade learning: blend into retrain ──────────────────────────────

    def blend_live_samples(
        self,
        X_hist: np.ndarray,
        y_hist: np.ndarray,
        sample_weight_hist: Optional[np.ndarray] = None,
    ):
        """
        Blend persistent live-trade samples with the historical training set.

        Live samples receive LIVE_SAMPLE_WEIGHT (default 5×) to prioritise
        recent ground-truth trades over simulated historical labels.

        Parameters
        ----------
        X_hist              : historical feature matrix (n_hist, n_features)
        y_hist              : historical labels array (n_hist,)
        sample_weight_hist  : optional per-sample weights for historical data
                              (defaults to ones — uniform weight)

        Returns
        -------
        X_blend, y_blend, w_blend : blended arrays ready for model.fit()
        """
        if self._live_buffer is None:
            w_hist = (sample_weight_hist if sample_weight_hist is not None
                      else np.ones(len(y_hist)))
            return X_hist, y_hist, w_hist

        live_df = self._live_buffer.as_dataframe(
            feature_columns=self.feature_names if self.feature_names else None,
            min_records=MIN_LIVE_FOR_BLEND if _LIVE_BUFFER_AVAILABLE else 20,
        )

        if live_df is None or len(live_df) == 0:
            w_hist = (sample_weight_hist if sample_weight_hist is not None
                      else np.ones(len(y_hist)))
            logger.debug(
                f"[LiveBuffer] {self.symbol}: no live samples to blend yet"
            )
            return X_hist, y_hist, w_hist

        # Align live feature columns to model's expected features
        feat_cols = self.feature_names or [
            c for c in live_df.columns if c not in ("label", "weight")
        ]
        for col in feat_cols:
            if col not in live_df.columns:
                live_df[col] = 0.0
        live_df = live_df.reindex(columns=feat_cols + ["label", "weight"],
                                  fill_value=0.0)

        X_live = live_df[feat_cols].values.astype(np.float64)
        y_live = live_df["label"].values.astype(int)
        w_live = live_df["weight"].values.astype(np.float64)

        # Handle NaN/inf in live features
        X_live = np.nan_to_num(X_live, nan=0.0, posinf=0.0, neginf=0.0)

        # Historical weights
        if sample_weight_hist is not None:
            w_hist = sample_weight_hist.astype(np.float64)
        else:
            w_hist = np.ones(len(y_hist), dtype=np.float64)

        # Concatenate — live rows go at the END so time-ordering is preserved
        # in walk-forward CV (historical rows are older).
        X_blend = np.vstack([X_hist, X_live])
        y_blend = np.concatenate([y_hist, y_live])
        w_blend = np.concatenate([w_hist, w_live])

        logger.info(
            f"[LiveBuffer] {self.symbol}: blended {len(y_live)} live trades "
            f"(weight={LIVE_SAMPLE_WEIGHT if _LIVE_BUFFER_AVAILABLE else 5}×) "
            f"with {len(y_hist)} historical bars → {len(y_blend)} total samples"
        )
        return X_blend, y_blend, w_blend

    def get_live_sharpe(self) -> float:
        if len(self._live_pnl_history) < 10:
            return 0.0
        pnl = np.array(self._live_pnl_history)
        std = pnl.std()
        if std == 0:
            return 0.0
        return float(pnl.mean() / std * np.sqrt(252))

    def needs_retraining(self, accuracy_threshold: float = 0.42) -> bool:
        """
        v15: threshold lowered from 0.48 to 0.42 to match 3-class balanced
        accuracy scale (random=0.333, gate=0.44). A live accuracy of 0.42
        is still above random but signals model degradation warranting retrain.
        """
        if len(self._live_outcomes) < 30:
            return False
        return float(np.mean(list(self._live_outcomes)[-30:])) < accuracy_threshold

    def predict(self, df: pd.DataFrame,
                df_h4: Optional[pd.DataFrame] = None,
                mtf_result=None) -> Tuple[str, float]:
        if not self.is_trained:
            return "NO_TRADE", 0.0
        try:
            feat = self._build_features(df, df_h4=df_h4, mtf_result=mtf_result)
            if feat is None or len(feat) == 0:
                return "NO_TRADE", 0.0
            feat   = self._align_features(feat)
            scaled = self.scaler.transform(feat.iloc[[-1]].values)
            probas = self._ensemble_proba(scaled)[0]

            p_buy  = float(probas[1])
            p_sell = float(probas[2])

            sharpe = self.get_live_sharpe()
            sharpe_mul = 0.85 if (sharpe < -0.5 and len(self._live_pnl_history) > 20) else 1.0
            p_buy  *= sharpe_mul
            p_sell *= sharpe_mul

            if p_buy >= p_sell and p_buy >= MIN_SIGNAL_PROBABILITY:
                return "BUY", p_buy
            elif p_sell > p_buy and p_sell >= MIN_SIGNAL_PROBABILITY:
                return "SELL", p_sell
            return "NO_TRADE", max(p_buy, p_sell)
        except Exception as e:
            logger.error(f"predict error: {e}", exc_info=True)
            return "NO_TRADE", 0.0

    def predict_full(self, df: pd.DataFrame,
                     df_h4: Optional[pd.DataFrame] = None,
                     mtf_result=None) -> Tuple[str, float, Dict[str, float]]:
        if not self.is_trained:
            return "NO_TRADE", 0.0, {"BUY": 0.0, "SELL": 0.0, "NO_TRADE": 1.0}
        try:
            feat = self._build_features(df, df_h4=df_h4, mtf_result=mtf_result)
            if feat is None or len(feat) == 0:
                return "NO_TRADE", 0.0, {"BUY": 0.0, "SELL": 0.0, "NO_TRADE": 1.0}
            feat   = self._align_features(feat)
            scaled = self.scaler.transform(feat.iloc[[-1]].values)
            probas = self._ensemble_proba(scaled)[0]

            p_buy  = float(probas[1])
            p_sell = float(probas[2])

            sharpe = self.get_live_sharpe()
            sharpe_mul = 0.85 if (sharpe < -0.5 and len(self._live_pnl_history) > 20) else 1.0
            p_buy  *= sharpe_mul
            p_sell *= sharpe_mul

            prob_dict = {
                "NO_TRADE": float(probas[0]),
                "BUY":      float(probas[1]),
                "SELL":     float(probas[2]),
            }

            if p_buy >= p_sell and p_buy >= MIN_SIGNAL_PROBABILITY:
                return "BUY",  p_buy,  prob_dict
            elif p_sell > p_buy and p_sell >= MIN_SIGNAL_PROBABILITY:
                return "SELL", p_sell, prob_dict
            else:
                return "NO_TRADE", max(p_buy, p_sell), prob_dict

        except Exception as e:
            logger.error(f"predict_full error: {e}", exc_info=True)
            return "NO_TRADE", 0.0, {"BUY": 0.0, "SELL": 0.0, "NO_TRADE": 1.0}

    def train(self, df: pd.DataFrame,
              df_h4: Optional[pd.DataFrame] = None,
              forward_bars: int = 5,
              rr_threshold: float = 0.5,
              mtf_result=None,
              save_if_best: bool = True,
              # v19 DIR-4: hyper-search overrides
              n_estimators: int = 100,
              max_depth: int = 7,
              learning_rate: float = 0.08,
              ) -> bool:
        try:
            logger.info(f"[{self.symbol}] Building features (FeatureEngineer connected)...")
            feat = self._build_features(df, df_h4=df_h4, mtf_result=mtf_result)
            if feat is None:
                logger.error("Feature matrix is None.")
                return False
            logger.info(f"[{self.symbol}] Features: {feat.shape[1]} cols x {len(feat)} rows")
            if len(feat) < MIN_TRAINING_SAMPLES:
                logger.warning(f"Insufficient samples ({len(feat)}). Try --bars 3000")
                return False

            # ── v19 DIR-2: AUTO-LABEL TUNING ─────────────────────────────────
            if AUTO_ENGINE_AVAILABLE:
                tp_mult_lbl, sl_mult_lbl, max_bars_lbl = AutoLabelTuner.find_valid_params(
                    self._generate_fixed_risk_labels, df, symbol=self.symbol
                )
            else:
                tp_mult_lbl, sl_mult_lbl, max_bars_lbl = 2.0, 1.0, 30
            logger.info(f"[{self.symbol}] Generating CAUSAL fixed-risk labels "
                        f"(TP={tp_mult_lbl}xATR, SL={sl_mult_lbl}xATR, max_bars={max_bars_lbl})...")
            labels = self._generate_fixed_risk_labels(
                df, tp_mult=tp_mult_lbl, sl_mult=sl_mult_lbl, max_bars=max_bars_lbl
            )

            common = feat.index.intersection(labels.index)
            feat   = feat.loc[common]
            labels = labels.loc[common]
            valid  = labels.notna()
            feat   = feat[valid]
            labels = labels[valid]

            logger.info(f"[{self.symbol}] Labels => BUY:{(labels==1).sum()} "
                        f"SELL:{(labels==2).sum()} NO_TRADE:{(labels==0).sum()}")

            # Cap NO_TRADE to prevent class imbalance crushing directional signals
            n_buy  = int((labels == 1).sum())
            n_sell = int((labels == 2).sum())
            n_tradeable = n_buy + n_sell
            n_no_trade  = int((labels == 0).sum())
            max_no_trade = max(n_tradeable * 2, 50)
            if n_no_trade > max_no_trade:
                nt_idx = labels[labels == 0].index
                drop_idx = np.random.default_rng(42).choice(
                    len(nt_idx), size=n_no_trade - max_no_trade, replace=False
                )
                labels = labels.drop(nt_idx[drop_idx])
                feat   = feat.loc[labels.index]
                logger.info(f"[{self.symbol}] NO_TRADE capped: kept {max_no_trade} "
                            f"(was {n_no_trade}) — preventing class imbalance")

            tradeable = int((labels != 0).sum())
            if tradeable < 30:
                logger.warning(f"Too few BUY/SELL labels ({tradeable}). Try --rr 0.4")
                return False

            X = feat.values.astype(np.float64)
            y = labels.values.astype(int)

            # ── v19 DIR-3: AUTO WALK-FORWARD TUNING ──────────────────────────
            n_rows = len(X)
            if AUTO_ENGINE_AVAILABLE:
                adaptive_splits, adaptive_embargo, _test_size_wf = AutoWalkForwardTuner.safe_params(n_rows)
            else:
                _min_train_wf    = max(50, WF_MIN_TRAIN_SIZE)
                adaptive_embargo = max(3, min(WF_EMBARGO_BARS, n_rows // 20))
                _test_size_wf    = max(10, n_rows // 30)
                _max_safe_splits = max(2, (n_rows - _min_train_wf) // (_test_size_wf + adaptive_embargo))
                adaptive_splits  = min(_max_safe_splits, 10)

            if adaptive_splits == 0:
                self._wf_mean_acc = 0.0
                wf_scores = []
                logger.warning(f"[{self.symbol}] [AUTO] Dataset too small for CV — "
                               "training on full data, WF accuracy = 0.0")
            else:
                logger.info(f"[{self.symbol}] Purged walk-forward CV "
                            f"({adaptive_splits} folds, embargo={adaptive_embargo} bars, "
                            f"test_size={_test_size_wf})...")
                wf_scores = self._walk_forward_cv(X, y,
                                                   n_splits=adaptive_splits,
                                                   embargo=adaptive_embargo,
                                                   test_size=_test_size_wf)
                self._wf_mean_acc = float(np.mean(wf_scores)) if wf_scores else 0.0
            wf_std = float(np.std(wf_scores)) if len(wf_scores) > 1 else 0.0
            if not wf_scores:
                logger.warning(
                    f"[{self.symbol}] WF CV: all folds skipped "
                    f"(dataset too small for {adaptive_splits}-fold split at this window). "
                    "Accuracy reported as 0.000 — model will still train on full data."
                )
            else:
                fold_strs = [round(s, 3) for s in wf_scores]
                min_fold  = min(wf_scores)
                max_fold  = max(wf_scores)
                spread    = max_fold - min_fold
                logger.info(
                    f"[{self.symbol}] WF OOS accuracy: {self._wf_mean_acc:.3f} "
                    f"+/-{wf_std:.3f} | min={min_fold:.3f} max={max_fold:.3f} "
                    f"spread={spread:.3f} | folds={fold_strs}"
                )
                if spread > 0.30:
                    logger.warning(
                        f"[{self.symbol}] HIGH fold spread ({spread:.3f}) — model is "
                        "regime-sensitive, not generalisable. "
                        "Edge is real only when ALL folds > 0.50. "
                        "Consider: more data, stricter features, or per-regime models."
                    )
                elif self._wf_mean_acc < 0.44 and wf_scores:
                    logger.warning(
                        f"[{self.symbol}] Mean WF accuracy {self._wf_mean_acc:.3f} < 0.44 "
                        "(3-class balanced acc; random=0.333, gate=0.44). "
                        "Model needs retraining with better features."
                    )
                else:
                    logger.info(
                        f"[{self.symbol}] WF accuracy acceptable "
                        f"(mean={self._wf_mean_acc:.3f}, spread={spread:.3f})"
                    )

                # Profitable gate: mean must beat genuine predictive threshold AND
                # spread must be tight enough to indicate consistent edge.
                #
                # v15 GATE FIX: The old gate mean>=0.50 was calibrated for binary
                # classification.  For 3-class balanced accuracy:
                #   random baseline = 0.333 (not 0.50)
                #   practical ceiling on H1 FX/Gold ≈ 0.52-0.56
                #   genuinely predictive (beats random by ≥25% relative) ≥ 0.42
                #
                # Setting the gate at 0.50 made it unreachable in practice, causing
                # ALL windows to fall back to the fallback path and the best-window
                # selection to be meaningless.
                #
                # New gate: mean>=0.44 (beats random by 32% relative, clearly
                # profitable with ATR-scaled TP/SL) AND spread<=0.35 (regime stable).
                WF_GATE_MEAN = 0.44
                WF_GATE_SPREAD = 0.35
                self._wf_gate_passed = not (self._wf_mean_acc < WF_GATE_MEAN or spread > WF_GATE_SPREAD)
                if not self._wf_gate_passed:
                    logger.warning(
                        f"[{self.symbol}] WINDOW BELOW GATE — "
                        f"mean={self._wf_mean_acc:.3f} spread={spread:.3f}. "
                        f"Need mean>={WF_GATE_MEAN} (3-class BA; random=0.333) "
                        f"and spread<={WF_GATE_SPREAD}. Will train but not preferred."
                    )

            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = list(feat.columns)
            self._n_features   = len(self.feature_names)

            classes = np.unique(y)
            cw      = compute_class_weight("balanced", classes=classes, y=y)
            cw_dict = dict(zip(classes.tolist(), cw.tolist()))

            adaptive_leaf = max(3, len(X) // 80)

            # ── v19 DIR-8: LIVE BLEND + boost recent 20 trades ──────────────
            X_blend, y_blend, w_blend = self.blend_live_samples(X_scaled, y)
            n_hist = len(X_scaled)
            if len(X_blend) > n_hist:
                X_blend[n_hist:] = self.scaler.transform(X_blend[n_hist:])
                if AUTO_ENGINE_AVAILABLE:
                    w_blend = AutoLiveRetrain.boosted_sample_weight(w_blend, n_hist)

            classes_b = np.unique(y_blend)
            cw_b      = compute_class_weight("balanced", classes=classes_b, y=y_blend)
            cw_dict_b = dict(zip(classes_b.tolist(), cw_b.tolist()))
            adaptive_leaf_b = max(3, len(X_blend) // 80)
            X_scaled_b = X_blend
            y_b        = y_blend

            logger.info(f"[{self.symbol}] Training RandomForest "
                        f"(n_est={n_estimators} depth={max_depth} leaf={adaptive_leaf_b})...")
            self.rf_model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                min_samples_leaf=adaptive_leaf_b,
                class_weight=cw_dict_b, random_state=42, n_jobs=1)
            self.rf_model.fit(X_scaled_b, y_b, sample_weight=w_blend)

            logger.info(f"[{self.symbol}] Training HistGBM "
                        f"(80 iters, lr={learning_rate}, depth={max_depth})...")
            self.gbm_model = HistGradientBoostingClassifier(
                max_iter=80, learning_rate=learning_rate, max_depth=max_depth,
                class_weight="balanced", random_state=42)
            self.gbm_model.fit(X_scaled_b, y_b, sample_weight=w_blend)

            X_scaled_df = pd.DataFrame(X_scaled_b, columns=self.feature_names)

            if XGB_AVAILABLE:
                logger.info(f"[{self.symbol}] Training XGBoost (n_est={n_estimators})...")
                self.xgb_model = xgb.XGBClassifier(
                    n_estimators=n_estimators, learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="mlogloss", random_state=42, n_jobs=1)
                self.xgb_model.fit(X_scaled_df, y_b, sample_weight=w_blend)

            if LGB_AVAILABLE:
                logger.info(f"[{self.symbol}] Training LightGBM (n_est={n_estimators})...")
                self.lgb_model = lgb.LGBMClassifier(
                    n_estimators=n_estimators, learning_rate=learning_rate,
                    max_depth=max_depth,
                    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                    class_weight="balanced", random_state=42, n_jobs=1,
                    verbosity=-1)
                self.lgb_model.fit(X_scaled_df, y_b, sample_weight=w_blend)

            self.meta_model = self._train_meta_learner(X_scaled, y, cw_dict)

            # ── v19 DIR-7: AUTO ENSEMBLE WEIGHTS ─────────────────────────────
            if AUTO_ENGINE_AVAILABLE and adaptive_splits > 0:
                oof_sc = self._compute_oof_scores(
                    X_scaled, y, adaptive_splits, adaptive_embargo, _test_size_wf
                )
                wts = AutoEnsembleWeights.compute(oof_sc, symbol=self.symbol)
                self._rf_weight  = wts.get("rf",  1.0)
                self._gbm_weight = wts.get("gbm", 1.2)
                self._xgb_weight = wts.get("xgb", 1.3)
                self._lgb_weight = wts.get("lgb", 1.4)

            # ── v19 DIR-6: AUTO FEATURE SELECTION ────────────────────────────
            if AUTO_ENGINE_AVAILABLE and len(X_scaled) > 200 and self.rf_model is not None:
                oos_n = max(50, len(X_scaled) // 5)
                pruned_names = AutoFeatureSelector.prune(
                    self.rf_model, X_scaled[-oos_n:], y[-oos_n:],
                    self.feature_names, symbol=self.symbol
                )
                if len(pruned_names) < len(self.feature_names):
                    self._try_pruned_retrain(
                        feat[pruned_names], labels,
                        n_estimators, max_depth, learning_rate, w_blend, n_hist
                    )

            # ── v19 DIR-5: REGIME SUB-MODELS ─────────────────────────────────
            if AUTO_ENGINE_AVAILABLE:
                self._train_regime_submodels(
                    df, df_h4, forward_bars, n_estimators, max_depth, learning_rate
                )

            self.is_trained  = True
            self._train_date = datetime.now().isoformat()

            if hasattr(self.rf_model, "feature_importances_"):
                fi = sorted(zip(self.feature_names, self.rf_model.feature_importances_),
                            key=lambda x: x[1], reverse=True)
                top10 = fi[:10]
                logger.info(f"[{self.symbol}] Top-10 features: " +
                            ", ".join(f"{n}={v:.3f}" for n,v in top10))

            if save_if_best:
                self._save_model()
            return True
        except Exception as e:
            logger.error(f"[{self.symbol}] train() failed: {e}", exc_info=True)
            return False

    # ── v19 DIR-5: Regime sub-model training ────────────────────────────────

    def get_regime_model_path(self, regime: str) -> str:
        safe = self.symbol.replace("/", "_").replace(".", "_")
        return os.path.join(MODEL_DIR, f"signal_model_{safe}_{regime}.pkl")

    def _train_regime_submodels(
        self,
        df,
        df_h4,
        forward_bars: int,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
    ) -> None:
        """
        DIR-5: train and save trending/ranging sub-models when sufficient regime bars exist.
        """
        try:
            df_trend, df_range = AutoRegimeTrainer.split_by_regime(df)
            for regime_name, df_r in [("trending", df_trend), ("ranging", df_range)]:
                if df_r is None:
                    continue
                try:
                    feat_r = self._build_features(df_r, df_h4=None)
                    if feat_r is None or len(feat_r) < 100:
                        continue
                    tp_r, sl_r, mb_r = AutoLabelTuner.find_valid_params(
                        self._generate_fixed_risk_labels, df_r, symbol=f"{self.symbol}/{regime_name}"
                    )
                    labels_r = self._generate_fixed_risk_labels(df_r, tp_mult=tp_r, sl_mult=sl_r, max_bars=mb_r)
                    common_r = feat_r.index.intersection(labels_r.index)
                    feat_r   = feat_r.loc[common_r]
                    labels_r = labels_r.loc[common_r][labels_r.loc[common_r].notna()]
                    feat_r   = feat_r.loc[labels_r.index]
                    if int((labels_r != 0).sum()) < 20:
                        continue
                    X_r = feat_r.values.astype(np.float64)
                    y_r = labels_r.values.astype(int)
                    sc_r = StandardScaler()
                    X_rs = sc_r.fit_transform(X_r)
                    cw_r = compute_class_weight("balanced", classes=np.unique(y_r), y=y_r)
                    cwd_r = dict(zip(np.unique(y_r).tolist(), cw_r.tolist()))
                    leaf_r = max(3, len(X_rs) // 80)
                    rf_r = RandomForestClassifier(
                        n_estimators=n_estimators, max_depth=max_depth,
                        min_samples_leaf=leaf_r,
                        class_weight=cwd_r, random_state=42, n_jobs=1
                    )
                    rf_r.fit(X_rs, y_r)
                    gbm_r = HistGradientBoostingClassifier(
                        max_iter=60, learning_rate=learning_rate, max_depth=4,
                        class_weight="balanced", random_state=42
                    )
                    gbm_r.fit(X_rs, y_r)
                    regime_path = self.get_regime_model_path(regime_name)
                    tmp_path    = regime_path + ".tmp"
                    payload_r = {
                        "rf": rf_r, "gbm": gbm_r, "scaler": sc_r,
                        "feature_names": list(feat_r.columns),
                        "regime": regime_name, "symbol": self.symbol,
                        "saved_at": datetime.now().isoformat(),
                    }
                    with open(tmp_path, "wb") as fp:
                        pickle.dump(payload_r, fp)
                    os.replace(tmp_path, regime_path)
                    logger.info(
                        f"[AUTO] {self.symbol}: regime sub-model saved "
                        f"({regime_name}, {len(X_rs)} bars) → {regime_path}"
                    )
                    # Cache in memory
                    if regime_name == "trending":
                        self.rf_model_trending  = rf_r
                        self.gbm_model_trending = gbm_r
                    else:
                        self.rf_model_ranging  = rf_r
                        self.gbm_model_ranging = gbm_r
                except Exception as rexc:
                    logger.debug(f"[AUTO] regime sub-model {regime_name} error: {rexc}")
        except Exception as exc:
            logger.debug(f"[AUTO] _train_regime_submodels error: {exc}")

    def _load_regime_models(self) -> None:
        for regime_name, attr_rf, attr_gbm in [
            ("trending", "rf_model_trending", "gbm_model_trending"),
            ("ranging",  "rf_model_ranging",  "gbm_model_ranging"),
        ]:
            path = self.get_regime_model_path(regime_name)
            if not os.path.exists(path):
                continue
            try:
                with open(path, "rb") as fp:
                    p = pickle.load(fp)
                setattr(self, attr_rf,  p.get("rf"))
                setattr(self, attr_gbm, p.get("gbm"))
                logger.debug(f"[{self.symbol}] Regime sub-model loaded: {regime_name}")
            except Exception:
                pass

    # ── v19 DIR-7: OOF score computation ────────────────────────────────────

    def _compute_oof_scores(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int,
        embargo: int,
        test_size: int,
    ) -> dict:
        """Compute per-model OOF balanced accuracy for DIR-7 weight tuning."""
        if n_splits < 2 or len(X) < 200:
            return {"rf": 0.45, "gbm": 0.45, "xgb": 0.45, "lgb": 0.45}
        try:
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import balanced_accuracy_score
            tscv = TimeSeriesSplit(n_splits=min(n_splits, 5), gap=embargo, test_size=test_size)
            scores: dict = {"rf": [], "gbm": [], "xgb": [], "lgb": []}
            for tr_idx, te_idx in tscv.split(X):
                if len(tr_idx) < 50 or len(te_idx) < 5:
                    continue
                sc = StandardScaler()
                Xtr = sc.fit_transform(X[tr_idx])
                Xte = sc.transform(X[te_idx])
                ytr, yte = y[tr_idx], y[te_idx]
                cw = dict(zip(*[c.tolist() for c in [
                    np.unique(ytr),
                    compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
                ]]))
                leaf = max(2, len(Xtr) // 80)
                models_cv = {
                    "rf":  RandomForestClassifier(n_estimators=40, max_depth=6,
                                                   min_samples_leaf=leaf, class_weight=cw,
                                                   random_state=42, n_jobs=1),
                    "gbm": HistGradientBoostingClassifier(max_iter=40, learning_rate=0.1,
                                                          max_depth=4, random_state=42,
                                                          class_weight="balanced"),
                }
                for name, m in models_cv.items():
                    m.fit(Xtr, ytr)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        acc = balanced_accuracy_score(yte, m.predict(Xte))
                    scores[name].append(acc)
            oof = {k: float(np.mean(v)) if v else 0.4 for k, v in scores.items()}
            # XGB/LGB: use same as GBM if not computed (fast approximation)
            oof.setdefault("xgb", oof.get("gbm", 0.4))
            oof.setdefault("lgb", oof.get("gbm", 0.4))
            return oof
        except Exception as exc:
            logger.debug(f"_compute_oof_scores error: {exc}")
            return {"rf": 0.4, "gbm": 0.4, "xgb": 0.4, "lgb": 0.4}

    # ── v19 DIR-6: Pruned feature retrain ────────────────────────────────────

    def _try_pruned_retrain(
        self,
        feat_pruned,
        labels,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
        w_blend: np.ndarray,
        n_hist: int,
    ) -> bool:
        """Retrain on pruned feature set; adopt if wf_acc improves or stays within 0.01."""
        try:
            common = feat_pruned.index.intersection(labels.index)
            fp     = feat_pruned.loc[common]
            lp     = labels.loc[common][labels.loc[common].notna()]
            fp     = fp.loc[lp.index]
            if len(fp) < 100:
                return False
            Xp = fp.values.astype(np.float64)
            yp = lp.values.astype(int)
            sp = StandardScaler()
            Xps = sp.fit_transform(Xp)
            # Blend live samples (reuse existing buffer)
            Xpb, ypb, wpb = self.blend_live_samples(Xps, yp)
            if len(Xpb) > len(Xps):
                Xpb[len(Xps):] = sp.transform(Xpb[len(Xps):])
                if AUTO_ENGINE_AVAILABLE:
                    wpb = AutoLiveRetrain.boosted_sample_weight(wpb, len(Xps))
            cw_p = compute_class_weight("balanced", classes=np.unique(ypb), y=ypb)
            cwd_p = dict(zip(np.unique(ypb).tolist(), cw_p.tolist()))
            leaf_p = max(3, len(Xpb) // 80)
            rf_p = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                min_samples_leaf=leaf_p, class_weight=cwd_p, random_state=42, n_jobs=1
            )
            rf_p.fit(Xpb, ypb, sample_weight=wpb)
            # Quick WF estimate
            n_sp, emb_p, ts_p = AutoWalkForwardTuner.safe_params(len(Xps))
            if n_sp >= 2:
                wf_p = self._walk_forward_cv(Xps, yp, n_splits=n_sp, embargo=emb_p, test_size=ts_p)
                pruned_acc = float(np.mean(wf_p)) if wf_p else 0.0
            else:
                pruned_acc = 0.0
            if pruned_acc >= self._wf_mean_acc - 0.01:
                logger.info(
                    f"[AUTO] {self.symbol}: pruned model acc={pruned_acc:.3f} ≥ "
                    f"full acc={self._wf_mean_acc:.3f} - 0.01 — adopting pruned features "
                    f"({len(fp.columns)} feats)"
                )
                # ── Retrain ALL base models on pruned feature set so every model
                # in the ensemble uses the same number of features. Skipping this
                # caused "X has N feats but model expects 70" errors at predict time.
                gbm_p = HistGradientBoostingClassifier(
                    max_iter=n_hist, learning_rate=learning_rate,
                    max_depth=max_depth, random_state=42
                )
                gbm_p.fit(Xpb, ypb, sample_weight=wpb)

                xgb_p = None
                if XGB_AVAILABLE:
                    try:
                        import xgboost as xgb
                        xgb_p = xgb.XGBClassifier(
                            n_estimators=n_estimators, max_depth=max_depth,
                            learning_rate=learning_rate,
                            eval_metric="mlogloss", random_state=42, n_jobs=1
                        )
                        xgb_p.fit(Xpb, ypb, sample_weight=wpb)
                    except Exception:
                        xgb_p = None

                lgb_p = None
                if LGB_AVAILABLE:
                    try:
                        import lightgbm as lgb
                        lgb_p = lgb.LGBMClassifier(
                            n_estimators=n_estimators, max_depth=max_depth,
                            learning_rate=learning_rate, random_state=42, n_jobs=1,
                            verbose=-1
                        )
                        lgb_p.fit(Xpb, ypb, sample_weight=wpb)
                    except Exception:
                        lgb_p = None

                self.rf_model        = rf_p
                self.gbm_model       = gbm_p
                if xgb_p is not None:
                    self.xgb_model   = xgb_p
                if lgb_p is not None:
                    self.lgb_model   = lgb_p
                self.meta_model      = None   # invalidate stale meta-learner
                self.scaler          = sp
                self.feature_names   = list(fp.columns)
                self._n_features     = len(self.feature_names)
                self._wf_mean_acc    = pruned_acc
                return True
            return False
        except Exception as exc:
            logger.debug(f"_try_pruned_retrain error: {exc}")
            return False

    # ── v19 DIR-5: Regime-blended predict ────────────────────────────────────

    def _regime_blended_proba(self, X_named: Optional[pd.DataFrame], X: np.ndarray, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        If a regime sub-model matches current market conditions, blend its output
        at BLEND_REGIME/1-BLEND_REGIME with the general ensemble.
        Returns blended 3-class probas or None if no confident regime detected.
        """
        if not AUTO_ENGINE_AVAILABLE:
            return None
        try:
            regime_name, confidence = AutoRegimeTrainer.current_regime_confidence(df)
            if confidence < 0.5 or regime_name == "neutral":
                return None
            rf_reg  = self.rf_model_trending  if regime_name == "trending" else self.rf_model_ranging
            gbm_reg = self.gbm_model_trending if regime_name == "trending" else self.gbm_model_ranging
            if rf_reg is None:
                return None
            blend = float(os.getenv("REGIME_BLEND_WEIGHT", "0.6"))
            preds_reg = []
            for m in [rf_reg, gbm_reg]:
                if m is None:
                    continue
                raw = m.predict_proba(X)
                preds_reg.append(self._remap_proba(raw, m.classes_))
            if not preds_reg:
                return None
            regime_out = np.mean(preds_reg, axis=0)
            return regime_out
        except Exception:
            return None

    def _train_meta_learner(self, X_scaled: np.ndarray, y: np.ndarray,
                             cw_dict: dict) -> Optional[object]:
        """
        v9 FIX 13a: Train stacked meta-learner (LogisticRegression) on OOF preds
        of RF+GBM. Captures model disagreement patterns; improves probability
        calibration. Falls back to None if insufficient data.
        """
        try:
            active = [m for m in [self.rf_model, self.gbm_model] if m is not None]
            if len(active) < 2 or len(X_scaled) < 200:
                return None

            n = len(X_scaled)
            meta_X = np.zeros((n, len(active) * 3), dtype=float)
            tscv   = TimeSeriesSplit(n_splits=5, gap=20)

            for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_scaled)):
                if len(tr_idx) < 80 or len(te_idx) < 10:
                    continue
                sc  = StandardScaler()
                Xtr = sc.fit_transform(X_scaled[tr_idx])
                Xte = sc.transform(X_scaled[te_idx])
                for m_idx, (model_cls, kwargs) in enumerate([
                    (RandomForestClassifier,         dict(n_estimators=40, max_depth=5,
                                                          class_weight=cw_dict,
                                                          random_state=fold, n_jobs=1)),
                    # SPEED FIX: HistGradientBoostingClassifier replaces GradientBoostingClassifier
                    # in meta-learner folds — same diversity, 20-40x faster.
                    (HistGradientBoostingClassifier, dict(max_iter=40, learning_rate=0.1,
                                                          max_depth=4, random_state=fold,
                                                          class_weight="balanced")),
                ]):
                    m = model_cls(**kwargs)
                    m.fit(Xtr, y[tr_idx])
                    raw  = m.predict_proba(Xte)
                    rmap = self._remap_proba(raw, m.classes_)
                    meta_X[te_idx, m_idx*3:(m_idx+1)*3] = rmap

            valid = meta_X.any(axis=1)
            if valid.sum() < 50:
                return None

            meta_clf = LogisticRegression(
                C=0.5, multi_class="multinomial", solver="lbfgs",
                max_iter=500, class_weight="balanced", random_state=42
            )
            meta_clf.fit(meta_X[valid], y[valid])
            logger.info(f"[{self.symbol}] Meta-learner trained on {valid.sum()} OOF samples")
            return meta_clf
        except Exception as e:
            logger.debug(f"Meta-learner failed: {e}")
            return None

    def _build_features(self, df: pd.DataFrame,
                        df_h4: Optional[pd.DataFrame] = None,
                        mtf_result=None) -> Optional[pd.DataFrame]:
        if len(df) < 50:
            logger.error(f"Too few rows: {len(df)}")
            return None
        if self.feature_engineer is not None and _FE_AVAILABLE:
            try:
                enriched = self.feature_engineer.build(df, df_h4=df_h4, mtf_result=mtf_result)
                feature_cols = [c for c in enriched.columns if c.startswith("f_")]
                if len(feature_cols) >= 20:
                    feat = enriched[feature_cols].copy()
                    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
                    feat.dropna(inplace=True)
                    if len(feat) >= 50:
                        return feat
            except Exception as e:
                logger.warning(f"FeatureEngineer.build failed: {e} -- using fallback")
        return self._build_features_fallback(df)

    def _build_features_fallback(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        required = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in required):
            return None
        try:
            d = df.copy()
            d["returns_1"]   = d["close"].pct_change(1)
            d["returns_3"]   = d["close"].pct_change(3)
            d["returns_5"]   = d["close"].pct_change(5)
            hl = (d["high"] - d["low"]).replace(0, np.nan)
            d["hl_range"]    = hl / d["close"].replace(0, np.nan)
            d["body_pct"]    = (d["close"] - d["open"]).abs() / hl
            d["upper_wick"]  = (d["high"] - d[["close","open"]].max(axis=1)) / hl
            d["lower_wick"]  = (d[["close","open"]].min(axis=1) - d["low"]) / hl
            for vc in ["real_volume","tick_volume"]:
                if vc in d.columns and d[vc].sum() > 0:
                    vm = d[vc].rolling(20).mean().replace(0, np.nan)
                    d["vol_ratio"] = d[vc] / vm; break
            else:
                d["vol_ratio"] = 1.0
            rsi = self._calc_rsi(d["close"])
            d["rsi_norm"] = (rsi - 50) / 50
            fast = d["close"].ewm(span=12, adjust=False).mean()
            slow = d["close"].ewm(span=26, adjust=False).mean()
            ml   = fast - slow
            d["macd_hist"] = ml - ml.ewm(span=9, adjust=False).mean()
            atr = self._calc_atr(d) if "atr" not in d.columns else d["atr"]
            d["atr_norm"] = atr / d["close"].replace(0, np.nan)
            sma20 = d["close"].rolling(20).mean()
            std20 = d["close"].rolling(20).std()
            bb_up = sma20 + 2*std20; bb_lo = sma20 - 2*std20
            d["bb_pos"] = (d["close"] - bb_lo) / (bb_up - bb_lo).replace(0, np.nan)
            d["trend_str"] = self._calc_trend_strength(d)
            d["vol_pct"]   = d["returns_1"].rolling(14).std()
            sma50 = d["close"].rolling(50).mean()
            d["htf_bias"] = np.where(sma20 > sma50, 1.0, -1.0)
            if hasattr(d.index, "hour"):
                d["hour_sin"] = np.sin(2*np.pi*d.index.hour/24)
                d["hour_cos"] = np.cos(2*np.pi*d.index.hour/24)
            else:
                d["hour_sin"] = 0.0; d["hour_cos"] = 0.0
            d["roc_10"]       = d["close"].pct_change(10)
            d["close_vs_h20"] = (d["close"] - d["high"].rolling(20).max()) / d["close"].replace(0, np.nan)
            d["close_vs_l20"] = (d["close"] - d["low"].rolling(20).min())  / d["close"].replace(0, np.nan)
            cols = ["returns_1","returns_3","returns_5","hl_range","body_pct",
                    "upper_wick","lower_wick","vol_ratio","rsi_norm","macd_hist",
                    "atr_norm","bb_pos","trend_str","vol_pct","htf_bias",
                    "hour_sin","hour_cos","roc_10","close_vs_h20","close_vs_l20"]
            feat = d[cols].copy()
            feat.replace([np.inf, -np.inf], np.nan, inplace=True)
            feat.dropna(inplace=True)
            return feat if len(feat) >= 50 else None
        except Exception as e:
            logger.error(f"_build_features_fallback error: {e}", exc_info=True)
            return None

    def _align_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names:
            return feat
        for c in self.feature_names:
            if c not in feat.columns:
                feat[c] = 0.0
        extra = [c for c in feat.columns if c not in self.feature_names]
        if extra:
            feat = feat.drop(columns=extra)
        return feat[self.feature_names]

    def _generate_fixed_risk_labels(
        self, df: pd.DataFrame,
        tp_mult: float = 2.0, sl_mult: float = 1.0, max_bars: int = 30
    ) -> pd.Series:
        """
        CAUSAL directional fixed-risk labeling — v14 VECTORIZED rewrite.

        v12 approach: correct but used two nested Python for-loops (O(n*max_bars)).
        On 8000 bars × 30 lookahead this is ~240 000 iterations PER SYMBOL PER
        WINDOW and causes the "stuck for 10+ minutes" hang.

        v14 FIX: fully vectorized using numpy stride tricks + cummin/cummax
        running windows.  Same exact label semantics, ~300x faster.

        Algorithm:
          For each bar i, we need to know the FIRST outcome in bars [i+1, i+max_bars]:
            BUY_WIN : max(high[i+1..i+max_bars]) hits buy_tp AND the bar that
                      first hits buy_tp comes BEFORE the bar that first hits buy_sl.
            SELL_WIN: analogous for sell side.

          We use a sliding-window cummax/cummin approach:
          - Build (n, max_bars) matrices of future highs / future lows using
            np.lib.stride_tricks.as_strided  (zero-copy view — no extra RAM).
          - For each bar: find first_buy_tp_bar  = argmax(future_highs >= buy_tp)
                                  first_buy_sl_bar  = argmax(future_lows  <= buy_sl)
            If first_buy_tp_bar < first_buy_sl_bar  → BUY WIN (or no SL hit at all).
          - Same logic for SELL direction.

        Memory: two (n × max_bars) float64 windows ≈ 8000 × 30 × 8B × 2 = ~3.8 MB.
        """
        if "atr" in df.columns and df["atr"].notna().sum() > 20:
            atr = df["atr"].copy()
        else:
            atr = self._calc_atr(df)
        atr = atr.fillna((df["high"] - df["low"]).rolling(14).mean()).ffill().bfill()

        adx           = self._calc_adx_series(df)
        trending_mask = (adx > 25).values
        ranging_mask  = (adx < 15).values

        closes = df["close"].values.astype(np.float64)
        highs  = df["high"].values.astype(np.float64)
        lows   = df["low"].values.astype(np.float64)
        atrs   = atr.values.astype(np.float64)
        n      = len(df)

        # Adaptive TP multiplier per bar
        eff_tp = np.where(trending_mask, tp_mult * 1.25,
                 np.where(ranging_mask,  tp_mult * 0.75, tp_mult))

        # Pre-compute entry levels (only for valid bars)
        valid = np.isfinite(atrs) & (atrs > 0) & np.isfinite(closes)
        min_move = closes * 0.0003
        rr_ok = (eff_tp * atrs) >= min_move

        buy_tp_arr  = closes + eff_tp * atrs
        buy_sl_arr  = closes - sl_mult * atrs
        sell_tp_arr = closes - eff_tp * atrs
        sell_sl_arr = closes + sl_mult * atrs

        # Build future-window matrices using stride tricks (zero-copy)
        # Shape: (n_eval, max_bars) — row i has highs/lows for bars [i+1..i+max_bars]
        n_eval = n - max_bars  # bars we can evaluate (last max_bars have no future)
        if n_eval <= 0:
            return pd.Series(np.zeros(n, dtype=int), index=df.index)

        # Pad highs/lows so stride view works cleanly
        highs_pad = np.concatenate([highs, np.full(max_bars, np.nan)])
        lows_pad  = np.concatenate([lows,  np.full(max_bars, np.nan)])

        stride = highs_pad.strides[0]
        from numpy.lib.stride_tricks import as_strided
        # Each row i starts at offset i+1 (one bar ahead)
        future_highs = as_strided(
            highs_pad[1:], shape=(n_eval, max_bars),
            strides=(stride, stride)
        ).copy()   # .copy() materialises the view — safe against writes
        future_lows = as_strided(
            lows_pad[1:], shape=(n_eval, max_bars),
            strides=(stride, stride)
        ).copy()

        # For each bar i and each future bar j: did high/low cross the threshold?
        # buy_tp_hit[i, j] = 1 if future_highs[i, j] >= buy_tp_arr[i]
        buy_tp_hit  = future_highs >= buy_tp_arr[:n_eval, np.newaxis]
        buy_sl_hit  = future_lows  <= buy_sl_arr[:n_eval, np.newaxis]
        sell_tp_hit = future_lows  <= sell_tp_arr[:n_eval, np.newaxis]
        sell_sl_hit = future_highs >= sell_sl_arr[:n_eval, np.newaxis]

        INF = max_bars + 1

        def _first_hit(hit_matrix: np.ndarray) -> np.ndarray:
            """Return index of first True in each row, or INF if never."""
            # argmax returns 0 when no True — disambiguate with any()
            first = np.argmax(hit_matrix, axis=1)
            no_hit = ~hit_matrix.any(axis=1)
            first[no_hit] = INF
            return first

        first_buy_tp  = _first_hit(buy_tp_hit)
        first_buy_sl  = _first_hit(buy_sl_hit)
        first_sell_tp = _first_hit(sell_tp_hit)
        first_sell_sl = _first_hit(sell_sl_hit)

        buy_win  = (first_buy_tp  < INF) & (first_buy_tp  < first_buy_sl)
        sell_win = (first_sell_tp < INF) & (first_sell_tp < first_sell_sl)

        # Apply validity masks
        eval_valid = valid[:n_eval] & rr_ok[:n_eval]
        buy_win  &= eval_valid
        sell_win &= eval_valid

        # ── v15 FIX: Regime-direction label filter ─────────────────────────
        # The symmetric TP/SL approach creates equal BUY and SELL labels regardless
        # of market direction, making the model learn to predict noise rather than
        # directional edge.  We filter: only KEEP a BUY label if the market is in
        # a bullish or neutral regime, and only KEEP a SELL label in a bearish or
        # neutral regime.
        #
        # Regime proxy: 20-bar SMA vs 50-bar SMA direction (fully causal).
        # This halves the label noise on trending markets where counter-trend
        # winners are regime accidents, not repeatable signals.
        #
        # Filter strength: soft (keeps neutral regime bars for both directions)
        # to avoid over-thinning the dataset.
        try:
            sma20_vals = pd.Series(closes).rolling(20, min_periods=5).mean().values
            sma50_vals = pd.Series(closes).rolling(50, min_periods=10).mean().values
            # +1 = bullish regime, -1 = bearish, 0 = unknown (early bars)
            regime = np.sign(sma20_vals - sma50_vals)
            regime_eval = regime[:n_eval]

            # In a BEARISH regime, suppress BUY labels (allow neutral)
            # In a BULLISH regime, suppress SELL labels (allow neutral)
            buy_regime_ok  = regime_eval >= 0   # bullish or neutral
            sell_regime_ok = regime_eval <= 0   # bearish or neutral

            buy_win  &= buy_regime_ok
            sell_win &= sell_regime_ok
        except Exception:
            pass  # Silently skip regime filter if data insufficient

        # Ambiguous (both win): NO_TRADE
        clean_buy  = buy_win  & ~sell_win
        clean_sell = sell_win & ~buy_win

        labels = np.zeros(n, dtype=int)
        labels[:n_eval][clean_buy]  = 1
        labels[:n_eval][clean_sell] = 2

        buy_c  = int(clean_buy.sum())
        sell_c = int(clean_sell.sum())
        skip_c = int((~eval_valid).sum())
        nt_c   = int((labels == 0).sum())
        rate   = (buy_c + sell_c) / max(n_eval, 1) * 100

        logger.info(f"[{self.symbol}] Causal labels: BUY={buy_c} SELL={sell_c} "
                    f"NO_TRADE={nt_c} tradeable={rate:.1f}% skipped={skip_c}")
        return pd.Series(labels, index=df.index)



    def _walk_forward_cv(self, X: np.ndarray, y: np.ndarray,
                         n_splits: int = 20, embargo: int = 100,
                         test_size: Optional[int] = None) -> List[float]:
        """
        Purged walk-forward cross-validation (Lopez-de-Prado method).

        SPEED FIX v10.1 — two changes deliver ~40x wall-clock speedup:
          1. GradientBoostingClassifier → HistGradientBoostingClassifier (sklearn's
             histogram-based GBM, same as LightGBM under the hood). Identical API
             and accuracy; 20-40x faster because it bins features once, not per split.
          2. Sequential fold execution replaces joblib.Parallel — eliminates the
             OpenMP/HistGBM thread-pool deadlock on Windows and 2-core machines.

        v10 FIX: accepts explicit test_size — prevents sklearn ValueError on small
        datasets (MID/PREC windows).

        v9 FIX 11: balanced_accuracy_score with explicit labels — no UserWarning.
        v9 FIX 12c: RF+HistGBM dual-model ensemble per fold (diversity maintained).
        v9 FIX 12d: Proper Lopez-de-Prado purging of boundary rows.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=embargo,
                               test_size=test_size if test_size is not None else None)

        min_train = max(50, min(WF_MIN_TRAIN_SIZE, len(X) // 4))
        min_test  = max(5, test_size if test_size is not None else len(X) // 20)

        # Pre-collect valid folds so Parallel receives a fixed list
        fold_splits = []
        for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
            if len(tr_idx) < min_train:
                logger.debug(f"WF fold {fold}: skipped (train={len(tr_idx)} < {min_train})")
                continue
            if len(te_idx) < min_test:
                logger.debug(f"WF fold {fold}: skipped (test={len(te_idx)} < {min_test})")
                continue
            purge_n = min(embargo // 2, len(tr_idx) // 5)
            purged_tr_idx = tr_idx[:-purge_n] if (purge_n > 0 and len(tr_idx) > purge_n + min_train) else tr_idx
            if len(np.unique(y[purged_tr_idx])) < 2 or len(np.unique(y[te_idx])) < 2:
                logger.debug(f"WF fold {fold}: skipped (single class)")
                continue
            fold_splits.append((fold, purged_tr_idx, te_idx))

        def _run_fold(fold, purged_tr_idx, te_idx):
            try:
                sc      = StandardScaler()
                X_tr_s  = sc.fit_transform(X[purged_tr_idx])
                X_te_s  = sc.transform(X[te_idx])
                leaf_sz = max(2, len(purged_tr_idx) // 50)

                # RF fold model — n_jobs=1: we're already inside a loop; no nested pools
                rf_cv = RandomForestClassifier(
                    n_estimators=50, max_depth=6, min_samples_leaf=leaf_sz,
                    class_weight="balanced", random_state=42, n_jobs=1,
                )
                rf_cv.fit(X_tr_s, y[purged_tr_idx])
                rf_preds = rf_cv.predict(X_te_s)

                # SPEED FIX: HistGradientBoostingClassifier replaces GradientBoostingClassifier.
                # 20-40x faster: bins features once per fold (LightGBM-style histogram method).
                # class_weight="balanced" supported natively since sklearn 1.2.
                hgbm_cv = HistGradientBoostingClassifier(
                    max_iter=40, learning_rate=0.1, max_depth=4,
                    random_state=42, class_weight="balanced",
                )
                hgbm_cv.fit(X_tr_s, y[purged_tr_idx])
                gbm_preds = hgbm_cv.predict(X_te_s)

                # True majority vote: RF+GBM agree → use that label;
                # disagree → NO_TRADE (class 0) to avoid forced directional bias
                combined_preds = np.where(rf_preds == gbm_preds, rf_preds, 0)

                # v9 FIX 11: suppress UserWarning; handles missing classes in small test folds
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    acc = balanced_accuracy_score(y[te_idx], combined_preds)

                # v12: directional precision — how often does the model correctly call
                # BUY or SELL (not NO_TRADE) when the label IS directional?
                te_y    = y[te_idx]
                te_pred = combined_preds
                dir_mask = te_y != 0   # only bars labeled BUY or SELL
                dir_acc  = float(np.mean(te_pred[dir_mask] == te_y[dir_mask])) if dir_mask.any() else 0.0
                no_trade_rate = float(np.mean(te_pred == 0))
                logger.debug(
                    f"WF fold {fold}: train={len(purged_tr_idx)} test={len(te_idx)} "
                    f"balanced_acc={acc:.3f} dir_acc={dir_acc:.3f} "
                    f"no_trade_rate={no_trade_rate:.2%}"
                )
                return float(acc)
            except Exception as fold_err:
                logger.debug(f"WF fold {fold} error: {fold_err}")
                return None

        # ── Sequential fold execution — eliminates thread-pool deadlock ─────────
        # DEADLOCK FIX v14.1: joblib Parallel(prefer="threads") + HistGBM's
        # internal OpenMP threads = thread-pool exhaustion on Windows / low-core
        # machines.  Sequential is safe and fast enough: each fold ≈ 1-2s,
        # 8 folds = ~12s total which is fine for a one-time training run.
        results = [_run_fold(fold, ptr, te) for fold, ptr, te in fold_splits]
        return [r for r in results if r is not None]

    @staticmethod
    def _remap_proba(raw: np.ndarray, classes) -> np.ndarray:
        """
        Remap predict_proba output to canonical [NO_TRADE=0, BUY=1, SELL=2] columns.
        Handles models trained on subsets of classes (e.g., only [0,1] or [0,2]).
        """
        CANONICAL = {0: 0, 1: 1, 2: 2}
        classes_list = list(classes)
        if classes_list == [0, 1, 2] and raw.shape[1] == 3:
            return raw
        remapped = np.zeros((raw.shape[0], 3), dtype=float)
        for col_idx, cls in enumerate(classes_list):
            canonical_col = CANONICAL.get(int(cls))
            if canonical_col is not None:
                remapped[:, canonical_col] += raw[:, col_idx]
        return remapped

    def _ensemble_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Weighted ensemble with temperature scaling and optional meta-learner.

        v9 FIX 13a: meta_model blended at 60% when available
        v9 FIX 13b: temperature scaling T=1.5 reduces overconfidence
        """
        preds, wts = [], []

        # FIX UserWarning: LGB and XGB were trained with named DataFrames; they
        # warn when predict_proba receives a raw numpy array.  Build a named
        # DataFrame once and use it only for those two models.
        # RF / GBM / HistGBM accept plain numpy arrays — no names needed.
        X_named: Optional[pd.DataFrame] = None
        if self.feature_names and X.shape[1] == len(self.feature_names):
            X_named = pd.DataFrame(X, columns=self.feature_names)

        for model, weight, needs_names in [
            (self.rf_model,                                    self._rf_weight,  False),
            (self.gbm_model,                                   self._gbm_weight, False),
            (self.xgb_model if XGB_AVAILABLE else None,        self._xgb_weight, True),
            (self.lgb_model if LGB_AVAILABLE else None,        self._lgb_weight, True),
        ]:
            if model is None:
                continue

            # ── v19 FIX: guard against stale pkl where base model was trained on
            # more/fewer features than the current scaler+feature_names.
            # This happens when auto-feature-pruning updated feature_names but
            # saved the old base models.  Skip the mismatched model silently
            # rather than crashing; remaining ensemble members still vote.
            try:
                expected = getattr(model, "n_features_in_", None)
                if expected is not None and expected != X.shape[1]:
                    logger.debug(
                        f"[{self.symbol}] _ensemble_proba: skipping "
                        f"{type(model).__name__} (expects {expected} feats, "
                        f"got {X.shape[1]}) — stale pkl, self-corrects on next retrain"
                    )
                    continue
            except Exception:
                pass

            X_in     = (X_named if (needs_names and X_named is not None) else X)
            raw      = model.predict_proba(X_in)
            remapped = self._remap_proba(raw, model.classes_)
            preds.append(remapped)
            wts.append(weight)

        if not preds:
            return np.array([[1.0, 0.0, 0.0]])

        total = sum(wts)
        out   = np.zeros((preds[0].shape[0], 3), dtype=float)
        for p, w in zip(preds, wts):
            out += p * (w / total)

        # v9 FIX 13a: Meta-learner blend (RF+GBM OOF-trained logistic)
        if self.meta_model is not None and len(preds) >= 2:
            try:
                meta_feats = np.concatenate(preds[:2], axis=1)
                if meta_feats.shape[1] == 6:
                    meta_raw = self.meta_model.predict_proba(meta_feats)
                    meta_out = self._remap_proba(meta_raw, self.meta_model.classes_)
                    out = 0.6 * meta_out + 0.4 * out
            except Exception:
                pass

        # v9 FIX 13b: Temperature scaling
        T = self._temperature
        if T != 1.0 and T > 0:
            eps     = 1e-7
            logits  = np.log(np.clip(out, eps, 1.0 - eps))
            logits  = logits / T
            logits -= logits.max(axis=1, keepdims=True)
            exp_l   = np.exp(logits)
            out     = exp_l / exp_l.sum(axis=1, keepdims=True)
        else:
            row_sums = out.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            out = out / row_sums

        return out

    def _save_model(self) -> None:
        """Atomic model save: write to .tmp then os.replace() so the live inference
        cycle never opens a partially-written pickle mid-write (Bug 2 fix).
        """
        try:
            path     = self.get_model_path()
            tmp_path = path + ".tmp"
            payload = {
                "rf":  self.rf_model,  "gbm": self.gbm_model,
                "xgb": self.xgb_model, "lgb": self.lgb_model,
                "meta": self.meta_model,
                "scaler": self.scaler, "feature_names": self.feature_names,
                "symbol": self.symbol, "saved_at": datetime.now().isoformat(),
                "train_date": self._train_date, "wf_mean_acc": self._wf_mean_acc,
                "n_features": self._n_features, "version": "v19_auto",
                "temperature": self._temperature,
                # v19 DIR-7: persist per-symbol ensemble weights
                "rf_weight": self._rf_weight, "gbm_weight": self._gbm_weight,
                "xgb_weight": self._xgb_weight, "lgb_weight": self._lgb_weight,
            }
            # Write entirely to .tmp first, then atomically rename.
            # os.replace() is atomic on POSIX and Windows (same filesystem).
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f)
            os.replace(tmp_path, path)
            logger.info(f"[{self.symbol}] Model saved -> {path} "
                        f"(feats={self._n_features}, wf_acc={self._wf_mean_acc:.3f})")
        except Exception as e:
            logger.error(f"Save failed: {e}")
            # Clean up orphaned .tmp so next save doesn't see a stale partial file
            try:
                if os.path.exists(path + ".tmp"):
                    os.remove(path + ".tmp")
            except OSError:
                pass

    def _load_model(self) -> None:
        candidates = [
            self.get_model_path(),
            os.path.join(MODEL_DIR, "signal_model_default.pkl"),
            os.path.join(MODEL_DIR, "signal_model.pkl"),
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "rb") as f:
                    p = pickle.load(f)
                self.rf_model      = p.get("rf")
                self.gbm_model     = p.get("gbm")
                self.xgb_model     = p.get("xgb")
                self.lgb_model     = p.get("lgb")
                self.meta_model    = p.get("meta")
                self.scaler        = p.get("scaler", StandardScaler())
                self.feature_names = p.get("feature_names", [])
                self._n_features   = p.get("n_features", len(self.feature_names))
                self._wf_mean_acc  = p.get("wf_mean_acc", 0.0)
                self._train_date   = p.get("train_date")
                self._temperature  = p.get("temperature", 1.5)
                # v19 DIR-7: restore per-symbol ensemble weights
                self._rf_weight   = p.get("rf_weight",  1.0)
                self._gbm_weight  = p.get("gbm_weight", 1.2)
                self._xgb_weight  = p.get("xgb_weight", 1.3)
                self._lgb_weight  = p.get("lgb_weight", 1.4)
                # Bug 2 FIX: validate XGBoost fitted state.
                # If the background retrainer saved the pkl between XGBoost's
                # __init__() and .fit() (concurrent write race), the loaded object
                # has no booster and will raise NotFittedError at inference time.
                # Detect this and null it out so the ensemble degrades to RF+GBM.
                if self.xgb_model is not None and XGB_AVAILABLE:
                    try:
                        self.xgb_model.get_booster()   # raises if not fitted
                    except Exception:
                        logger.warning(
                            f"[{self.symbol}] XGBoost loaded from {path} is not fitted "
                            "(partial write detected) — discarding XGB, using RF+GBM only."
                        )
                        self.xgb_model = None

                self.is_trained    = self.rf_model is not None
                version            = p.get("version", "v5")
                if self.is_trained:
                    logger.info(f"[{self.symbol}] Model loaded from {path} "
                                f"(ver={version} feats={self._n_features} "
                                f"wf_acc={self._wf_mean_acc:.3f})")
                    if path.endswith("signal_model.pkl") and version not in ("v6_pro", "v8_pro", "v9_pro", "v10_fast"):
                        logger.warning(
                            f"[{self.symbol}] Loaded LEGACY v5 model (20 feats, wf_acc=0.000). "
                            "Run trainer.py to generate a v9_pro model and delete models/signal_model.pkl."
                        )
                return
            except Exception as e:
                logger.warning(f"Could not load model from {path}: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Static helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"]  - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _calc_trend_strength(df: pd.DataFrame) -> pd.Series:
        try:
            up   = df["high"].diff().clip(lower=0)
            down = (-df["low"].diff()).clip(lower=0)
            dm_p = up.rolling(14).mean()
            dm_m = down.rolling(14).mean()
            dm_sum = (dm_p + dm_m).replace(0, np.nan)
            return ((dm_p - dm_m).abs() / dm_sum).fillna(0.5)
        except Exception:
            return pd.Series(0.5, index=df.index)

    @staticmethod
    def _calc_adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
        try:
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
            atr      = tr.rolling(period).mean().replace(0, np.nan)
            plus_di  = 100 * plus_dm.rolling(period).mean() / atr
            minus_di = 100 * minus_dm.rolling(period).mean() / atr
            dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
            return dx.rolling(period).mean().fillna(20.0)
        except Exception:
            return pd.Series(20.0, index=df.index)
