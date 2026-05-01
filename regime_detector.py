"""
regime_detector.py — Market Regime Classifier (AI EA v17)
=========================================================
Identifies current market regime and dynamically adjusts strategy parameters.

Regimes:
  TRENDING_BULL    — strong directional uptrend
  TRENDING_BEAR    — strong directional downtrend
  RANGING          — oscillating, no directional bias (generic)
  RANGING_SCALP    — confirmed high-quality range → LTF scalp mode activated
  BREAKOUT         — just exited a range, high momentum
  REVERSAL         — potential exhaustion / regime change
  VOLATILE         — high ATR, spike conditions
  DEAD             — ultra-low volatility, Asian session rest

v7 upgrade:
  - RANGING_SCALP sub-regime: triggered when H4+H1 both range AND range quality
    is high enough to warrant drilling into LTFs for scalp entries.
  - Range quality scoring built into detector (tightness, touch count, age)
  - HTF/LTF regime divergence detection (e.g. trending on H4 but ranging on H1)
  - Each regime carries recommended parameter overrides

Each regime carries recommended parameter overrides:
  - min_signal_prob, sl_mult, tp_mult, max_trades, scoring_weights
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class Regime(Enum):
    TRENDING_BULL  = "trending_bull"
    TRENDING_BEAR  = "trending_bear"
    RANGING        = "ranging"
    RANGING_SCALP  = "ranging_scalp"   # v7: high-quality range → LTF drill
    BREAKOUT       = "breakout"
    REVERSAL       = "reversal"
    VOLATILE       = "volatile"
    DEAD           = "dead"
    UNKNOWN        = "unknown"


@dataclass
class RegimeConfig:
    """Per-regime parameter recommendations."""
    regime: Regime
    min_signal_prob: float  = 0.42   # v17: realistic for T=1.5 3-class model (peak ~0.45-0.65)
    sl_atr_mult: float      = 1.5
    tp_atr_mult: float      = 2.5
    max_trades_day: int     = 10
    risk_per_trade: float   = 0.007   # fraction of equity
    score_ml_weight: float  = 0.40
    score_trend_weight: float = 0.30
    score_structure_weight: float = 0.20
    score_session_weight: float = 0.10
    trade_allowed: bool     = True
    notes: str              = ""


# Pre-tuned configs per regime
REGIME_CONFIGS: Dict[Regime, RegimeConfig] = {
    Regime.TRENDING_BULL: RegimeConfig(
        regime=Regime.TRENDING_BULL,
        min_signal_prob=0.40,   # v17: T=1.5 3-class model; bull trend = lower bar (with-trend)
        sl_atr_mult=1.2,
        tp_atr_mult=3.0,
        max_trades_day=8,
        risk_per_trade=0.008,
        score_ml_weight=0.35,
        score_trend_weight=0.40,
        score_structure_weight=0.15,
        score_session_weight=0.10,
        notes="Trend following: wider TP, tighter SL"
    ),
    Regime.TRENDING_BEAR: RegimeConfig(
        regime=Regime.TRENDING_BEAR,
        min_signal_prob=0.40,   # v17: with-trend bear — same as bull
        sl_atr_mult=1.2,
        tp_atr_mult=3.0,
        max_trades_day=8,
        risk_per_trade=0.008,
        score_ml_weight=0.35,
        score_trend_weight=0.40,
        score_structure_weight=0.15,
        score_session_weight=0.10,
        notes="Trend following short: wider TP, tighter SL"
    ),
    Regime.RANGING: RegimeConfig(
        regime=Regime.RANGING,
        min_signal_prob=0.44,   # v17: slightly higher than trending (counter-trend risk)
        sl_atr_mult=0.8,
        tp_atr_mult=1.5,
        max_trades_day=6,
        risk_per_trade=0.005,
        score_ml_weight=0.30,
        score_trend_weight=0.20,
        score_structure_weight=0.40,
        score_session_weight=0.10,
        notes="Range trading: tighter TP, higher confidence required"
    ),
    Regime.RANGING_SCALP: RegimeConfig(
        regime=Regime.RANGING_SCALP,
        min_signal_prob=0.38,       # v17: LTF scalper uses its own multi-confirm threshold
        sl_atr_mult=0.6,            # LTF ATR-based — very tight
        tp_atr_mult=2.5,            # TP2 runner to far side of range
        max_trades_day=12,          # scalps are smaller, allow more
        risk_per_trade=0.004,       # smaller risk per scalp
        score_ml_weight=0.25,
        score_trend_weight=0.10,    # trend doesn't matter much in a range
        score_structure_weight=0.45, # structure / OB / FVG matters most
        score_session_weight=0.20,  # session critical for scalps
        notes="High-quality range confirmed — LTF scalp mode active (M15/M5)"
    ),
    Regime.BREAKOUT: RegimeConfig(
        regime=Regime.BREAKOUT,
        min_signal_prob=0.45,   # v17: BREAKOUT
        sl_atr_mult=1.0,
        tp_atr_mult=4.0,
        max_trades_day=5,
        risk_per_trade=0.006,
        score_ml_weight=0.45,
        score_trend_weight=0.30,
        score_structure_weight=0.15,
        score_session_weight=0.10,
        notes="Breakout: very wide TP, momentum ride"
    ),
    Regime.REVERSAL: RegimeConfig(
        regime=Regime.REVERSAL,
        min_signal_prob=0.46,   # v17: REVERSAL
        sl_atr_mult=1.0,
        tp_atr_mult=2.0,
        max_trades_day=4,
        risk_per_trade=0.004,
        score_ml_weight=0.40,
        score_trend_weight=0.20,
        score_structure_weight=0.35,
        score_session_weight=0.05,
        notes="Reversal: high confidence required, reduced size"
    ),
    Regime.VOLATILE: RegimeConfig(
        regime=Regime.VOLATILE,
        min_signal_prob=0.48,   # v17: VOLATILE
        sl_atr_mult=2.0,
        tp_atr_mult=2.5,
        max_trades_day=3,
        risk_per_trade=0.003,
        score_ml_weight=0.40,
        score_trend_weight=0.25,
        score_structure_weight=0.25,
        score_session_weight=0.10,
        notes="High volatility: wider SL, small size, very selective"
    ),
    Regime.DEAD: RegimeConfig(
        regime=Regime.DEAD,
        min_signal_prob=0.55,   # v17: DEAD (trading suspended anyway)
        sl_atr_mult=1.5,
        tp_atr_mult=2.0,
        max_trades_day=2,
        risk_per_trade=0.003,
        trade_allowed=False,
        notes="Dead market: trading suspended"
    ),
    Regime.UNKNOWN: RegimeConfig(
        regime=Regime.UNKNOWN,
        min_signal_prob=0.42,   # v17: UNKNOWN — conservative default
        trade_allowed=True,
        notes="Unknown regime: using conservative defaults"
    ),
}


@dataclass
class RegimeSnapshot:
    regime: Regime
    confidence: float         = 0.0      # 0–1
    config: Optional[RegimeConfig] = None
    atr_percentile: float     = 0.5
    adx: float                = 0.0
    trend_slope: float        = 0.0      # linear regression slope
    volatility_z: float       = 0.0      # volatility z-score vs recent history
    range_tightness: float    = 0.0      # high-low compression
    breakout_strength: float  = 0.0
    # v7: range quality metadata
    range_quality: float      = 0.0      # 0-1, used to trigger RANGING_SCALP
    range_high: float         = 0.0      # identified range top
    range_low: float          = 0.0      # identified range bottom
    range_width_atr: float    = 0.0      # range width in ATR units
    htf_ranging: bool         = False    # H4 also ranging (cross-TF confirmation)
    scalp_mode: bool          = False    # True when RANGING_SCALP is active
    reasons: list             = field(default_factory=list)
    timestamp: str            = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RegimeDetector:
    """
    Classifies market regime from OHLCV data.
    v7: accepts optional df_h4 to cross-validate ranging on H4 and trigger
        RANGING_SCALP sub-regime when range quality is high enough.
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,
        adx_strong_threshold: float = 40.0,
        vol_lookback: int = 100,
        vol_spike_z: float = 2.0,
        vol_dead_z: float = -1.5,
        regime_history: int = 5,
        # v7: range scalp thresholds
        range_quality_scalp_min: float = 0.45,  # min quality to trigger RANGING_SCALP
        range_min_touches: int = 2,              # min touches each side to qualify
        range_min_width_atr: float = 1.5,
        range_max_width_atr: float = 7.0,
    ):
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_strong_threshold = adx_strong_threshold
        self.vol_lookback = vol_lookback
        self.vol_spike_z = vol_spike_z
        self.vol_dead_z = vol_dead_z
        self._history: list = []
        self._history_limit = regime_history
        self.range_quality_scalp_min = range_quality_scalp_min
        self.range_min_touches       = range_min_touches
        self.range_min_width_atr     = range_min_width_atr
        self.range_max_width_atr     = range_max_width_atr

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def detect(self, df: pd.DataFrame, df_h4: Optional[pd.DataFrame] = None) -> "RegimeSnapshot":
        """
        Main entry: classify regime from H1 OHLCV DataFrame.
        v7: accepts optional df_h4 for cross-TF range validation.
        Returns RegimeSnapshot with regime + recommended config.
        """
        snap = RegimeSnapshot(regime=Regime.UNKNOWN, confidence=0.0)
        try:
            if df is None or len(df) < max(self.adx_period * 3, 50):
                snap.reasons.append("insufficient_data")
                snap.config = REGIME_CONFIGS[Regime.UNKNOWN]
                return snap

            snap = self._classify(df)

            # v7: If RANGING detected on H1, check H4 and score range quality
            if snap.regime == Regime.RANGING:
                snap = self._evaluate_range_quality(snap, df, df_h4)

            snap.config = REGIME_CONFIGS.get(snap.regime, REGIME_CONFIGS[Regime.UNKNOWN])

            self._history.append(snap.regime)
            if len(self._history) > self._history_limit:
                self._history.pop(0)

            logger.info(
                f"[REGIME] {snap.regime.value} | confidence={snap.confidence:.2f} | "
                f"ADX={snap.adx:.1f} | vol_z={snap.volatility_z:.2f} | "
                f"scalp_mode={snap.scalp_mode} | range_q={snap.range_quality:.2f} | "
                f"reasons={snap.reasons}"
            )

        except Exception as e:
            logger.error(f"RegimeDetector.detect error: {e}", exc_info=True)
            snap.regime = Regime.UNKNOWN
            snap.config = REGIME_CONFIGS[Regime.UNKNOWN]

        return snap

    def get_config(self, snap: "RegimeSnapshot") -> RegimeConfig:
        return snap.config or REGIME_CONFIGS[Regime.UNKNOWN]

    def stable_regime(self) -> Optional[Regime]:
        """Return regime if it has been consistent over last N cycles."""
        if len(self._history) < self._history_limit:
            return None
        if len(set(self._history)) == 1:
            return self._history[-1]
        return None

    # ─────────────────────────────────────────────────────────────────
    # v7: Range quality evaluation → RANGING_SCALP upgrade
    # ─────────────────────────────────────────────────────────────────

    def _evaluate_range_quality(
        self,
        snap: "RegimeSnapshot",
        df_h1: pd.DataFrame,
        df_h4: Optional[pd.DataFrame],
    ) -> "RegimeSnapshot":
        """
        When H1 is RANGING, score the range quality.
        If quality >= threshold AND H4 is also ranging → upgrade to RANGING_SCALP.
        Populates snap.range_high/low/quality/scalp_mode.
        """
        try:
            atr = self._calc_atr(df_h1)
            if atr <= 0:
                return snap

            n   = min(40, len(df_h1))
            sub = df_h1.iloc[-n:]

            swing_h = float(sub["high"].rolling(5, center=True).max().dropna().max())
            swing_l = float(sub["low"].rolling(5, center=True).min().dropna().min())

            rng_w = swing_h - swing_l
            if rng_w <= 0:
                return snap

            width_atr = rng_w / atr

            # Range width gate
            if not (self.range_min_width_atr <= width_atr <= self.range_max_width_atr):
                snap.reasons.append(f"range_width_oob_{width_atr:.2f}atr")
                return snap

            # Touch counting
            tol = atr * 0.25
            touches_h = int((sub["high"] >= swing_h - tol).sum())
            touches_l = int((sub["low"]  <= swing_l + tol).sum())

            if touches_h < self.range_min_touches or touches_l < self.range_min_touches:
                snap.reasons.append(f"low_touches_H={touches_h}_L={touches_l}")
                return snap

            # Quality score
            touch_score = min((touches_h + touches_l) / 10.0, 0.4)
            width_score = 0.30 if 2.0 <= width_atr <= 5.0 else 0.12
            age_score   = 0.0
            # Age: bars since last close outside range
            age_bars = 0
            for i in range(len(sub) - 1, -1, -1):
                c = float(sub["close"].iloc[i])
                if c > swing_h or c < swing_l:
                    break
                age_bars += 1
            if 8 <= age_bars <= 50:
                age_score = 0.30
            elif 4 <= age_bars < 8:
                age_score = 0.15

            quality = touch_score + width_score + age_score

            snap.range_high       = round(swing_h, 5)
            snap.range_low        = round(swing_l, 5)
            snap.range_quality    = round(quality, 3)
            snap.range_width_atr  = round(width_atr, 2)

            # Cross-validate with H4: is H4 also ranging?
            htf_ranging = False
            if df_h4 is not None and len(df_h4) >= 30:
                h4_adx, _, _ = self._calc_adx(df_h4.iloc[-50:] if len(df_h4) >= 50 else df_h4)
                htf_ranging  = h4_adx < self.adx_trend_threshold
                snap.htf_ranging = htf_ranging
                if htf_ranging:
                    quality += 0.12   # bonus for H4 confirmation
                    snap.range_quality = round(min(quality, 1.0), 3)
                    snap.reasons.append("h4_ranging_confirmed")

            # Upgrade to RANGING_SCALP?
            if snap.range_quality >= self.range_quality_scalp_min:
                snap.regime    = Regime.RANGING_SCALP
                snap.scalp_mode = True
                snap.reasons.append(
                    f"range_scalp_activated_q={snap.range_quality:.2f}"
                    f"_H={touches_h}_L={touches_l}_age={age_bars}"
                )
                logger.info(
                    f"[REGIME] RANGING_SCALP activated: quality={snap.range_quality:.2f} "
                    f"range=[{snap.range_low:.5f}, {snap.range_high:.5f}] "
                    f"width={width_atr:.1f}atr htf_ranging={htf_ranging}"
                )
            else:
                snap.reasons.append(f"range_quality_too_low_{snap.range_quality:.2f}")

        except Exception as exc:
            logger.warning(f"_evaluate_range_quality error: {exc}")

        return snap

    # ─────────────────────────────────────────────────────────────────
    # Core classification
    # ─────────────────────────────────────────────────────────────────

    def _classify(self, df: pd.DataFrame) -> RegimeSnapshot:
        snap = RegimeSnapshot(regime=Regime.UNKNOWN)

        atr       = self._calc_atr(df)
        adx, plus_di, minus_di = self._calc_adx(df)
        vol_z     = self._vol_z_score(df)
        slope     = self._trend_slope(df)
        tightness = self._range_tightness(df, atr)
        breakout  = self._breakout_strength(df, atr)

        snap.atr_percentile = self._atr_percentile(df, atr)
        snap.adx            = adx
        snap.volatility_z   = vol_z
        snap.trend_slope    = slope
        snap.range_tightness = tightness
        snap.breakout_strength = breakout

        # ── Priority 1: Dead market ───────────────────────────────────
        if vol_z < self.vol_dead_z and adx < 15:
            snap.regime = Regime.DEAD
            snap.confidence = min(1.0, abs(vol_z) / 3.0)
            snap.reasons.append(f"dead_vol_z={vol_z:.2f}")
            return snap

        # ── Priority 2: Volatile spike ────────────────────────────────
        if vol_z > self.vol_spike_z:
            snap.regime = Regime.VOLATILE
            snap.confidence = min(1.0, vol_z / 4.0)
            snap.reasons.append(f"vol_spike_z={vol_z:.2f}")
            return snap

        # ── Priority 3: Strong trend ──────────────────────────────────
        if adx > self.adx_strong_threshold:
            if plus_di > minus_di:
                snap.regime = Regime.TRENDING_BULL
                snap.reasons.append(f"adx={adx:.1f}_bull")
            else:
                snap.regime = Regime.TRENDING_BEAR
                snap.reasons.append(f"adx={adx:.1f}_bear")
            snap.confidence = min(1.0, (adx - self.adx_strong_threshold) / 25.0 + 0.6)
            return snap

        # ── Priority 4: Breakout detection ────────────────────────────
        if breakout > 0.7 and adx > self.adx_trend_threshold:
            snap.regime = Regime.BREAKOUT
            snap.confidence = breakout
            snap.reasons.append(f"breakout_strength={breakout:.2f}")
            return snap

        # ── Priority 5: Moderate trend ───────────────────────────────
        if adx > self.adx_trend_threshold:
            if plus_di > minus_di and slope > 0:
                snap.regime = Regime.TRENDING_BULL
                snap.confidence = (adx - self.adx_trend_threshold) / (self.adx_strong_threshold - self.adx_trend_threshold)
                snap.reasons.append(f"moderate_bull_adx={adx:.1f}")
            elif minus_di > plus_di and slope < 0:
                snap.regime = Regime.TRENDING_BEAR
                snap.confidence = (adx - self.adx_trend_threshold) / (self.adx_strong_threshold - self.adx_trend_threshold)
                snap.reasons.append(f"moderate_bear_adx={adx:.1f}")
            else:
                snap.regime = Regime.RANGING
                snap.confidence = 0.4
                snap.reasons.append("conflicting_di")
            return snap

        # ── Priority 6: Reversal detection ───────────────────────────
        if self._is_reversal(df, adx, vol_z):
            snap.regime = Regime.REVERSAL
            snap.confidence = 0.5
            snap.reasons.append("reversal_pattern_detected")
            return snap

        # ── Default: Ranging ─────────────────────────────────────────
        snap.regime = Regime.RANGING
        snap.confidence = max(0.3, 1.0 - adx / 25.0)
        snap.reasons.append(f"ranging_adx={adx:.1f}_tight={tightness:.2f}")
        return snap

    def _is_reversal(self, df: pd.DataFrame, adx: float, vol_z: float) -> bool:
        """Check for exhaustion signals: RSI divergence proxy + ADX declining from high."""
        try:
            if len(df) < 30:
                return False
            # ADX recently was high but declining
            atr_recent = self._calc_adx(df.iloc[-20:])[0]
            atr_older  = self._calc_adx(df.iloc[-50:-20])[0]
            adx_declining = atr_older > self.adx_strong_threshold and atr_recent < atr_older * 0.8
            # Price made new extreme but volume/momentum diverged (simplified proxy)
            price_change = (float(df["close"].iloc[-1]) - float(df["close"].iloc[-20])) / float(df["close"].iloc[-20])
            momentum = df["close"].diff().rolling(10).mean().iloc[-1]
            divergence = abs(price_change) > 0.02 and abs(momentum) < abs(df["close"].diff().rolling(10).mean().iloc[-20]) * 0.5
            return adx_declining or divergence
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────────────
    # Indicator calculators
    # ─────────────────────────────────────────────────────────────────

    def _calc_atr(self, df: pd.DataFrame) -> float:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"]  - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(self.adx_period).mean().iloc[-1])

    def _calc_adx(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """Returns (ADX, +DI, -DI)."""
        try:
            period = min(self.adx_period, len(df) // 3)
            high, low, close = df["high"], df["low"], df["close"]
            plus_dm  = (high.diff()).clip(lower=0)
            minus_dm = (-low.diff()).clip(lower=0)
            # Zero out where opposite DM is larger
            mask = plus_dm > minus_dm
            plus_dm  = plus_dm.where(mask, 0)
            minus_dm = minus_dm.where(~mask, 0)

            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low  - close.shift()).abs(),
            ], axis=1).max(axis=1)

            atr14 = tr.rolling(period).mean().replace(0, np.nan)
            plus_di  = 100 * plus_dm.rolling(period).mean()  / atr14
            minus_di = 100 * minus_dm.rolling(period).mean() / atr14
            dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
            adx = dx.rolling(period).mean()

            return (
                float(adx.iloc[-1]),
                float(plus_di.iloc[-1]),
                float(minus_di.iloc[-1]),
            )
        except Exception:
            return 20.0, 15.0, 15.0

    def _vol_z_score(self, df: pd.DataFrame) -> float:
        """Z-score of current ATR vs rolling historical ATR."""
        try:
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"]  - df["close"].shift()).abs(),
            ], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean()
            lb = min(self.vol_lookback, len(atr14) - 20)
            if lb < 20:
                return 0.0
            hist = atr14.iloc[-lb:]
            mean_v = hist.mean()
            std_v  = hist.std()
            if std_v == 0:
                return 0.0
            return float((atr14.iloc[-1] - mean_v) / std_v)
        except Exception:
            return 0.0

    def _trend_slope(self, df: pd.DataFrame, period: int = 30) -> float:
        """Linear regression slope of close prices (normalised)."""
        try:
            closes = df["close"].iloc[-period:].values
            x = np.arange(len(closes))
            slope = np.polyfit(x, closes, 1)[0]
            return float(slope / closes.mean()) if closes.mean() != 0 else 0.0
        except Exception:
            return 0.0

    def _range_tightness(self, df: pd.DataFrame, atr: float, period: int = 20) -> float:
        """How compressed is the recent range vs ATR. Low = tight (ranging)."""
        try:
            recent = df.iloc[-period:]
            price_range = float(recent["high"].max() - recent["low"].min())
            expected = atr * period ** 0.5
            return max(0.0, 1.0 - price_range / expected) if expected > 0 else 0.0
        except Exception:
            return 0.0

    def _breakout_strength(self, df: pd.DataFrame, atr: float, lookback: int = 30) -> float:
        """Recent breakout from a prior range. 0=no breakout, 1=strong breakout."""
        try:
            prior = df.iloc[-lookback-5:-5]
            recent = df.iloc[-5:]
            if len(prior) < 10:
                return 0.0
            prior_high = float(prior["high"].max())
            prior_low  = float(prior["low"].min())
            current    = float(recent["close"].iloc[-1])
            if current > prior_high:
                return min(1.0, (current - prior_high) / (atr * 2))
            elif current < prior_low:
                return min(1.0, (prior_low - current) / (atr * 2))
            return 0.0
        except Exception:
            return 0.0

    def _atr_percentile(self, df: pd.DataFrame, current_atr: float) -> float:
        """Where does current ATR sit in its historical distribution?"""
        try:
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"]  - df["close"].shift()).abs(),
            ], axis=1).max(axis=1)
            hist = tr.rolling(14).mean().dropna()
            if len(hist) < 20:
                return 0.5
            return float((hist < current_atr).mean())
        except Exception:
            return 0.5
