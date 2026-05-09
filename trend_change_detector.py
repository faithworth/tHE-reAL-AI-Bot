"""
trend_change_detector.py — Short-Term Trend Change & Reversal Detector (AI EA v21)
=====================================================================================
Solves the "buying twice in a downtrend" problem visible in the charts:
  • EA entered BUY twice during a clear TRENDING_BEAR phase (US30, H1)
  • Both lost; next day it resumed buying (late reversal catch)
  • Root cause: regime_detector only uses ADX/DI on H1; misses intraday
    structure breaks (CHoCH / BOS) that signal the trend has already flipped.

What this module adds
----------------------
  1. TrendChangeDetector.analyse(df_h1, df_h4, df_m15) → TrendChangeSnapshot
       Combines four fast signals:
         a. CHoCH / BOS  — structural higher-highs / lower-lows breaks
         b. EMA cross    — short EMA crosses below long EMA (or vice versa)
         c. RSI regime   — RSI < 45 (bearish zone) vs > 55 (bullish zone)
         d. Price vs VWAP — price below VWAP = bearish intraday bias

  2. DirectionBias enum:  BULL | BEAR | NEUTRAL | TRANSITION_TO_BULL | TRANSITION_TO_BEAR
       TRANSITION states are the key addition — they block counter-trend entries
       BEFORE the new regime is fully confirmed (prevents the "bought twice" loss).

  3. Integration hook:
       In ai_ea.py _process_symbol():
         bias = trend_change_det.analyse(df, df_h4, df_m15)
         if bias.direction == DirectionBias.BEAR and signal == "BUY":
             logger.info("TrendChange BEAR bias — BUY blocked")
             return
         if bias.direction == DirectionBias.TRANSITION_TO_BEAR and signal == "BUY":
             score = round(score * 0.60, 4)  # heavy penalty, likely filtered by score gate

Usage
-----
    from trend_change_detector import TrendChangeDetector, DirectionBias
    tcd = TrendChangeDetector()
    snap = tcd.analyse(df_h1, df_h4=df_h4, df_m15=df_m15)
    # snap.direction, snap.confidence, snap.reasons
"""

import logging
import warnings

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List

# Suppress FutureWarning from pandas rolling operations on older numpy
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class DirectionBias(Enum):
    BULL               = "bull"
    BEAR               = "bear"
    NEUTRAL            = "neutral"
    TRANSITION_TO_BULL = "transition_to_bull"   # bear → bull flip in progress
    TRANSITION_TO_BEAR = "transition_to_bear"   # bull → bear flip in progress


@dataclass
class TrendChangeSnapshot:
    direction:   DirectionBias  = DirectionBias.NEUTRAL
    confidence:  float          = 0.0          # 0–1
    short_bias:  str            = "neutral"    # "bull" | "bear" | "neutral" (H1-level)
    htf_bias:    str            = "neutral"    # "bull" | "bear" | "neutral" (H4-level)
    choch:       bool           = False        # Change of Character detected
    bos:         bool           = False        # Break of Structure detected
    ema_cross:   str            = "none"       # "bull_cross" | "bear_cross" | "none"
    rsi_bias:    str            = "neutral"    # "bull" | "bear" | "neutral"
    vwap_bias:   str            = "neutral"    # "above" | "below" | "neutral"
    reasons:     List[str]      = field(default_factory=list)
    block_buy:   bool           = False        # convenience: True when BEAR confirmed
    block_sell:  bool           = False        # convenience: True when BULL confirmed


class TrendChangeDetector:
    """
    Detects intraday trend changes and reversals to prevent entering
    against the dominant short-term direction.

    Parameters
    ----------
    ema_fast        : Fast EMA period (default 9)
    ema_slow        : Slow EMA period (default 21)
    rsi_period      : RSI period (default 14)
    rsi_bull_zone   : RSI above this = bullish bias (default 55)
    rsi_bear_zone   : RSI below this = bearish bias (default 45)
    swing_lookback  : Bars to look back for swing highs/lows (default 20)
    htf_weight      : How much H4 bias overrides H1 (0–1, default 0.40)
    transition_conf : Confidence threshold above which TRANSITION becomes BEAR/BULL (default 0.65)
    """

    def __init__(
        self,
        ema_fast:        int   = 9,
        ema_slow:        int   = 21,
        rsi_period:      int   = 14,
        rsi_bull_zone:   float = 55.0,
        rsi_bear_zone:   float = 45.0,
        swing_lookback:  int   = 20,
        htf_weight:      float = 0.40,
        transition_conf: float = 0.65,
    ):
        self.ema_fast        = ema_fast
        self.ema_slow        = ema_slow
        self.rsi_period      = rsi_period
        self.rsi_bull_zone   = rsi_bull_zone
        self.rsi_bear_zone   = rsi_bear_zone
        self.swing_lookback  = swing_lookback
        self.htf_weight      = htf_weight
        self.transition_conf = transition_conf

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def analyse(
        self,
        df_h1:  pd.DataFrame,
        df_h4:  Optional[pd.DataFrame] = None,
        df_m15: Optional[pd.DataFrame] = None,
    ) -> TrendChangeSnapshot:
        """
        Main entry point. Returns TrendChangeSnapshot with direction bias.

        Priority logic:
          1. If H4 is clearly BEAR and H1 just flipped to BEAR  → BEAR (block BUY)
          2. If H4 is BEAR but H1 is still BULL                 → TRANSITION_TO_BEAR
          3. If both H1 and H4 are BULL                         → BULL (block SELL)
          4. Mixed / insufficient signals                        → NEUTRAL
        """
        snap = TrendChangeSnapshot()

        try:
            if df_h1 is None or len(df_h1) < max(self.ema_slow * 2, 30):
                snap.reasons.append("insufficient_h1_data")
                return snap

            # ── H1 analysis ─────────────────────────────────────────────────
            h1_score, h1_reasons = self._analyse_timeframe(df_h1, "H1")
            snap.short_bias = self._score_to_bias(h1_score)
            snap.reasons.extend(h1_reasons)

            # ── CHoCH / BOS on H1 ───────────────────────────────────────────
            choch, bos, struct_reasons = self._detect_structure_break(df_h1)
            snap.choch = choch
            snap.bos   = bos
            snap.reasons.extend(struct_reasons)

            # ── EMA cross direction ─────────────────────────────────────────
            snap.ema_cross = self._ema_cross_direction(df_h1)
            if snap.ema_cross == "bear_cross":
                h1_score -= 0.25
                snap.reasons.append("H1_ema_bear_cross")
            elif snap.ema_cross == "bull_cross":
                h1_score += 0.25
                snap.reasons.append("H1_ema_bull_cross")

            # CHoCH/BOS score adjustment
            if snap.choch:
                # CHoCH is a strong reversal signal — weight heavily
                if snap.short_bias == "bull" or h1_score > 0:
                    h1_score -= 0.35   # bull structure broken bearishly
                    snap.reasons.append("H1_CHoCH_bear_reversal")
                else:
                    h1_score += 0.35   # bear structure broken bullishly
                    snap.reasons.append("H1_CHoCH_bull_reversal")
            if snap.bos:
                if snap.ema_cross == "bear_cross" or h1_score < 0:
                    h1_score -= 0.15
                    snap.reasons.append("H1_BOS_bear_continuation")
                else:
                    h1_score += 0.15
                    snap.reasons.append("H1_BOS_bull_continuation")

            # ── H4 analysis (higher-timeframe context) ───────────────────────
            h4_score = 0.0
            if df_h4 is not None and len(df_h4) >= max(self.ema_slow * 2, 30):
                h4_score, h4_reasons = self._analyse_timeframe(df_h4, "H4")
                snap.htf_bias = self._score_to_bias(h4_score)
                snap.reasons.extend(h4_reasons)

            # ── M15 quick confirmation (optional) ───────────────────────────
            m15_score = 0.0
            if df_m15 is not None and len(df_m15) >= max(self.ema_slow * 2, 30):
                m15_score, m15_reasons = self._analyse_timeframe(df_m15, "M15")
                snap.reasons.extend(m15_reasons)

            # ── Composite score ──────────────────────────────────────────────
            # Weights: H4=40%, H1=45%, M15=15%
            h1_w  = 0.45
            h4_w  = self.htf_weight          # 0.40
            m15_w = 1.0 - h1_w - h4_w       # 0.15

            composite = (h1_score * h1_w) + (h4_score * h4_w) + (m15_score * m15_w)
            composite  = max(-1.0, min(1.0, composite))
            snap.confidence = abs(composite)

            # ── RSI and VWAP for quick supplementary signals ─────────────────
            rsi_bias  = self._rsi_bias(df_h1)
            vwap_bias = self._vwap_bias(df_h1)
            snap.rsi_bias  = rsi_bias
            snap.vwap_bias = vwap_bias

            # Minor adjustments from RSI / VWAP
            if rsi_bias == "bear":
                composite -= 0.08
                snap.reasons.append("H1_rsi_bearish_zone")
            elif rsi_bias == "bull":
                composite += 0.08
                snap.reasons.append("H1_rsi_bullish_zone")
            if vwap_bias == "below":
                composite -= 0.05
                snap.reasons.append("H1_price_below_vwap")
            elif vwap_bias == "above":
                composite += 0.05
                snap.reasons.append("H1_price_above_vwap")

            composite = max(-1.0, min(1.0, composite))
            snap.confidence = abs(composite)

            # ── Direction decision ───────────────────────────────────────────
            snap.direction = self._decide_direction(composite, snap)

            # ── Convenience block flags ──────────────────────────────────────
            snap.block_buy  = snap.direction in (
                DirectionBias.BEAR, DirectionBias.TRANSITION_TO_BEAR
            )
            snap.block_sell = snap.direction in (
                DirectionBias.BULL, DirectionBias.TRANSITION_TO_BULL
            )

            logger.info(
                f"[TrendChange] {snap.direction.value} | conf={snap.confidence:.2f} | "
                f"h1={snap.short_bias} h4={snap.htf_bias} | "
                f"choch={snap.choch} bos={snap.bos} ema={snap.ema_cross} | "
                f"block_buy={snap.block_buy} block_sell={snap.block_sell}"
            )

        except Exception as exc:
            logger.error(f"TrendChangeDetector.analyse error: {exc}", exc_info=True)
            snap.reasons.append(f"error:{exc}")

        return snap

    # ─────────────────────────────────────────────────────────────────────────
    # Internal analysis helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _analyse_timeframe(
        self,
        df: pd.DataFrame,
        label: str,
    ):
        """
        Score a single timeframe's directional bias.
        Returns (score: float [-1,+1], reasons: list[str])
        score > 0 = bullish, score < 0 = bearish
        """
        reasons = []
        score   = 0.0

        try:
            close = df["close"].astype(float)
            high  = df["high"].astype(float)
            low   = df["low"].astype(float)

            # ── EMA structure ────────────────────────────────────────────────
            ema_fast = close.ewm(span=self.ema_fast, adjust=False).mean()
            ema_slow = close.ewm(span=self.ema_slow, adjust=False).mean()
            if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
                score += 0.30
                reasons.append(f"{label}_ema_bull_stack")
            else:
                score -= 0.30
                reasons.append(f"{label}_ema_bear_stack")

            # ── Higher-high / lower-low structure (last N swings) ────────────
            n = min(self.swing_lookback, len(df) - 2)
            recent_highs = high.iloc[-n:].values
            recent_lows  = low.iloc[-n:].values
            mid = n // 2

            # Compare first half vs second half
            first_half_h  = recent_highs[:mid].max()  if mid > 0 else float(high.mean())
            second_half_h = recent_highs[mid:].max()  if mid > 0 else float(high.mean())
            first_half_l  = recent_lows[:mid].min()   if mid > 0 else float(low.mean())
            second_half_l = recent_lows[mid:].min()   if mid > 0 else float(low.mean())

            hh = second_half_h > first_half_h
            hl = second_half_l > first_half_l
            lh = second_half_h < first_half_h
            ll = second_half_l < first_half_l

            if hh and hl:
                score += 0.35
                reasons.append(f"{label}_HH_HL_bull_structure")
            elif lh and ll:
                score -= 0.35
                reasons.append(f"{label}_LH_LL_bear_structure")
            elif lh and hl:
                # Lower high + higher low = compression/inside = neutral
                reasons.append(f"{label}_compression_neutral")
            elif hh and ll:
                # Higher high + lower low = expansion = no clear bias
                reasons.append(f"{label}_expansion_no_bias")

            # ── Linear regression slope ──────────────────────────────────────
            n_slope = min(30, len(close))
            x       = np.arange(n_slope)
            y       = close.iloc[-n_slope:].values
            if len(y) >= 3:
                coeffs = np.polyfit(x, y, 1)
                slope_pct = coeffs[0] / (float(close.mean()) + 1e-12)
                if slope_pct > 0.0002:
                    score += 0.15
                    reasons.append(f"{label}_positive_slope")
                elif slope_pct < -0.0002:
                    score -= 0.15
                    reasons.append(f"{label}_negative_slope")

        except Exception as exc:
            logger.debug(f"_analyse_timeframe({label}) error: {exc}")

        return float(max(-1.0, min(1.0, score))), reasons

    def _detect_structure_break(self, df: pd.DataFrame):
        """
        Detect CHoCH (Change of Character) and BOS (Break of Structure).
        CHoCH: previous HH broken by LL, or previous LL broken by HH
        BOS: clean continuation break beyond prior swing
        Returns (choch: bool, bos: bool, reasons: list)
        """
        choch   = False
        bos     = False
        reasons = []

        try:
            if len(df) < 10:
                return False, False, ["insufficient_data_for_structure"]

            high  = df["high"].astype(float)
            low   = df["low"].astype(float)
            close = df["close"].astype(float)

            n = min(self.swing_lookback, len(df) - 4)

            # Find swing highs and lows using rolling window
            swing_highs = high.rolling(5, center=True).max()
            swing_lows  = low.rolling(5, center=True).min()

            # Recent swing high and low (last N bars)
            recent_swing_h = float(swing_highs.iloc[-n:].max())
            recent_swing_l = float(swing_lows.iloc[-n:].min())
            prev_swing_h   = float(swing_highs.iloc[-n*2:-n].max()) if len(df) >= n*2 else recent_swing_h
            prev_swing_l   = float(swing_lows.iloc[-n*2:-n].min())  if len(df) >= n*2 else recent_swing_l

            current_close  = float(close.iloc[-1])
            prev_close     = float(close.iloc[-5])  # 5 bars ago

            # BOS detection: price breaks beyond prior swing extreme
            if current_close > prev_swing_h * 1.001:
                bos = True
                reasons.append(f"BOS_bull_break_above_{prev_swing_h:.5f}")
            elif current_close < prev_swing_l * 0.999:
                bos = True
                reasons.append(f"BOS_bear_break_below_{prev_swing_l:.5f}")

            # CHoCH detection: recent trend extreme is broken in opposite direction
            # Bullish trend broken: had HH but now price makes LL below recent swing low
            had_bull_structure = (recent_swing_h > prev_swing_h) and (prev_close > prev_swing_l)
            if had_bull_structure and current_close < recent_swing_l * 0.999:
                choch = True
                reasons.append(f"CHoCH_bull_to_bear:close={current_close:.5f}<swing_l={recent_swing_l:.5f}")

            # Bearish trend broken: had LL but now price makes HH above recent swing high
            had_bear_structure = (recent_swing_l < prev_swing_l) and (prev_close < prev_swing_h)
            if had_bear_structure and current_close > recent_swing_h * 1.001:
                choch = True
                reasons.append(f"CHoCH_bear_to_bull:close={current_close:.5f}>swing_h={recent_swing_h:.5f}")

        except Exception as exc:
            logger.debug(f"_detect_structure_break error: {exc}")

        return choch, bos, reasons

    def _ema_cross_direction(self, df: pd.DataFrame) -> str:
        """
        Detect if a fresh EMA cross occurred in the last 3 bars.
        Returns "bull_cross", "bear_cross", or "none".
        """
        try:
            close    = df["close"].astype(float)
            ema_fast = close.ewm(span=self.ema_fast, adjust=False).mean()
            ema_slow = close.ewm(span=self.ema_slow, adjust=False).mean()

            diff = ema_fast - ema_slow
            if len(diff) < 4:
                return "none"

            # Check last 3 bars for a sign change
            for i in range(-3, 0):
                prev = diff.iloc[i - 1]
                curr = diff.iloc[i]
                if prev < 0 and curr >= 0:
                    return "bull_cross"
                elif prev > 0 and curr <= 0:
                    return "bear_cross"
        except Exception:
            pass
        return "none"

    def _rsi_bias(self, df: pd.DataFrame) -> str:
        """Returns 'bull', 'bear', or 'neutral' based on RSI zone."""
        try:
            close  = df["close"].astype(float)
            delta  = close.diff()
            gain   = delta.clip(lower=0)
            loss   = (-delta).clip(lower=0)
            avg_g  = gain.rolling(self.rsi_period).mean()
            avg_l  = loss.rolling(self.rsi_period).mean()
            rs     = avg_g / (avg_l.replace(0, np.nan))
            rsi    = 100 - (100 / (1 + rs))
            val    = float(rsi.iloc[-1])
            if val > self.rsi_bull_zone:
                return "bull"
            elif val < self.rsi_bear_zone:
                return "bear"
        except Exception:
            pass
        return "neutral"

    def _vwap_bias(self, df: pd.DataFrame) -> str:
        """
        Returns 'above' (bullish) or 'below' (bearish) based on whether
        current close is above or below the rolling VWAP.
        Uses a 24-bar rolling VWAP as intraday proxy.
        """
        try:
            close  = df["close"].astype(float)
            high   = df["high"].astype(float)
            low    = df["low"].astype(float)
            vol    = df.get("tick_volume", df.get("volume", pd.Series(
                np.ones(len(df)), index=df.index
            ))).astype(float)

            typical = (high + low + close) / 3.0
            n       = min(24, len(df))
            vwap    = (typical * vol).rolling(n).sum() / vol.rolling(n).sum()

            if float(close.iloc[-1]) > float(vwap.iloc[-1]):
                return "above"
            else:
                return "below"
        except Exception:
            pass
        return "neutral"

    def _score_to_bias(self, score: float) -> str:
        if score > 0.15:
            return "bull"
        elif score < -0.15:
            return "bear"
        return "neutral"

    def _decide_direction(self, composite: float, snap: TrendChangeSnapshot) -> DirectionBias:
        """
        Map composite score + structural signals to a DirectionBias enum.

        Decision matrix:
          composite >= +0.45  AND H4 bull  → BULL
          composite >= +0.20               → TRANSITION_TO_BULL (if H4 was bear)
          composite <= -0.45  AND H4 bear  → BEAR
          composite <= -0.20               → TRANSITION_TO_BEAR (if H4 was bull)
          else                             → NEUTRAL
        """
        htf_bull = snap.htf_bias == "bull"
        htf_bear = snap.htf_bias == "bear"

        # Strong BEAR: both H1 and H4 bearish
        if composite <= -0.45 and (htf_bear or snap.htf_bias == "neutral"):
            return DirectionBias.BEAR
        if composite <= -0.45 and htf_bear:
            return DirectionBias.BEAR

        # Strong BULL: both H1 and H4 bullish
        if composite >= 0.45 and (htf_bull or snap.htf_bias == "neutral"):
            return DirectionBias.BULL
        if composite >= 0.45 and htf_bull:
            return DirectionBias.BULL

        # Transition to BEAR: H1 bearish, H4 still neutral or mixed
        if composite <= -0.20:
            if snap.choch:
                # CHoCH confirms structural flip — stronger signal
                return DirectionBias.TRANSITION_TO_BEAR
            if snap.ema_cross == "bear_cross":
                return DirectionBias.TRANSITION_TO_BEAR
            if htf_bear:
                return DirectionBias.BEAR
            return DirectionBias.TRANSITION_TO_BEAR if snap.confidence >= 0.25 else DirectionBias.NEUTRAL

        # Transition to BULL: H1 bullish, H4 still neutral or mixed
        if composite >= 0.20:
            if snap.choch:
                return DirectionBias.TRANSITION_TO_BULL
            if snap.ema_cross == "bull_cross":
                return DirectionBias.TRANSITION_TO_BULL
            if htf_bull:
                return DirectionBias.BULL
            return DirectionBias.TRANSITION_TO_BULL if snap.confidence >= 0.25 else DirectionBias.NEUTRAL

        return DirectionBias.NEUTRAL
