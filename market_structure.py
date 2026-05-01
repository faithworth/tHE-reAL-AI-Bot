"""
market_structure.py
-------------------
Institutional market-structure analysis module.

Implements:
  - Break of Structure (BOS)
  - Change of Character (CHoCH)
  - Trend bias detection
  - Range vs trend detection
  - Liquidity sweep detection

All functions operate on a standard OHLCV DataFrame with a DatetimeIndex.

v7 FIX: is_trade_aligned_with_structure() no longer mindlessly blocks all
ranging-market trades.  In RANGING regime the ranging scalper handles
precision entries — but the trend engine also gets a chance when:
  a) A liquidity sweep is present (original logic — strong confirmation), OR
  b) structure_score >= 0.35 (some swing structure exists), OR
  c) MTF bias aligns with the signal (HTF context supports the trade).
This prevents the EA from going completely silent during consolidation
while still guarding against blindly fading strong trends.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class MarketStructureAnalyzer:
    """
    Stateless market-structure analyser.
    Call `analyse(df)` to get a full structure snapshot for the latest bar.
    """

    def __init__(
        self,
        swing_lookback: int = 10,
        bos_threshold: float = 0.0,   # extra pips above/below swing — 0 = exact break
        range_atr_mult: float = 1.5,  # max ATR multiple for "range" regime
    ):
        self.swing_lookback = swing_lookback
        self.bos_threshold = bos_threshold
        self.range_atr_mult = range_atr_mult

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, df: pd.DataFrame) -> Dict:
        """
        Run full market-structure analysis on the provided DataFrame.

        Returns
        -------
        dict with keys:
          trend          : 'bullish' | 'bearish' | 'ranging'
          bos            : True if a Break of Structure just occurred
          bos_direction  : 'bullish' | 'bearish' | None
          choch          : True if a Change of Character detected
          choch_direction: 'bullish' | 'bearish' | None
          liquidity_sweep: True if last bar swept a swing level
          sweep_direction: 'high' | 'low' | None
          latest_swing_high: float
          latest_swing_low : float
          structure_score  : float [0, 1] — overall quality of current structure
        """
        result = {
            "trend": "ranging",
            "bos": False,
            "bos_direction": None,
            "choch": False,
            "choch_direction": None,
            "liquidity_sweep": False,
            "sweep_direction": None,
            "latest_swing_high": 0.0,
            "latest_swing_low": 0.0,
            "structure_score": 0.0,
        }

        if df is None or len(df) < self.swing_lookback * 3:
            return result

        try:
            swings = self._find_swings(df)
            if not swings["highs"] and not swings["lows"]:
                return result

            result["latest_swing_high"] = swings["highs"][-1] if swings["highs"] else df["high"].max()
            result["latest_swing_low"] = swings["lows"][-1] if swings["lows"] else df["low"].min()

            trend = self._detect_trend(df, swings)
            result["trend"] = trend

            bos, bos_dir = self._detect_bos(df, swings, trend)
            result["bos"] = bos
            result["bos_direction"] = bos_dir

            choch, choch_dir = self._detect_choch(df, swings, trend)
            result["choch"] = choch
            result["choch_direction"] = choch_dir

            sweep, sweep_dir = self._detect_liquidity_sweep(df, swings)
            result["liquidity_sweep"] = sweep
            result["sweep_direction"] = sweep_dir

            result["structure_score"] = self._score_structure(result, df)

        except Exception as e:
            logger.error(f"MarketStructure.analyse error: {e}", exc_info=True)

        return result

    def is_trade_aligned_with_structure(
        self, signal: str, structure: Dict
    ) -> bool:
        """
        Returns True if the trade signal is aligned with market structure.

        Trending rules (unchanged):
          - Bullish trend  → allow BUY; allow SELL only on CHoCH confirmation
          - Bearish trend  → allow SELL; allow BUY only on CHoCH confirmation

        Ranging rules (v7 FIX — no longer mindlessly blocks):
          Allow the signal in a ranging market when ANY of:
            a) Liquidity sweep present (stop hunt / institutional entry)
            b) structure_score >= 0.35 (meaningful swing structure exists)
            c) MTF bias matches the signal (higher-timeframe context aligns)

        The RANGING_SCALP sub-regime (high-quality ranges) is handled
        entirely by RangingScalper and never reaches this function.
        """
        trend = structure.get("trend", "ranging")
        choch = structure.get("choch", False)
        choch_dir = structure.get("choch_direction")
        sweep = structure.get("liquidity_sweep", False)
        struct_score = structure.get("structure_score", 0.0)

        # MTF bias from analyse_with_mtf() enrichment
        mtf_bias = structure.get("mtf_bias", "neutral")

        if signal == "BUY":
            if trend == "bullish":
                return True
            if trend == "bearish" and choch and choch_dir == "bullish":
                return True
            if trend == "ranging":
                mtf_supports = mtf_bias == "bullish"
                return sweep or struct_score >= 0.35 or mtf_supports
            return False

        if signal == "SELL":
            if trend == "bearish":
                return True
            if trend == "bullish" and choch and choch_dir == "bearish":
                return True
            if trend == "ranging":
                mtf_supports = mtf_bias == "bearish"
                return sweep or struct_score >= 0.35 or mtf_supports
            return False

        return False  # NO_TRADE always passes as "not blocked" but no trade is placed

    # ------------------------------------------------------------------
    # Swing detection
    # ------------------------------------------------------------------

    def _find_swings(self, df: pd.DataFrame) -> Dict:
        """
        Identify swing highs and swing lows using a local-max / local-min
        approach with `swing_lookback` candles on each side.
        """
        lb = self.swing_lookback
        highs = []
        lows = []

        for i in range(lb, len(df) - lb):
            window_high = df["high"].iloc[i - lb: i + lb + 1]
            window_low = df["low"].iloc[i - lb: i + lb + 1]

            if df["high"].iloc[i] == window_high.max():
                highs.append((i, float(df["high"].iloc[i])))
            if df["low"].iloc[i] == window_low.min():
                lows.append((i, float(df["low"].iloc[i])))

        return {
            "highs": [h[1] for h in highs],
            "lows": [l[1] for l in lows],
            "high_idx": [h[0] for h in highs],
            "low_idx": [l[0] for l in lows],
        }

    # ------------------------------------------------------------------
    # Trend
    # ------------------------------------------------------------------

    def _detect_trend(self, df: pd.DataFrame, swings: Dict) -> str:
        """
        Higher-highs + higher-lows = bullish.
        Lower-highs + lower-lows = bearish.
        Otherwise ranging.
        """
        highs = swings["highs"]
        lows = swings["lows"]

        if len(highs) < 2 or len(lows) < 2:
            # Not enough swings — fall back to SMA comparison
            if len(df) >= 50:
                sma20 = df["close"].rolling(20).mean().iloc[-1]
                sma50 = df["close"].rolling(50).mean().iloc[-1]
                if sma20 > sma50:
                    return "bullish"
                if sma20 < sma50:
                    return "bearish"
            return "ranging"

        recent_highs = highs[-4:]
        recent_lows = lows[-4:]

        hh = all(recent_highs[i] > recent_highs[i - 1] for i in range(1, len(recent_highs)))
        hl = all(recent_lows[i] > recent_lows[i - 1] for i in range(1, len(recent_lows)))
        lh = all(recent_highs[i] < recent_highs[i - 1] for i in range(1, len(recent_highs)))
        ll = all(recent_lows[i] < recent_lows[i - 1] for i in range(1, len(recent_lows)))

        if hh and hl:
            return "bullish"
        if lh and ll:
            return "bearish"
        return "ranging"

    # ------------------------------------------------------------------
    # BOS — Break of Structure
    # ------------------------------------------------------------------

    def _detect_bos(
        self, df: pd.DataFrame, swings: Dict, trend: str
    ) -> Tuple[bool, Optional[str]]:
        """
        BOS: price closes BEYOND the most recent swing high (bullish BOS)
        or swing low (bearish BOS) in the direction of the prevailing trend.
        """
        last_close = float(df["close"].iloc[-1])
        highs = swings["highs"]
        lows = swings["lows"]

        if trend == "bullish" and highs:
            prev_high = highs[-1]
            if last_close > prev_high + self.bos_threshold:
                logger.debug(f"Bullish BOS: close={last_close:.5f} > swing_high={prev_high:.5f}")
                return True, "bullish"

        if trend == "bearish" and lows:
            prev_low = lows[-1]
            if last_close < prev_low - self.bos_threshold:
                logger.debug(f"Bearish BOS: close={last_close:.5f} < swing_low={prev_low:.5f}")
                return True, "bearish"

        return False, None

    # ------------------------------------------------------------------
    # CHoCH — Change of Character
    # ------------------------------------------------------------------

    def _detect_choch(
        self, df: pd.DataFrame, swings: Dict, trend: str
    ) -> Tuple[bool, Optional[str]]:
        """
        CHoCH: price breaks a swing level AGAINST the prevailing trend,
        signalling a potential trend reversal.
        """
        last_close = float(df["close"].iloc[-1])
        highs = swings["highs"]
        lows = swings["lows"]

        # In a bearish trend: bullish CHoCH when price breaks above swing high
        if trend == "bearish" and highs:
            prev_high = highs[-1]
            if last_close > prev_high:
                logger.debug(f"Bullish CHoCH in bearish trend: close={last_close:.5f} > swing_high={prev_high:.5f}")
                return True, "bullish"

        # In a bullish trend: bearish CHoCH when price breaks below swing low
        if trend == "bullish" and lows:
            prev_low = lows[-1]
            if last_close < prev_low:
                logger.debug(f"Bearish CHoCH in bullish trend: close={last_close:.5f} < swing_low={prev_low:.5f}")
                return True, "bearish"

        return False, None

    # ------------------------------------------------------------------
    # Liquidity sweep
    # ------------------------------------------------------------------

    def _detect_liquidity_sweep(
        self, df: pd.DataFrame, swings: Dict
    ) -> Tuple[bool, Optional[str]]:
        """
        Liquidity sweep: the last candle wick temporarily breaches a swing
        level but the candle CLOSES back inside — a stop hunt / liquidity grab.
        """
        if len(df) < 2:
            return False, None

        last = df.iloc[-1]
        highs = swings["highs"]
        lows = swings["lows"]

        # High sweep: wick above swing high but closed below it
        if highs:
            swing_h = highs[-1]
            if last["high"] > swing_h and last["close"] < swing_h:
                logger.debug(f"Liquidity sweep HIGH: wick={last['high']:.5f} > {swing_h:.5f}, closed below")
                return True, "high"

        # Low sweep: wick below swing low but closed above it
        if lows:
            swing_l = lows[-1]
            if last["low"] < swing_l and last["close"] > swing_l:
                logger.debug(f"Liquidity sweep LOW: wick={last['low']:.5f} < {swing_l:.5f}, closed above")
                return True, "low"

        return False, None

    # ------------------------------------------------------------------
    # Structure quality score
    # ------------------------------------------------------------------

    def _score_structure(self, result: Dict, df: pd.DataFrame) -> float:
        """
        Returns a float [0, 1] rating structure clarity.
        Higher = clearer, more tradeable structure.
        """
        score = 0.0

        if result["trend"] != "ranging":
            score += 0.4

        if result["bos"]:
            score += 0.25

        if result["choch"]:
            score += 0.2

        if result["liquidity_sweep"]:
            score += 0.15

        return min(score, 1.0)

    def analyse_with_mtf(self, df: pd.DataFrame, mtf_result=None) -> Dict:
        """
        v5: Combine single-TF structure with MTF confluence result.
        Returns enriched structure dict with mtf_* keys added.
        The mtf_bias key is used by is_trade_aligned_with_structure() to
        allow ranging-market trades when higher-timeframe context aligns.
        """
        result = self.analyse(df)
        if mtf_result is None:
            return result
        try:
            result["mtf_bias"]            = getattr(mtf_result, "bias", "neutral")
            result["mtf_score"]           = getattr(mtf_result, "score", 0.0)
            result["mtf_htf_aligned"]     = getattr(mtf_result, "htf_aligned", False)
            result["mtf_ltf_confirmed"]   = getattr(mtf_result, "ltf_confirmed", False)
            result["mtf_liquidity_swept"] = getattr(mtf_result, "liquidity_swept", False)
            result["mtf_killzone"]        = getattr(mtf_result, "killzone_active", False)
            result["mtf_pd_ok"]           = getattr(mtf_result, "premium_discount_ok", False)
            result["mtf_reasons"]         = getattr(mtf_result, "reasons", [])
            # Boost structure_score with MTF evidence
            boost = (0.15 if result["mtf_htf_aligned"] else 0) + \
                    (0.10 if result["mtf_ltf_confirmed"] else 0) + \
                    (0.05 if result["mtf_liquidity_swept"] else 0)
            result["structure_score"] = min(1.0, result.get("structure_score", 0.0) + boost)
        except Exception as e:
            logger.debug(f"analyse_with_mtf enrichment error: {e}")
        return result
