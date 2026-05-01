"""
ranging_scalper.py — Intelligent Range-Market Scalping Engine (AI EA v17)
=========================================================================
Activates when H4/H1 are detected as RANGING by RegimeDetector.
Drills into M15 → M5 → M1 for precision scalp entries at range extremes.

Pro-master trading logic implemented:
  1. HTF Range boundary identification (H4/H1 swing highs/lows)
  2. Range quality scoring (tight vs. sloppy, age, volume profile)
  3. LTF drill-down waterfall: H1 ranging → M15 structure → M5 entry
  4. Liquidity sweep confirmation at range extremes (stop-hunt entry)
  5. ICT concepts: OB mitigation, FVG fill, displacement candles
  6. Dynamic micro SL/TP sized to LTF ATR (not HTF ATR)
  7. Quick-exit rules: momentum fade, mid-range stall, opposing sweep
  8. Scalp session filter: only during London/NY overlaps + Asian range plays
  9. Mean-reversion probability scoring (not just direction — also timing)
 10. Compounding scalps: partial close at mid-range, runner to far side

Author: AI EA v17 Upgrade
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime, time as dtime, timezone
from enum import Enum, auto

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Data classes
# ─────────────────────────────────────────────────────────────────────────────

class ScalpZone(Enum):
    RANGE_HIGH   = "range_high"    # Shorting from top of range
    RANGE_LOW    = "range_low"     # Buying from bottom of range
    MID_RANGE    = "mid_range"     # No trade — avoid the middle
    OUTSIDE      = "outside"       # Price outside range — possible breakout


class ScalpSignal(Enum):
    BUY          = "BUY"
    SELL         = "SELL"
    NO_TRADE     = "NO_TRADE"


@dataclass
class RangeContext:
    """HTF range boundaries extracted from H4/H1."""
    range_high: float   = 0.0
    range_low: float    = 0.0
    range_mid: float    = 0.0
    range_atr: float    = 0.0          # ATR of the H1 timeframe
    range_width_atr: float = 0.0       # range width / ATR ratio
    range_age_bars: int = 0            # how many bars range has held
    range_quality: float = 0.0         # 0-1 quality score
    touches_high: int   = 0            # # of times top was tested
    touches_low: int    = 0            # # of times bottom was tested
    is_valid: bool      = False
    notes: List[str]    = field(default_factory=list)


@dataclass
class LTFEntry:
    """Precision entry from LTF analysis."""
    signal: ScalpSignal         = ScalpSignal.NO_TRADE
    entry_price: float          = 0.0
    sl_price: float             = 0.0
    tp1_price: float            = 0.0    # Partial close (mid-range)
    tp2_price: float            = 0.0    # Runner (far side of range)
    confidence: float           = 0.0    # 0-1
    entry_tf: str               = ""     # m15 / m5 / m1
    zone: ScalpZone             = ScalpZone.MID_RANGE
    ltf_atr: float              = 0.0
    liquidity_swept: bool       = False
    ob_mitigated: bool          = False
    fvg_present: bool           = False
    displacement: bool          = False  # strong impulse candle confirming
    reasons: List[str]          = field(default_factory=list)
    timestamp: str              = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ─────────────────────────────────────────────────────────────────────────────
# RangeScalper
# ─────────────────────────────────────────────────────────────────────────────

class RangingScalper:
    """
    Intelligent range-market scalping engine.

    Usage:
        scalper = RangingScalper(broker)
        entry = scalper.analyse(symbol, df_h1, df_h4)
        if entry.signal != ScalpSignal.NO_TRADE:
            # use entry.entry_price, entry.sl_price, entry.tp1_price ...
    """

    def __init__(
        self,
        broker,
        # Range identification
        range_lookback_h1: int     = 40,    # bars of H1 to define range
        range_proximity_pct: float = 0.15,  # % of range width = "near extreme"
        min_range_width_atr: float = 1.5,   # minimum range width in ATR units
        max_range_width_atr: float = 8.0,   # too wide = sloppy, skip
        min_range_age_bars: int    = 6,     # range must be at least 6 H1 bars old
        min_touches: int           = 2,     # both extremes must have ≥N touches
        # LTF drill-down
        ltf_sequence: List[str]    = None,  # drill order
        use_m5_entry: bool         = True,
        use_m1_entry: bool         = False, # only for very tight setups
        # Risk
        scalp_sl_atr_mult: float   = 0.6,  # LTF ATR-based SL — tight
        scalp_tp1_atr_mult: float  = 1.2,  # first partial
        scalp_tp2_atr_mult: float  = 2.5,  # runner to far side
        scalp_risk_fraction: float = 0.004, # 0.4% per scalp (smaller than trend)
        min_rr: float              = 1.8,  # minimum R:R before taking trade
        # Confidence
        min_confidence: float      = 0.60,
    ):
        self.broker                = broker
        self.range_lookback_h1     = range_lookback_h1
        self.range_proximity_pct   = range_proximity_pct
        self.min_range_width_atr   = min_range_width_atr
        self.max_range_width_atr   = max_range_width_atr
        self.min_range_age_bars    = min_range_age_bars
        self.min_touches           = min_touches
        self.ltf_sequence          = ltf_sequence or ["m15", "m5"]
        self.use_m5_entry          = use_m5_entry
        self.use_m1_entry          = use_m1_entry
        self.scalp_sl_atr_mult     = scalp_sl_atr_mult
        self.scalp_tp1_atr_mult    = scalp_tp1_atr_mult
        self.scalp_tp2_atr_mult    = scalp_tp2_atr_mult
        self.scalp_risk_fraction   = scalp_risk_fraction
        self.min_rr                = min_rr
        self.min_confidence        = min_confidence

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def analyse(
        self,
        symbol: str,
        df_h1: pd.DataFrame,
        df_h4: Optional[pd.DataFrame] = None,
    ) -> LTFEntry:
        """
        Full ranging-market scalp analysis.
        Returns LTFEntry — check .signal != NO_TRADE before trading.
        """
        no_trade = LTFEntry(signal=ScalpSignal.NO_TRADE)
        try:
            # 1. Build H1/H4 range context
            rctx = self._build_range_context(df_h1, df_h4)
            if not rctx.is_valid:
                logger.debug(f"[RangeScalper] {symbol}: range not valid ({rctx.notes})")
                return no_trade

            # 2. Session filter — pro traders know WHEN to scalp ranges
            if not self._session_ok():
                logger.debug(f"[RangeScalper] {symbol}: outside scalp sessions")
                return no_trade

            # 3. Where is price in the range right now?
            current_price = float(df_h1["close"].iloc[-1])
            zone = self._classify_zone(current_price, rctx)
            if zone == ScalpZone.MID_RANGE:
                logger.debug(f"[RangeScalper] {symbol}: price at mid-range — no scalp")
                return no_trade
            if zone == ScalpZone.OUTSIDE:
                logger.debug(f"[RangeScalper] {symbol}: price outside range — possible breakout, skip")
                return no_trade

            # 4. Drill into LTFs for precision entry
            entry = self._drill_ltf(symbol, zone, rctx)
            if entry.signal == ScalpSignal.NO_TRADE:
                return no_trade

            # 5. Final R:R gate
            risk  = abs(entry.entry_price - entry.sl_price)
            rew1  = abs(entry.tp1_price   - entry.entry_price)
            if risk <= 0 or (rew1 / risk) < self.min_rr:
                logger.debug(f"[RangeScalper] {symbol}: R:R {rew1/risk:.2f} < {self.min_rr} — skip")
                return no_trade

            logger.info(
                f"[RangeScalper] {symbol} {entry.signal.value} | zone={zone.value} | "
                f"conf={entry.confidence:.2f} | entry={entry.entry_price:.5f} "
                f"SL={entry.sl_price:.5f} TP1={entry.tp1_price:.5f} TP2={entry.tp2_price:.5f} | "
                f"R:R1={rew1/risk:.1f} | {entry.reasons}"
            )
            return entry

        except Exception as exc:
            logger.error(f"[RangeScalper] {symbol} analyse() error: {exc}", exc_info=True)
            return no_trade

    # ─────────────────────────────────────────────────────────────────
    # Range context construction
    # ─────────────────────────────────────────────────────────────────

    def _build_range_context(
        self, df_h1: pd.DataFrame, df_h4: Optional[pd.DataFrame]
    ) -> RangeContext:
        ctx = RangeContext()
        try:
            n = min(self.range_lookback_h1, len(df_h1))
            if n < 20:
                ctx.notes.append("too_few_h1_bars")
                return ctx

            recent = df_h1.iloc[-n:]
            atr    = self._calc_atr(df_h1, 14)

            # Primary range from H1 swing highs/lows
            swing_h, swing_l = self._swing_levels(recent)

            # If H4 is available, use it to validate / widen the range definition
            if df_h4 is not None and len(df_h4) >= 20:
                h4_recent  = df_h4.iloc[-20:]
                h4_swing_h, h4_swing_l = self._swing_levels(h4_recent)
                # Take the tighter of H1 and H4 to define clean institutional range
                swing_h = min(swing_h, h4_swing_h) if h4_swing_h > 0 else swing_h
                swing_l = max(swing_l, h4_swing_l) if h4_swing_l > 0 else swing_l
                ctx.notes.append("h4_validated")

            if swing_h <= swing_l or atr <= 0:
                ctx.notes.append("invalid_swing_levels")
                return ctx

            rng_width   = swing_h - swing_l
            width_in_atr = rng_width / atr

            if width_in_atr < self.min_range_width_atr:
                ctx.notes.append(f"range_too_narrow_{width_in_atr:.2f}atr")
                return ctx
            if width_in_atr > self.max_range_width_atr:
                ctx.notes.append(f"range_too_wide_{width_in_atr:.2f}atr")
                return ctx

            touches_h = self._count_touches(recent, swing_h, atr, side="high")
            touches_l = self._count_touches(recent, swing_l, atr, side="low")

            # FIX: Require at least 1 touch each side, and at least one side with 2+.
            # Old requirement (both sides >= 2) was too strict — the log shows H=2,L=1
            # or H=1,L=1 constantly → always NO_TRADE. A single clean test of a level
            # is a valid range boundary; quality score will penalize weaker setups.
            min_touch_each = 1
            if touches_h < min_touch_each or touches_l < min_touch_each:
                ctx.notes.append(f"insufficient_touches_H={touches_h}_L={touches_l}")
                return ctx
            if max(touches_h, touches_l) < self.min_touches:
                # Neither side has been touched twice — range not confirmed enough
                ctx.notes.append(f"range_not_confirmed_H={touches_h}_L={touches_l}")
                return ctx

            # Range age (bars since it first formed)
            range_age = self._estimate_range_age(recent, swing_h, swing_l)
            if range_age < self.min_range_age_bars:
                ctx.notes.append(f"range_too_young_{range_age}bars")
                return ctx

            # Quality score: 0-1
            quality = self._score_range_quality(
                touches_h, touches_l, width_in_atr, range_age
            )

            ctx.range_high        = swing_h
            ctx.range_low         = swing_l
            ctx.range_mid         = (swing_h + swing_l) / 2
            ctx.range_atr         = atr
            ctx.range_width_atr   = width_in_atr
            ctx.range_age_bars    = range_age
            ctx.range_quality     = quality
            ctx.touches_high      = touches_h
            ctx.touches_low       = touches_l
            ctx.is_valid          = quality >= 0.30
            ctx.notes.append(f"quality={quality:.2f}")

        except Exception as exc:
            logger.error(f"_build_range_context error: {exc}", exc_info=True)

        return ctx

    # ─────────────────────────────────────────────────────────────────
    # LTF drill-down: M15 → M5 precision entry
    # ─────────────────────────────────────────────────────────────────

    def _drill_ltf(
        self, symbol: str, zone: ScalpZone, rctx: RangeContext
    ) -> LTFEntry:
        """
        Walk down timeframes. Try M15 first; if no clean setup found,
        go to M5 for tighter entry. Returns best LTFEntry found.
        """
        best = LTFEntry(signal=ScalpSignal.NO_TRADE)

        for tf in self.ltf_sequence:
            try:
                bars = 200 if tf == "m15" else 300
                df_ltf = self.broker.get_market_data(symbol, tf, bars)
                if df_ltf is None or len(df_ltf) < 30:
                    continue

                entry = self._analyse_ltf(df_ltf, tf, zone, rctx)
                if entry.signal == ScalpSignal.NO_TRADE:
                    continue

                # Higher confidence or first valid entry wins
                if entry.confidence > best.confidence:
                    best = entry
                    best.entry_tf = tf

                # Stop drilling if we already have high confidence
                if best.confidence >= 0.78:
                    break

            except Exception as exc:
                logger.warning(f"[RangeScalper] drill_ltf {tf} error: {exc}")
                continue

        return best

    def _analyse_ltf(
        self,
        df: pd.DataFrame,
        tf: str,
        zone: ScalpZone,
        rctx: RangeContext,
    ) -> LTFEntry:
        """
        Analyse a single LTF dataframe for a scalp setup at the range extreme.
        Implements institutional concepts:
          - Stop hunt / liquidity sweep
          - Order block (OB) mitigation
          - Fair value gap (FVG) fill
          - Displacement candle confirmation
          - M15 BOS in favour of the fade
        """
        entry = LTFEntry(signal=ScalpSignal.NO_TRADE, zone=zone)
        try:
            ltf_atr = self._calc_atr(df, 14)
            if ltf_atr <= 0:
                return entry

            current    = float(df["close"].iloc[-1])
            is_buy     = (zone == ScalpZone.RANGE_LOW)
            direction  = ScalpSignal.BUY if is_buy else ScalpSignal.SELL
            extreme    = rctx.range_low if is_buy else rctx.range_high
            confidence = rctx.range_quality  # start from range quality

            reasons: List[str] = []

            # ── 1. Liquidity sweep (stop-hunt beyond extreme) ─────────
            swept = self._detect_liquidity_sweep(df, extreme, is_buy, ltf_atr)
            if swept:
                confidence += 0.18
                entry.liquidity_swept = True
                reasons.append("liq_sweep")

            # ── 2. Order Block mitigation ─────────────────────────────
            ob_level = self._find_order_block(df, is_buy, extreme, ltf_atr)
            ob_mitigated = False
            if ob_level and abs(current - ob_level) < ltf_atr * 0.8:
                confidence += 0.15
                ob_mitigated = True
                entry.ob_mitigated = True
                reasons.append(f"ob_mit@{ob_level:.5f}")

            # ── 3. Fair Value Gap ─────────────────────────────────────
            fvg_low, fvg_high = self._detect_fvg(df, is_buy)
            fvg_present = False
            if fvg_low > 0 and fvg_high > 0:
                mid_fvg = (fvg_low + fvg_high) / 2
                if abs(current - mid_fvg) < ltf_atr * 0.5:
                    confidence += 0.10
                    fvg_present = True
                    entry.fvg_present = True
                    reasons.append("fvg_fill")

            # ── 4. Displacement candle (strong rejection) ─────────────
            displacement = self._detect_displacement(df, is_buy)
            if displacement:
                confidence += 0.12
                entry.displacement = True
                reasons.append("displacement")

            # ── 5. LTF BOS in favour of the fade ─────────────────────
            ltf_bos = self._detect_ltf_bos(df, is_buy)
            if ltf_bos:
                confidence += 0.10
                reasons.append("ltf_bos")

            # ── 6. RSI divergence / exhaustion ───────────────────────
            rsi_exhaust = self._detect_rsi_exhaustion(df, is_buy)
            if rsi_exhaust:
                confidence += 0.08
                reasons.append("rsi_exhaust")

            # ── 7. Engulfing / pin bar reversal candle ────────────────
            reversal_candle = self._detect_reversal_candle(df, is_buy)
            if reversal_candle:
                confidence += 0.07
                reasons.append("reversal_candle")

            # Require at least 2 confirmations beyond base quality
            n_extras = sum([swept, ob_mitigated, fvg_present, displacement, ltf_bos, rsi_exhaust, reversal_candle])
            if n_extras < 2:
                reasons.append(f"only_{n_extras}_confirms")
                return entry  # NO_TRADE

            confidence = min(confidence, 0.97)

            if confidence < self.min_confidence:
                reasons.append(f"conf_too_low_{confidence:.2f}")
                return entry  # NO_TRADE

            # ── Construct precise SL / TP ─────────────────────────────
            sl_buffer   = ltf_atr * self.scalp_sl_atr_mult
            tp1_dist    = ltf_atr * self.scalp_tp1_atr_mult
            tp2_dist    = ltf_atr * self.scalp_tp2_atr_mult

            if is_buy:
                sl_price  = current - sl_buffer
                # If we detected a sweep, SL goes below the sweep low
                if swept:
                    sweep_low = float(df["low"].iloc[-3:].min())
                    sl_price  = min(sl_price, sweep_low - ltf_atr * 0.2)
                tp1_price = min(current + tp1_dist, rctx.range_mid)   # partial at mid
                tp2_price = min(current + tp2_dist, rctx.range_high - ltf_atr * 0.3)  # runner
            else:
                sl_price  = current + sl_buffer
                if swept:
                    sweep_high = float(df["high"].iloc[-3:].max())
                    sl_price   = max(sl_price, sweep_high + ltf_atr * 0.2)
                tp1_price = max(current - tp1_dist, rctx.range_mid)
                tp2_price = max(current - tp2_dist, rctx.range_low + ltf_atr * 0.3)

            entry.signal      = direction
            entry.entry_price = current
            entry.sl_price    = round(sl_price, 5)
            entry.tp1_price   = round(tp1_price, 5)
            entry.tp2_price   = round(tp2_price, 5)
            entry.confidence  = round(confidence, 4)
            entry.ltf_atr     = ltf_atr
            entry.reasons     = reasons

        except Exception as exc:
            logger.warning(f"_analyse_ltf error: {exc}", exc_info=True)
            return LTFEntry(signal=ScalpSignal.NO_TRADE)

        return entry

    # ─────────────────────────────────────────────────────────────────
    # ICT / Smart-money detectors
    # ─────────────────────────────────────────────────────────────────

    def _detect_liquidity_sweep(
        self, df: pd.DataFrame, extreme: float,
        is_buy: bool, atr: float, lookback: int = 5
    ) -> bool:
        """
        Detect if price spiked BEYOND the range extreme (sweeping stops)
        and then closed BACK inside on the same or next candle.
        This is the single highest-probability setup in institutional trading.
        """
        try:
            recent = df.iloc[-lookback:]
            for i in range(len(recent) - 1):
                row = recent.iloc[i]
                close_row = recent.iloc[i + 1] if i + 1 < len(recent) else recent.iloc[i]
                if is_buy:
                    # Swept low: wick below extreme, but closed above
                    if float(row["low"]) < extreme - atr * 0.1 and float(close_row["close"]) > extreme - atr * 0.3:
                        return True
                else:
                    # Swept high: wick above extreme, but closed below
                    if float(row["high"]) > extreme + atr * 0.1 and float(close_row["close"]) < extreme + atr * 0.3:
                        return True
        except Exception:
            pass
        return False

    def _find_order_block(
        self, df: pd.DataFrame, is_buy: bool,
        extreme: float, atr: float, lookback: int = 30
    ) -> Optional[float]:
        """
        Find the last bearish candle before a bullish impulse (bull OB)
        or last bullish candle before a bearish impulse (bear OB)
        near the range extreme.
        """
        try:
            search = df.iloc[-lookback:]
            for i in range(len(search) - 3, 0, -1):
                row   = search.iloc[i]
                close = float(row["close"])
                open_ = float(row["open"])
                high  = float(row["high"])
                low   = float(row["low"])

                if is_buy:
                    # Last bearish candle near range low
                    if close < open_ and abs(low - extreme) < atr * 1.5:
                        # Confirm: next 2 bars moved up strongly
                        next_close = float(search.iloc[min(i + 2, len(search) - 1)]["close"])
                        if next_close > high:
                            return (high + low) / 2
                else:
                    # Last bullish candle near range high
                    if close > open_ and abs(high - extreme) < atr * 1.5:
                        next_close = float(search.iloc[min(i + 2, len(search) - 1)]["close"])
                        if next_close < low:
                            return (high + low) / 2
        except Exception:
            pass
        return None

    def _detect_fvg(
        self, df: pd.DataFrame, is_buy: bool, lookback: int = 20
    ) -> Tuple[float, float]:
        """
        Detect a Fair Value Gap (3-candle imbalance).
        Returns (fvg_low, fvg_high) or (0, 0) if none found.
        """
        try:
            recent = df.iloc[-lookback:]
            for i in range(len(recent) - 2, 1, -1):
                c1 = recent.iloc[i - 2]
                # c2 = recent.iloc[i - 1]  # the impulse candle
                c3 = recent.iloc[i]
                if is_buy:
                    # Bullish FVG: gap between c1 high and c3 low
                    gap_lo = float(c1["high"])
                    gap_hi = float(c3["low"])
                    if gap_hi > gap_lo:
                        return gap_lo, gap_hi
                else:
                    # Bearish FVG: gap between c3 high and c1 low
                    gap_lo = float(c3["high"])
                    gap_hi = float(c1["low"])
                    if gap_hi > gap_lo:
                        return gap_lo, gap_hi
        except Exception:
            pass
        return 0.0, 0.0

    def _detect_displacement(
        self, df: pd.DataFrame, is_buy: bool, lookback: int = 5
    ) -> bool:
        """
        Displacement: a strong impulse candle (body > 1.5 * recent avg body)
        in the direction of the intended trade, showing institutional intent.
        """
        try:
            recent   = df.iloc[-lookback:]
            bodies   = (recent["close"] - recent["open"]).abs()
            avg_body = bodies.mean()
            last_body = float(bodies.iloc[-1])
            last_candle = recent.iloc[-1]
            is_bullish  = float(last_candle["close"]) > float(last_candle["open"])
            if is_buy and is_bullish and last_body > avg_body * 1.5:
                return True
            if not is_buy and not is_bullish and last_body > avg_body * 1.5:
                return True
        except Exception:
            pass
        return False

    def _detect_ltf_bos(
        self, df: pd.DataFrame, is_buy: bool, swing_n: int = 5
    ) -> bool:
        """
        LTF Break of Structure: most recent price broke above/below the last
        LTF swing high/low (micro-BOS confirming reversal from range extreme).
        """
        try:
            n     = min(len(df), 30)
            sub   = df.iloc[-n:]
            current = float(sub["close"].iloc[-1])
            if is_buy:
                # BOS = current price above last LTF swing high
                recent_swings = sub["high"].rolling(swing_n).max().iloc[-10:]
                prev_swing_h  = float(recent_swings.iloc[-3]) if len(recent_swings) >= 3 else 0.0
                return current > prev_swing_h > 0
            else:
                recent_swings = sub["low"].rolling(swing_n).min().iloc[-10:]
                prev_swing_l  = float(recent_swings.iloc[-3]) if len(recent_swings) >= 3 else 999999.0
                return current < prev_swing_l < 999999.0
        except Exception:
            pass
        return False

    def _detect_rsi_exhaustion(
        self, df: pd.DataFrame, is_buy: bool
    ) -> bool:
        """
        RSI showing exhaustion at range extreme:
          - BUY  at bottom: RSI < 30 (oversold) and ticking up
          - SELL at top   : RSI > 70 (overbought) and ticking down
        """
        try:
            rsi = self._calc_rsi(df["close"], 14)
            if len(rsi) < 3:
                return False
            curr_rsi = float(rsi.iloc[-1])
            prev_rsi = float(rsi.iloc[-2])
            if is_buy and curr_rsi < 32 and curr_rsi > prev_rsi:
                return True
            if not is_buy and curr_rsi > 68 and curr_rsi < prev_rsi:
                return True
        except Exception:
            pass
        return False

    def _detect_reversal_candle(
        self, df: pd.DataFrame, is_buy: bool
    ) -> bool:
        """
        Bullish/bearish reversal candle patterns:
          - Hammer / bullish engulfing for buys
          - Shooting star / bearish engulfing for sells
        """
        try:
            if len(df) < 2:
                return False
            cur  = df.iloc[-1]
            prev = df.iloc[-2]
            o, h, l, c = float(cur["open"]), float(cur["high"]), float(cur["low"]), float(cur["close"])
            body   = abs(c - o)
            hl     = h - l
            if hl == 0:
                return False
            upper_wick = (h - max(c, o)) / hl
            lower_wick = (min(c, o) - l) / hl
            body_pct   = body / hl

            if is_buy:
                # Hammer: small body at top, long lower wick
                hammer = lower_wick > 0.55 and body_pct < 0.35 and c > o
                # Bullish engulfing
                prev_bearish = float(prev["close"]) < float(prev["open"])
                engulf = (prev_bearish and c > float(prev["open"]) and o < float(prev["close"]))
                return hammer or engulf
            else:
                # Shooting star
                star = upper_wick > 0.55 and body_pct < 0.35 and c < o
                # Bearish engulfing
                prev_bullish = float(prev["close"]) > float(prev["open"])
                engulf = (prev_bullish and c < float(prev["open"]) and o > float(prev["close"]))
                return star or engulf
        except Exception:
            pass
        return False

    # ─────────────────────────────────────────────────────────────────
    # Range helpers
    # ─────────────────────────────────────────────────────────────────

    def _classify_zone(self, price: float, rctx: RangeContext) -> ScalpZone:
        """Classify where price sits within the range."""
        if price > rctx.range_high * 1.002:
            return ScalpZone.OUTSIDE
        if price < rctx.range_low * 0.998:
            return ScalpZone.OUTSIDE
        width  = rctx.range_high - rctx.range_low
        buffer = width * self.range_proximity_pct
        if price >= rctx.range_high - buffer:
            return ScalpZone.RANGE_HIGH
        if price <= rctx.range_low + buffer:
            return ScalpZone.RANGE_LOW
        return ScalpZone.MID_RANGE

    def _swing_levels(self, df: pd.DataFrame, n: int = 5) -> Tuple[float, float]:
        """
        Find significant swing high / low using a rolling window.
        Returns (swing_high, swing_low).
        """
        try:
            swing_high = float(df["high"].rolling(n, center=True).max().dropna().max())
            swing_low  = float(df["low"].rolling(n, center=True).min().dropna().min())
            return swing_high, swing_low
        except Exception:
            return 0.0, 0.0

    def _count_touches(
        self, df: pd.DataFrame, level: float,
        atr: float, side: str, tolerance_mult: float = 0.25
    ) -> int:
        """Count how many bars came within tolerance of the level."""
        tolerance = atr * tolerance_mult
        if side == "high":
            return int((df["high"] >= level - tolerance).sum())
        else:
            return int((df["low"]  <= level + tolerance).sum())

    def _estimate_range_age(
        self, df: pd.DataFrame, range_h: float, range_l: float
    ) -> int:
        """
        Estimate how many bars the current range has been intact
        (no close above range_h or below range_l).
        """
        try:
            bars = 0
            for i in range(len(df) - 1, -1, -1):
                c = float(df["close"].iloc[i])
                if c > range_h or c < range_l:
                    break
                bars += 1
            return bars
        except Exception:
            return 0

    def _score_range_quality(
        self,
        touches_h: int,
        touches_l: int,
        width_atr: float,
        age_bars: int,
    ) -> float:
        """Score range 0–1. More touches, right width, older = better."""
        score = 0.0
        # Touch score (max 0.4)
        touch_score = min((touches_h + touches_l) / 10.0, 0.4)
        score += touch_score
        # Width score: sweet spot is 2-5 ATR wide (max 0.3)
        if 2.0 <= width_atr <= 5.0:
            score += 0.30
        elif 1.5 <= width_atr < 2.0 or 5.0 < width_atr <= 6.5:
            score += 0.15
        # Age score (max 0.3): older but not stale
        if 10 <= age_bars <= 50:
            score += 0.30
        elif 6 <= age_bars < 10:
            score += 0.15
        elif age_bars > 50:
            score += 0.10   # very old ranges can break
        return min(score, 1.0)

    # ─────────────────────────────────────────────────────────────────
    # Session filter
    # ─────────────────────────────────────────────────────────────────

    def _session_ok(self) -> bool:
        """
        Best scalp sessions:
          - Asian session (00:00–03:00 UTC): clean ranging, low spread
          - London open range play (07:00–09:30 UTC)
          - Pre-overlap (09:30–12:00 UTC): FIX — was missing this valid window
          - London/NY overlap (12:00–17:00 UTC): highest volume fades
          - NY afternoon range (18:00–21:00 UTC)
        Avoid: 22:00–00:00 UTC (rollover), news times (handled by NewsFilter).
        """
        now_h = datetime.now(timezone.utc).hour
        now_m = datetime.now(timezone.utc).minute
        time_frac = now_h + now_m / 60.0

        # Asian range play
        if 0.0 <= time_frac < 3.0:
            return True
        # London open range fade + pre-overlap
        if 7.0 <= time_frac < 12.0:
            return True
        # London/NY overlap (prime)
        if 12.0 <= time_frac < 17.0:
            return True
        # NY afternoon range
        if 18.0 <= time_frac < 21.0:
            return True

        return False

    # ─────────────────────────────────────────────────────────────────
    # Indicator calculators
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
        try:
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"]  - df["close"].shift()).abs(),
            ], axis=1).max(axis=1)
            val = tr.rolling(period).mean().iloc[-1]
            return float(val) if not np.isnan(val) else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        try:
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(period).mean()
            loss  = (-delta.clip(upper=0)).rolling(period).mean()
            rs    = gain / loss.replace(0, np.nan)
            return 100 - (100 / (1 + rs))
        except Exception:
            return pd.Series(50.0, index=close.index)


# ─────────────────────────────────────────────────────────────────────────────
# Quick-exit manager for open scalp positions
# ─────────────────────────────────────────────────────────────────────────────

class ScalpExitManager:
    """
    Monitors open scalp positions and triggers quick exits on:
      - Price stalling at mid-range (no follow-through)
      - Opposing liquidity sweep while in trade
      - Time-based exit (max hold time for scalps)
      - Momentum fade (LTF momentum reversal)
    """

    def __init__(
        self,
        max_hold_bars_m15: int = 16,    # ~4 hours on M15
        mid_range_exit: bool   = True,
        momentum_fade_exit: bool = True,
    ):
        self.max_hold_bars_m15  = max_hold_bars_m15
        self.mid_range_exit     = mid_range_exit
        self.momentum_fade_exit = momentum_fade_exit

    def should_exit_early(
        self,
        pos_type: str,          # "buy" | "sell"
        entry_price: float,
        current_price: float,
        range_ctx: RangeContext,
        df_ltf: pd.DataFrame,
        bars_held: int,
    ) -> Tuple[bool, str]:
        """
        Returns (should_exit, reason).
        """
        direction = 1 if pos_type == "buy" else -1
        profit_pts = (current_price - entry_price) * direction
        ltf_atr = self._calc_atr(df_ltf)

        # 1. Time-based exit
        if bars_held >= self.max_hold_bars_m15:
            return True, "max_hold_time_exceeded"

        # 2. Price reached mid-range — take partial profits via TP1 (handled externally)
        # But if price stalls at mid-range and starts reversing, exit runner too
        if self.mid_range_exit and profit_pts > 0:
            mid = range_ctx.range_mid
            near_mid = abs(current_price - mid) < ltf_atr * 0.4
            reversing = self._is_reversing(df_ltf, pos_type)
            if near_mid and reversing:
                return True, "mid_range_stall_reverse"

        # 3. Opposing sweep — institutional reversal against us
        opposing_swept = self._detect_opposing_sweep(df_ltf, pos_type, ltf_atr)
        if opposing_swept and profit_pts < ltf_atr * 0.3:
            return True, "opposing_liquidity_sweep"

        # 4. Momentum fade — LTF momentum collapsing
        if self.momentum_fade_exit and profit_pts > ltf_atr * 0.5:
            if self._momentum_fading(df_ltf, pos_type):
                return True, "ltf_momentum_fade"

        return False, ""

    def _is_reversing(self, df: pd.DataFrame, pos_type: str) -> bool:
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            if pos_type == "buy":
                return float(last["close"]) < float(prev["close"]) and float(last["close"]) < float(last["open"])
            else:
                return float(last["close"]) > float(prev["close"]) and float(last["close"]) > float(last["open"])
        except Exception:
            return False

    def _detect_opposing_sweep(self, df: pd.DataFrame, pos_type: str, atr: float) -> bool:
        try:
            last = df.iloc[-1]
            prev_high = float(df["high"].iloc[-5:-1].max())
            prev_low  = float(df["low"].iloc[-5:-1].min())
            if pos_type == "buy" and float(last["high"]) > prev_high + atr * 0.5:
                return True
            if pos_type == "sell" and float(last["low"]) < prev_low - atr * 0.5:
                return True
        except Exception:
            pass
        return False

    def _momentum_fading(self, df: pd.DataFrame, pos_type: str, period: int = 5) -> bool:
        try:
            mom  = df["close"].diff(period)
            curr = float(mom.iloc[-1])
            prev = float(mom.iloc[-2])
            if pos_type == "buy":
                return curr < prev * 0.4 and curr > 0
            else:
                return curr > prev * 0.4 and curr < 0
        except Exception:
            return False

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
        try:
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"]  - df["close"].shift()).abs(),
            ], axis=1).max(axis=1)
            val = tr.rolling(period).mean().iloc[-1]
            return float(val) if not np.isnan(val) else 0.0
        except Exception:
            return 0.0
