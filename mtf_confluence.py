"""
mtf_confluence.py — Multi-Timeframe Confluence Engine (AI EA v13)
================================================================
7-Tier MTF architecture for maximum edge and precision.

Architecture:
  D1  → Macro trend / weekly bias, dominant order blocks
  H4  → Swing context, key structure levels, major order blocks
  H3  → Intermediate structure, BOS/CHoCH confirmation layer
  H1  → Entry-level confirmation, session structure, FVGs
  M30 → Sub-session context, mid-range structure
  M15 → Precision entry, liquidity sweeps, mitigation timing
  M10 → Ultra-precision entry trigger, micro-structure, final confirmation

Usage:
    engine = MTFConfluenceEngine(fetcher)
    ctx = engine.get_confluence(symbol)
    # ctx.score [0,1], ctx.bias ('bullish'|'bearish'|'neutral'), ctx.entry_zone
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class TimeframeContext:
    tf: str
    trend: str          = "ranging"    # bullish | bearish | ranging
    bos: bool           = False
    choch: bool         = False
    choch_dir: str      = ""
    ob_level: float     = 0.0          # nearest order block price
    ob_type: str        = ""           # bullish | bearish
    fvg_low: float      = 0.0
    fvg_high: float     = 0.0
    fvg_filled: bool    = True
    swing_high: float   = 0.0
    swing_low: float    = 0.0
    atr: float          = 0.0
    structure_score: float = 0.0
    premium_discount: str = "neutral"  # premium | discount | neutral


@dataclass
class ConfluenceResult:
    """Full 7-tier multi-timeframe confluence decision packet."""
    symbol: str
    bias: str           = "neutral"    # bullish | bearish | neutral
    score: float        = 0.0         # 0–1, quality of the confluence
    d1:   Optional[TimeframeContext] = None
    h4:   Optional[TimeframeContext] = None
    h3:   Optional[TimeframeContext] = None
    h1:   Optional[TimeframeContext] = None
    m30:  Optional[TimeframeContext] = None
    m15:  Optional[TimeframeContext] = None
    m10:  Optional[TimeframeContext] = None

    # Entry zone
    entry_low: float    = 0.0
    entry_high: float   = 0.0
    nearest_ob: float   = 0.0
    nearest_fvg_low: float  = 0.0
    nearest_fvg_high: float = 0.0

    # Risk levels
    invalidation: float = 0.0         # price that invalidates thesis
    target_1r: float    = 0.0         # 1R target
    target_2r: float    = 0.0         # 2R target
    target_3r: float    = 0.0         # 3R target (extended run)

    # Confluence flags
    macro_aligned: bool  = False       # D1 macro bias set
    htf_aligned: bool    = False       # D1+H4+H3 all agree
    mtf_aligned: bool    = False       # H1+M30 mid-tier confirmation
    ltf_confirmed: bool  = False       # M15+M10 precision entry confirmed
    killzone_active: bool = False
    liquidity_swept: bool = False
    premium_discount_ok: bool = False  # Buying in discount / selling in premium
    tier_score: int     = 0            # 0-7 count of confirming tiers

    reasons: List[str]  = field(default_factory=list)
    timestamp: str      = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MTFConfluenceEngine:
    """
    Fetches D1, H4, H3, H1, M30, M15, M10 data and builds a full 7-tier
    confluence picture for maximum institutional edge.
    """

    TIMEFRAMES = {
        "d1":  {"bars": 365, "tf_str": "d1"},
        "h4":  {"bars": 500, "tf_str": "h4"},
        "h3":  {"bars": 400, "tf_str": "h3"},
        "h1":  {"bars": 500, "tf_str": "h1"},
        "m30": {"bars": 300, "tf_str": "m30"},
        "m15": {"bars": 300, "tf_str": "m15"},
        "m10": {"bars": 200, "tf_str": "m10"},
    }

    # ICT killzones (UTC hour ranges)
    KILLZONES = {
        "london_open":   (7,  9),
        "ny_open":       (12, 14),
        "london_close":  (15, 16),
        "asian_range":   (1,  4),
    }

    def __init__(self, fetcher, swing_lookback: int = 10):
        """
        Parameters
        ----------
        fetcher : Object with get_candles(symbol, tf, bars) method
                  OR a BaseBroker instance with get_market_data(symbol, tf, bars).
                  Both interfaces are supported transparently.
        """
        self.fetcher = fetcher
        self.swing_lookback = swing_lookback
        # Detect which interface the fetcher exposes
        self._use_market_data = (
            hasattr(fetcher, 'get_market_data')
            and not hasattr(fetcher, 'get_candles')
        )

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def get_confluence(self, symbol: str) -> ConfluenceResult:
        """Main entry point: build full 7-tier MTF confluence for a symbol."""
        result = ConfluenceResult(symbol=symbol)
        try:
            frames: Dict[str, pd.DataFrame] = {}
            for tf_name, cfg in self.TIMEFRAMES.items():
                df = self._fetch(symbol, cfg["tf_str"], cfg["bars"])
                if df is not None and len(df) >= 30:
                    frames[tf_name] = df

            if not frames:
                result.reasons.append("no_data_available")
                return result

            # Build per-timeframe context for all 7 tiers
            result.d1  = self._analyse_tf(frames.get("d1"),  "d1")
            result.h4  = self._analyse_tf(frames.get("h4"),  "h4")
            result.h3  = self._analyse_tf(frames.get("h3"),  "h3")
            result.h1  = self._analyse_tf(frames.get("h1"),  "h1")
            result.m30 = self._analyse_tf(frames.get("m30"), "m30")
            result.m15 = self._analyse_tf(frames.get("m15"), "m15")
            result.m10 = self._analyse_tf(frames.get("m10"), "m10")

            # Determine overall bias and score
            result = self._build_confluence(result, frames)

        except Exception as e:
            logger.error(f"MTFConfluence.get_confluence error: {e}", exc_info=True)
            result.reasons.append(f"error:{e}")

        return result

    # ─────────────────────────────────────────────────────────────────
    # Per-timeframe analysis
    # ─────────────────────────────────────────────────────────────────

    def _analyse_tf(self, df: Optional[pd.DataFrame], tf: str) -> Optional[TimeframeContext]:
        if df is None or len(df) < self.swing_lookback * 3:
            return None
        ctx = TimeframeContext(tf=tf)
        try:
            swings = self._find_swings(df)
            ctx.swing_high = swings["highs"][-1] if swings["highs"] else float(df["high"].max())
            ctx.swing_low  = swings["lows"][-1]  if swings["lows"]  else float(df["low"].min())

            ctx.trend = self._detect_trend(df, swings)
            ctx.bos, bos_dir = self._detect_bos(df, swings, ctx.trend)
            ctx.choch, ctx.choch_dir = self._detect_choch(df, swings, ctx.trend)

            ctx.atr = self._calc_atr_value(df)

            # Order blocks
            ob_level, ob_type = self._find_nearest_ob(df, ctx.trend)
            ctx.ob_level = ob_level
            ctx.ob_type  = ob_type

            # FVG
            fvg = self._find_nearest_fvg(df, ctx.trend)
            ctx.fvg_low    = fvg["low"]
            ctx.fvg_high   = fvg["high"]
            ctx.fvg_filled = fvg["filled"]

            # Premium/Discount (price relative to last swing range)
            ctx.premium_discount = self._classify_pd_zone(df, ctx)

            ctx.structure_score = self._score_tf(ctx)
        except Exception as e:
            logger.error(f"_analyse_tf({tf}) error: {e}", exc_info=True)
        return ctx

    # ─────────────────────────────────────────────────────────────────
    # Confluence building
    # ─────────────────────────────────────────────────────────────────

    def _build_confluence(
        self, result: ConfluenceResult, frames: Dict[str, pd.DataFrame]
    ) -> ConfluenceResult:
        d1, h4, h3, h1, m30, m15, m10 = (
            result.d1, result.h4, result.h3, result.h1,
            result.m30, result.m15, result.m10
        )
        last_price = self._last_close(frames)

        # ── Step 1: D1 macro bias (top of cascade) ───────────────────────────
        d1_bias = "neutral"
        if d1:
            if d1.trend == "bullish" or (d1.choch and d1.choch_dir == "bullish"):
                d1_bias = "bullish"
            elif d1.trend == "bearish" or (d1.choch and d1.choch_dir == "bearish"):
                d1_bias = "bearish"
        result.macro_aligned = d1_bias != "neutral"
        if result.macro_aligned:
            result.reasons.append(f"d1_macro_{d1_bias}")

        # ── Step 2: H4 structural bias ────────────────────────────────────────
        h4_bias = "neutral"
        if h4:
            if h4.trend == "bullish" or (h4.choch and h4.choch_dir == "bullish"):
                h4_bias = "bullish"
            elif h4.trend == "bearish" or (h4.choch and h4.choch_dir == "bearish"):
                h4_bias = "bearish"

        # Cascade: if D1 has bias, H4 must agree; if D1 neutral, H4 leads
        dominant_bias = d1_bias if d1_bias != "neutral" else h4_bias
        if h4_bias == "neutral" and d1_bias == "neutral":
            dominant_bias = "neutral"

        # ── Step 3: H3 intermediate confirmation ─────────────────────────────
        h3_confirmed = False
        if h3 and dominant_bias != "neutral":
            if h3.trend in (dominant_bias, "ranging") or \
               (h3.choch and h3.choch_dir == dominant_bias) or h3.bos:
                h3_confirmed = True
                result.reasons.append(f"h3_confirms_{dominant_bias}")

        # ── Step 4: HTF alignment — D1+H4+H3 ────────────────────────────────
        htf_votes = sum([
            d1_bias == dominant_bias and dominant_bias != "neutral",
            h4_bias == dominant_bias and dominant_bias != "neutral",
            h3_confirmed,
        ])
        result.htf_aligned = dominant_bias != "neutral" and htf_votes >= 2
        if result.htf_aligned:
            result.reasons.append(f"htf_aligned_{htf_votes}/3_tiers")

        # ── Step 5: H1 session confirmation ──────────────────────────────────
        h1_ok = False
        if h1 and dominant_bias != "neutral":
            if h1.trend in (dominant_bias, "ranging"):
                h1_ok = True
                result.reasons.append("h1_trend_aligned")
            if h1.bos:
                result.reasons.append("h1_bos")
            if h1.choch and h1.choch_dir == dominant_bias:
                h1_ok = True
                result.reasons.append("h1_choch_matches_htf")

        # ── Step 6: M30 sub-session filter ───────────────────────────────────
        m30_ok = False
        if m30 and dominant_bias != "neutral":
            if m30.trend in (dominant_bias, "ranging") or \
               (m30.choch and m30.choch_dir == dominant_bias):
                m30_ok = True
                result.reasons.append("m30_confirms")

        result.mtf_aligned = h1_ok or m30_ok

        # ── Step 7: M15 precision entry ───────────────────────────────────────
        m15_ok = False
        if m15 and result.htf_aligned:
            if m15.choch and m15.choch_dir == dominant_bias:
                m15_ok = True
                result.reasons.append("m15_choch_entry")
            elif m15.bos:
                m15_ok = True
                result.reasons.append("m15_bos_entry")

        # ── Step 8: M10 ultra-precision trigger ───────────────────────────────
        m10_ok = False
        if m10 and (m15_ok or result.htf_aligned):
            if m10.choch and m10.choch_dir == dominant_bias:
                m10_ok = True
                result.reasons.append("m10_choch_trigger")
            elif m10.bos and m15_ok:
                m10_ok = True
                result.reasons.append("m10_bos_trigger")
            elif not m10.fvg_filled and m10.fvg_low > 0:
                m10_ok = True
                result.reasons.append("m10_fvg_entry")

        result.ltf_confirmed = m15_ok or m10_ok

        # ── Step 9: Tier count (0-7) ──────────────────────────────────────────
        result.tier_score = sum([
            d1_bias == dominant_bias and dominant_bias != "neutral",
            h4_bias == dominant_bias and dominant_bias != "neutral",
            h3_confirmed,
            h1_ok,
            m30_ok,
            m15_ok,
            m10_ok,
        ])

        # ── Step 10: Premium/Discount filter ─────────────────────────────────
        for ctx in [d1, h4]:
            if ctx:
                if (dominant_bias == "bullish" and ctx.premium_discount == "discount") or \
                   (dominant_bias == "bearish" and ctx.premium_discount == "premium"):
                    result.premium_discount_ok = True
                    result.reasons.append(f"pd_zone_{ctx.tf}_{ctx.premium_discount}")
                    break

        # ── Step 11: Liquidity sweep (M15 or M10) ────────────────────────────
        for sweep_tf in ["m10", "m15"]:
            if frames.get(sweep_tf) is not None:
                if self._check_liquidity_sweep(frames[sweep_tf]):
                    result.liquidity_swept = True
                    result.reasons.append(f"{sweep_tf}_liquidity_swept")
                    break

        # ── Step 12: Killzone check ───────────────────────────────────────────
        result.killzone_active = self._is_killzone_active()
        if result.killzone_active:
            result.reasons.append("killzone_active")

        # ── Step 13: Entry zone and risk levels ───────────────────────────────
        # Use M10 ATR first (most precise), fall back up the chain
        atr = 0.0
        for ctx in [m10, m15, m30, h1, h4]:
            if ctx and ctx.atr > 0:
                atr = ctx.atr
                break
        atr = atr or 0.0001

        result = self._compute_levels(result, dominant_bias, last_price, atr)

        # ── Step 14: Final bias and score ─────────────────────────────────────
        result.bias  = dominant_bias
        result.score = self._compute_score(result)

        logger.info(
            f"[MTF7] {result.symbol} | bias={result.bias} | score={result.score:.3f} | "
            f"tiers={result.tier_score}/7 | htf={result.htf_aligned} | "
            f"mtf={result.mtf_aligned} | ltf={result.ltf_confirmed} | "
            f"reasons={result.reasons[:5]}"
        )
        return result

    def _compute_levels(
        self,
        result: ConfluenceResult,
        bias: str,
        price: float,
        atr: float,
    ) -> ConfluenceResult:
        # Use most precise available context for entry zone
        entry_ctx = result.m10 or result.m15 or result.m30 or result.h1
        if bias == "bullish":
            if entry_ctx and entry_ctx.ob_level > 0 and entry_ctx.ob_type == "bullish" and entry_ctx.ob_level < price:
                result.nearest_ob   = entry_ctx.ob_level
                result.entry_low    = entry_ctx.ob_level
                result.entry_high   = entry_ctx.ob_level + atr * 0.5
            elif entry_ctx and not entry_ctx.fvg_filled and entry_ctx.fvg_low > 0:
                result.nearest_fvg_low  = entry_ctx.fvg_low
                result.nearest_fvg_high = entry_ctx.fvg_high
                result.entry_low  = entry_ctx.fvg_low
                result.entry_high = entry_ctx.fvg_high
            else:
                result.entry_low  = price - atr * 0.3
                result.entry_high = price + atr * 0.2
            result.invalidation = result.entry_low - atr * 1.5
            result.target_1r    = result.entry_high + atr * 1.5
            result.target_2r    = result.entry_high + atr * 3.0
            result.target_3r    = result.entry_high + atr * 5.0

        elif bias == "bearish":
            if entry_ctx and entry_ctx.ob_level > 0 and entry_ctx.ob_type == "bearish" and entry_ctx.ob_level > price:
                result.nearest_ob   = entry_ctx.ob_level
                result.entry_high   = entry_ctx.ob_level
                result.entry_low    = entry_ctx.ob_level - atr * 0.5
            elif entry_ctx and not entry_ctx.fvg_filled and entry_ctx.fvg_high > 0:
                result.nearest_fvg_low  = entry_ctx.fvg_low
                result.nearest_fvg_high = entry_ctx.fvg_high
                result.entry_high = entry_ctx.fvg_high
                result.entry_low  = entry_ctx.fvg_low
            else:
                result.entry_high = price + atr * 0.3
                result.entry_low  = price - atr * 0.2
            result.invalidation = result.entry_high + atr * 1.5
            result.target_1r    = result.entry_low - atr * 1.5
            result.target_2r    = result.entry_low - atr * 3.0
            result.target_3r    = result.entry_low - atr * 5.0

        return result

    def _compute_score(self, r: ConfluenceResult) -> float:
        score = 0.0
        # Macro and HTF alignment (most important)
        if r.macro_aligned:        score += 0.10
        if r.htf_aligned:          score += 0.25
        # Mid-tier confirmation
        if r.mtf_aligned:          score += 0.15
        # LTF precision entry
        if r.ltf_confirmed:        score += 0.20
        # Context quality modifiers
        if r.premium_discount_ok:  score += 0.10
        if r.liquidity_swept:      score += 0.08
        if r.killzone_active:      score += 0.07
        if r.bias != "neutral":    score += 0.05
        # Tier count bonus: every 2 confirming tiers above 3 = +0.01
        tier_bonus = max(0, (r.tier_score - 3)) * 0.01
        score += tier_bonus
        return round(min(score, 1.0), 4)

    # ─────────────────────────────────────────────────────────────────
    # Order block detection (institutional candle-based)
    # ─────────────────────────────────────────────────────────────────

    def _find_nearest_ob(
        self, df: pd.DataFrame, trend: str
    ) -> Tuple[float, str]:
        """
        An order block is the last bearish candle before a bullish impulse
        (bullish OB) or last bullish candle before a bearish impulse (bearish OB).
        Returns (price_level, type).
        """
        try:
            last_price = float(df["close"].iloc[-1])
            lookback = min(50, len(df) - 3)

            if trend == "bullish":
                # Look for bearish OB candles below current price
                for i in range(len(df) - 2, len(df) - lookback, -1):
                    candle = df.iloc[i]
                    next3 = df.iloc[i + 1: i + 4]
                    # Bearish candle followed by strong bullish move
                    if (candle["close"] < candle["open"] and
                            next3["close"].max() > candle["open"] * 1.001):
                        ob_mid = (candle["high"] + candle["low"]) / 2
                        if ob_mid < last_price:
                            return float(ob_mid), "bullish"

            elif trend == "bearish":
                # Look for bullish OB candles above current price
                for i in range(len(df) - 2, len(df) - lookback, -1):
                    candle = df.iloc[i]
                    next3 = df.iloc[i + 1: i + 4]
                    # Bullish candle followed by strong bearish move
                    if (candle["close"] > candle["open"] and
                            next3["close"].min() < candle["open"] * 0.999):
                        ob_mid = (candle["high"] + candle["low"]) / 2
                        if ob_mid > last_price:
                            return float(ob_mid), "bearish"

        except Exception as e:
            logger.debug(f"_find_nearest_ob error: {e}")
        return 0.0, ""

    # ─────────────────────────────────────────────────────────────────
    # Fair Value Gap detection (3-candle imbalance)
    # ─────────────────────────────────────────────────────────────────

    def _find_nearest_fvg(
        self, df: pd.DataFrame, trend: str
    ) -> Dict:
        """
        A bullish FVG: low of candle[i+2] > high of candle[i].
        A bearish FVG: high of candle[i+2] < low of candle[i].
        Returns the most recent unfilled gap relevant to the trend.
        """
        result = {"low": 0.0, "high": 0.0, "filled": True}
        try:
            last_price = float(df["close"].iloc[-1])
            lookback = min(100, len(df) - 3)

            for i in range(len(df) - 3, len(df) - lookback, -1):
                c0, c2 = df.iloc[i], df.iloc[i + 2]

                if trend == "bullish":
                    # Bullish FVG: gap between high[i] and low[i+2]
                    if c2["low"] > c0["high"]:
                        fvg_low  = float(c0["high"])
                        fvg_high = float(c2["low"])
                        # Check if filled by subsequent price action
                        subsequent = df.iloc[i + 3:]
                        filled = not subsequent.empty and (subsequent["low"] <= fvg_low).any()
                        if not filled and fvg_low < last_price:
                            return {"low": fvg_low, "high": fvg_high, "filled": False}

                elif trend == "bearish":
                    # Bearish FVG: gap between low[i] and high[i+2]
                    if c2["high"] < c0["low"]:
                        fvg_low  = float(c2["high"])
                        fvg_high = float(c0["low"])
                        subsequent = df.iloc[i + 3:]
                        filled = not subsequent.empty and (subsequent["high"] >= fvg_high).any()
                        if not filled and fvg_high > last_price:
                            return {"low": fvg_low, "high": fvg_high, "filled": False}

        except Exception as e:
            logger.debug(f"_find_nearest_fvg error: {e}")
        return result

    # ─────────────────────────────────────────────────────────────────
    # Premium / Discount zones (50% of swing range)
    # ─────────────────────────────────────────────────────────────────

    def _classify_pd_zone(self, df: pd.DataFrame, ctx: TimeframeContext) -> str:
        """
        ICT concept: buy in discount (below 50% of swing range),
        sell in premium (above 50%).
        """
        try:
            if ctx.swing_high <= ctx.swing_low:
                return "neutral"
            midpoint = (ctx.swing_high + ctx.swing_low) / 2
            last_close = float(df["close"].iloc[-1])
            if last_close < midpoint:
                return "discount"
            elif last_close > midpoint:
                return "premium"
        except Exception:
            pass
        return "neutral"

    # ─────────────────────────────────────────────────────────────────
    # Liquidity sweep (M15)
    # ─────────────────────────────────────────────────────────────────

    def _check_liquidity_sweep(self, df: pd.DataFrame) -> bool:
        """
        Sweep: last candle wicked through a recent swing level
        but closed back inside — engineered liquidity grab.
        """
        try:
            swings = self._find_swings(df)
            last = df.iloc[-1]
            if swings["highs"] and last["high"] > swings["highs"][-1] and last["close"] < swings["highs"][-1]:
                return True
            if swings["lows"] and last["low"] < swings["lows"][-1] and last["close"] > swings["lows"][-1]:
                return True
        except Exception:
            pass
        return False

    # ─────────────────────────────────────────────────────────────────
    # Killzone detection
    # ─────────────────────────────────────────────────────────────────

    def _is_killzone_active(self) -> bool:
        hour = datetime.now(timezone.utc).hour
        for name, (start, end) in self.KILLZONES.items():
            if start <= hour < end:
                return True
        return False

    # ─────────────────────────────────────────────────────────────────
    # Structure helpers (shared with MarketStructureAnalyzer)
    # ─────────────────────────────────────────────────────────────────

    def _find_swings(self, df: pd.DataFrame) -> Dict:
        lb = self.swing_lookback
        highs, lows = [], []
        for i in range(lb, len(df) - lb):
            w_h = df["high"].iloc[i - lb: i + lb + 1]
            w_l = df["low"].iloc[i - lb: i + lb + 1]
            if df["high"].iloc[i] == w_h.max():
                highs.append((i, float(df["high"].iloc[i])))
            if df["low"].iloc[i] == w_l.min():
                lows.append((i, float(df["low"].iloc[i])))
        return {
            "highs": [h[1] for h in highs],
            "lows":  [l[1] for l in lows],
            "high_idx": [h[0] for h in highs],
            "low_idx":  [l[0] for l in lows],
        }

    def _detect_trend(self, df: pd.DataFrame, swings: Dict) -> str:
        highs, lows = swings["highs"], swings["lows"]
        if len(highs) < 2 or len(lows) < 2:
            if len(df) >= 50:
                s20 = df["close"].rolling(20).mean().iloc[-1]
                s50 = df["close"].rolling(50).mean().iloc[-1]
                return "bullish" if s20 > s50 else ("bearish" if s20 < s50 else "ranging")
            return "ranging"
        rh, rl = highs[-4:], lows[-4:]
        hh = all(rh[i] > rh[i - 1] for i in range(1, len(rh)))
        hl = all(rl[i] > rl[i - 1] for i in range(1, len(rl)))
        lh = all(rh[i] < rh[i - 1] for i in range(1, len(rh)))
        ll = all(rl[i] < rl[i - 1] for i in range(1, len(rl)))
        if hh and hl: return "bullish"
        if lh and ll: return "bearish"

        # Strict swing structure failed — use multi-indicator vote as fallback.
        # This handles strong trending assets (e.g. metals with ADX>40) where
        # the last 4 swings aren't perfectly sequential but price is clearly
        # directional.  A 3/4 majority vote is required so we don't flip on noise.
        if len(df) >= 50:
            close = df["close"]
            s20   = close.rolling(20).mean()
            s50   = close.rolling(50).mean()
            s200  = close.rolling(200).mean() if len(df) >= 200 else None

            last_close = float(close.iloc[-1])
            last_s20   = float(s20.iloc[-1])
            last_s50   = float(s50.iloc[-1])

            # Slope: direction of SMA over last N bars
            slope_n = max(5, len(df) // 20)
            s20_slope = float(s20.iloc[-1]) - float(s20.iloc[-slope_n])
            s50_slope = float(s50.iloc[-1]) - float(s50.iloc[-slope_n])

            bull_votes = sum([
                last_s20 > last_s50,                              # fast > slow MA
                s20_slope > 0,                                    # fast MA rising
                s50_slope > 0,                                    # slow MA rising
                last_close > last_s50,                            # price above slow MA
                (s200 is not None and last_close > float(s200.iloc[-1])),  # price above 200
            ])
            bear_votes = sum([
                last_s20 < last_s50,
                s20_slope < 0,
                s50_slope < 0,
                last_close < last_s50,
                (s200 is not None and last_close < float(s200.iloc[-1])),
            ])
            max_votes = 5 if s200 is not None else 4
            required  = max_votes - 1  # 4/5 or 3/4 — clear majority
            if bull_votes >= required: return "bullish"
            if bear_votes >= required: return "bearish"
        return "ranging"

    def _detect_bos(self, df: pd.DataFrame, swings: Dict, trend: str) -> Tuple[bool, str]:
        last = float(df["close"].iloc[-1])
        if trend == "bullish" and swings["highs"] and last > swings["highs"][-1]:
            return True, "bullish"
        if trend == "bearish" and swings["lows"] and last < swings["lows"][-1]:
            return True, "bearish"
        return False, ""

    def _detect_choch(self, df: pd.DataFrame, swings: Dict, trend: str) -> Tuple[bool, str]:
        last = float(df["close"].iloc[-1])
        if trend == "bearish" and swings["highs"] and last > swings["highs"][-1]:
            return True, "bullish"
        if trend == "bullish" and swings["lows"] and last < swings["lows"][-1]:
            return True, "bearish"
        return False, ""

    def _calc_atr_value(self, df: pd.DataFrame, period: int = 14) -> float:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"]  - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.rolling(period).mean()
        val = atr_series.dropna()
        return float(val.iloc[-1]) if len(val) > 0 else 0.0

    def _score_tf(self, ctx: TimeframeContext) -> float:
        s = 0.0
        if ctx.trend != "ranging":    s += 0.3
        if ctx.bos:                   s += 0.2
        if ctx.choch:                 s += 0.2
        if ctx.ob_level > 0:          s += 0.15
        if not ctx.fvg_filled:        s += 0.15
        return min(s, 1.0)

    def _last_close(self, frames: Dict) -> float:
        for tf in ["m10", "m15", "m30", "h1", "h3", "h4", "d1"]:
            if tf in frames and len(frames[tf]) > 0:
                return float(frames[tf]["close"].iloc[-1])
        return 0.0

    def _fetch(self, symbol: str, tf: str, bars: int) -> Optional[pd.DataFrame]:
        """
        Fetch candle data from either get_candles() or get_market_data()
        depending on what the injected fetcher/broker exposes.
        """
        try:
            if self._use_market_data:
                # BaseBroker interface (MT5Adapter, IBKRAdapter, CTraderAdapter)
                return self.fetcher.get_market_data(symbol, tf, bars)
            else:
                # BrokerDataFetcher / MT5DataFetcher interface
                return self.fetcher.get_candles(symbol, tf, bars)
        except Exception as e:
            logger.warning(f"MTF fetch {symbol}/{tf}: {e}")
            return None
