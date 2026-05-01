"""
signal_scorer.py — Composite trade signal scoring engine (AI EA v4)
--------------------------------------------------------------------
Combines multiple quality dimensions into a single normalised score [0, 1].
Only high-quality signals (score >= MIN_SCORE) should be sent to execution.

Scoring dimensions
------------------
1. ML probability      (0–40 pts)  — raw model confidence
2. Trend alignment     (0–25 pts)  — signal matches market structure trend
3. Structure quality   (0–20 pts)  — BOS / CHoCH / sweep presence
4. Volatility regime   (0–10 pts)  — trading in good ATR range
5. Session bonus       (0–5  pts)  — London / NY overlap bonus

MAX possible = 100 pts → normalised to [0, 1]
"""

import logging
from datetime import datetime, time, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)

MIN_SCORE         = 0.58    # minimum composite score to allow execution (raised in v5)
SESSION_UTC_START = time(13, 0)   # London/NY overlap start
SESSION_UTC_END   = time(16, 0)   # London/NY overlap end

# v5 scoring weights — can be overridden by regime detector
DEFAULT_WEIGHTS = {
    "ml_prob":    0.35,   # 35 pts max
    "mtf":        0.25,   # 25 pts max (NEW: MTF confluence)
    "trend":      0.15,   # 15 pts max (reduced — MTF takes some weight)
    "structure":  0.15,   # 15 pts max
    "volatility": 0.05,   # 5  pts max
    "session":    0.05,   # 5  pts max
}


class SignalScorer:
    """
    Score a prospective trade and decide whether it is worth taking.

    Parameters
    ----------
    min_score : float  Minimum composite score [0, 1] to return allow=True.
    """

    def __init__(self, min_score: float = MIN_SCORE, weights: dict = None):
        self.min_score = min_score
        self.weights   = weights or DEFAULT_WEIGHTS.copy()

    def update_weights(self, new_weights: dict) -> None:
        """Allow regime detector to adjust scoring weights at runtime."""
        self.weights = {**DEFAULT_WEIGHTS, **new_weights}
        logger.info(f"[SCORER] Weights updated: {self.weights}")

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        signal: str,
        ml_probability: float,
        structure: Dict,
        atr_pips: float,
        symbol: str = "",
        utc_now: Optional[datetime] = None,
        mtf_result=None,          # v5: ConfluenceResult from MTFConfluenceEngine
    ) -> Dict:
        """
        Compute the composite quality score for a proposed trade.

        Parameters
        ----------
        signal         : 'BUY' | 'SELL'
        ml_probability : probability from SignalEngine [0, 1]
        structure      : dict from MarketStructureAnalyzer.analyse()
        atr_pips       : current ATR in pips (symbol-adjusted)
        symbol         : instrument name (for volatility thresholds)
        utc_now        : override for session detection (default: now)

        Returns
        -------
        dict with keys:
            score        float [0, 1]
            allow        bool
            breakdown    dict — points per dimension
            reason       str  — human-readable summary
        """
        if signal not in ("BUY", "SELL"):
            return self._reject("invalid_signal", 0.0)

        w = self.weights

        pts_ml        = self._score_ml(ml_probability)       * (w.get("ml_prob",    0.35) / 0.35)
        pts_mtf       = self._score_mtf(signal, mtf_result)  * (w.get("mtf",        0.25) / 0.25)
        pts_trend     = self._score_trend(signal, structure)  * (w.get("trend",      0.15) / 0.15)
        pts_structure = self._score_structure(structure)      * (w.get("structure",  0.15) / 0.15)
        pts_volatility= self._score_volatility(atr_pips, symbol) * (w.get("volatility", 0.05) / 0.05)
        pts_session   = self._score_session(utc_now)          * (w.get("session",    0.05) / 0.05)

        # Normalise to 100-point scale using weights
        raw_total = (
            pts_ml        * w.get("ml_prob",    0.35) * 100 +
            pts_mtf       * w.get("mtf",        0.25) * 100 +
            pts_trend     * w.get("trend",      0.15) * 100 +
            pts_structure * w.get("structure",  0.15) * 100 +
            pts_volatility* w.get("volatility", 0.05) * 100 +
            pts_session   * w.get("session",    0.05) * 100
        )
        # pts_xxx are already in [0,max] — compute as weighted sum / 100
        total = pts_ml + pts_mtf + pts_trend + pts_structure + pts_volatility + pts_session
        normalised = round(total / 100.0, 4)

        allow = normalised >= self.min_score

        breakdown = {
            "ml_prob":       round(pts_ml,         2),
            "mtf_confluence": round(pts_mtf,       2),
            "trend_align":   round(pts_trend,       2),
            "structure":     round(pts_structure,   2),
            "volatility":    round(pts_volatility,  2),
            "session":       round(pts_session,     2),
        }
        reason = (
            f"{signal} {symbol} score={normalised:.3f} "
            f"[ml={pts_ml:.0f} mtf={pts_mtf:.0f} trend={pts_trend:.0f} "
            f"struct={pts_structure:.0f} vol={pts_volatility:.0f} "
            f"sess={pts_session:.0f}] → {'ALLOW' if allow else 'SKIP'}"
        )
        logger.info(f"[SCORER] {reason}")

        return {
            "score":     normalised,
            "allow":     allow,
            "breakdown": breakdown,
            "reason":    reason,
        }

    # ── Scoring components ────────────────────────────────────────────────────

    @staticmethod
    def _score_mtf(signal: str, mtf_result=None) -> float:
        """
        25 pts max. 7-tier MTF scoring — maximum edge extraction.
        Each confirming tier adds points; full cascade = 25.
        """
        if mtf_result is None:
            return 12.5   # neutral if MTF not computed — no penalty
        try:
            pts = 0.0
            bias          = getattr(mtf_result, "bias", "neutral")
            htf_aligned   = getattr(mtf_result, "htf_aligned",  False)
            mtf_aligned   = getattr(mtf_result, "mtf_aligned",  False)
            ltf_confirmed = getattr(mtf_result, "ltf_confirmed", False)
            macro_aligned = getattr(mtf_result, "macro_aligned", False)
            tier_score    = getattr(mtf_result, "tier_score",    0)
            swept         = getattr(mtf_result, "liquidity_swept", False)
            kz            = getattr(mtf_result, "killzone_active", False)
            pd_ok         = getattr(mtf_result, "premium_discount_ok", False)

            # Core bias agreement (signal must agree with cascade bias)
            if (signal == "BUY"  and bias == "bullish") or \
               (signal == "SELL" and bias == "bearish"):
                pts += 8.0
            elif bias == "neutral":
                pts += 3.0  # neutral bias: partial credit

            # Macro D1 alignment (new in 7-tier)
            if macro_aligned:        pts += 2.0
            # HTF alignment (D1+H4+H3)
            if htf_aligned:          pts += 5.0
            # Mid-tier alignment (H1+M30)
            if mtf_aligned:          pts += 3.0
            # LTF precision confirmed (M15+M10)
            if ltf_confirmed:        pts += 3.0
            # Tier count bonus: 0.5 per confirming tier above 3
            pts += max(0, tier_score - 3) * 0.5
            # Context quality
            if pd_ok:                pts += 1.5
            if swept:                pts += 1.0
            if kz:                   pts += 1.0

            return round(min(pts, 25.0), 2)
        except Exception:
            return 12.5

    @staticmethod
    def _score_ml(prob: float) -> float:
        """
        40 pts max.
        prob < 0.65  → 0 pts (should not reach scorer with prob this low)
        prob 0.65    → 10 pts
        prob 0.80    → 25 pts
        prob 1.00    → 40 pts
        """
        if prob < 0.65:
            return 0.0
        return round(min(40.0, (prob - 0.65) / 0.35 * 40.0), 2)

    @staticmethod
    def _score_trend(signal: str, structure: Dict) -> float:
        """
        25 pts max.
        Signal aligned with structural trend → full points.
        Signal against trend but CHoCH confirmed → partial.
        Ranging market → small bonus only on sweep.
        """
        trend = structure.get("trend", "ranging")
        choch = structure.get("choch", False)
        sweep = structure.get("liquidity_sweep", False)

        if signal == "BUY":
            if trend == "bullish":
                return 25.0
            if trend == "bearish" and choch and structure.get("choch_direction") == "bullish":
                return 15.0
            if trend == "ranging" and sweep:
                return 10.0
            return 0.0

        # SELL
        if trend == "bearish":
            return 25.0
        if trend == "bullish" and choch and structure.get("choch_direction") == "bearish":
            return 15.0
        if trend == "ranging" and sweep:
            return 10.0
        return 0.0

    @staticmethod
    def _score_structure(structure: Dict) -> float:
        """
        20 pts max.
        Uses the pre-computed structure_score [0, 1] × 20.
        """
        return round(structure.get("structure_score", 0.0) * 20.0, 2)

    @staticmethod
    def _score_volatility(atr_pips: float, symbol: str) -> float:
        """
        10 pts max.
        Trade in a 'good' volatility range — not dead, not spiking.
        Thresholds adjusted per instrument class.
        """
        sym = symbol.upper()

        # Define ideal ATR range per instrument
        if "BTC" in sym or "ETH" in sym:
            low, ideal_low, ideal_high, high = 500, 1500, 5000, 15000
        elif "XAU" in sym or "GOLD" in sym:
            low, ideal_low, ideal_high, high = 50, 150, 500, 2000
        elif "OIL" in sym:
            low, ideal_low, ideal_high, high = 10, 30, 100, 300
        else:  # Forex default
            low, ideal_low, ideal_high, high = 3, 8, 30, 80

        if atr_pips <= low or atr_pips >= high:
            return 0.0          # too dead or too wild
        if ideal_low <= atr_pips <= ideal_high:
            return 10.0         # ideal range
        if atr_pips < ideal_low:
            ratio = (atr_pips - low) / (ideal_low - low) if ideal_low != low else 0
            return round(ratio * 10.0, 2)
        # atr_pips > ideal_high
        ratio = (high - atr_pips) / (high - ideal_high) if high != ideal_high else 0
        return round(max(0.0, ratio) * 10.0, 2)

    @staticmethod
    @staticmethod
    def _score_session(utc_now: Optional[datetime] = None) -> float:
        """
        5 pts max.  Bug 3 FIX: the 03:00–07:00 UTC window was completely unscored
        (returned 0 pts), dragging every composite score down during early London
        pre-open and the Asian metals session.  New tiers:

          London/NY overlap  13:00–16:00 UTC → 5 pts  (peak liquidity)
          London alone        07:00–16:00 UTC → 3 pts
          NY alone           16:00–22:00 UTC → 2 pts
          Asian + pre-London 00:00–07:00 UTC → 2 pts  (metals/crypto still active)
          True dead zone     22:00–00:00 UTC → 0 pts
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)
        t = utc_now.time()

        OVERLAP_START = time(13, 0)
        OVERLAP_END   = time(16, 0)
        LONDON_START  = time(7,  0)
        LONDON_END    = time(16, 0)
        NY_START      = time(16, 0)
        NY_END        = time(22, 0)
        # Asian / pre-London covers 00:00–07:00 UTC — metals and BTC are liquid here
        ASIAN_START   = time(0,  0)
        ASIAN_END     = time(7,  0)

        in_overlap = OVERLAP_START <= t < OVERLAP_END
        in_london  = LONDON_START  <= t < LONDON_END
        in_ny      = NY_START      <= t < NY_END
        in_asian   = ASIAN_START   <= t < ASIAN_END

        if in_overlap:
            return 5.0
        if in_london:
            return 3.0
        if in_ny:
            return 2.0
        if in_asian:
            return 2.0
        # 22:00–00:00 UTC: genuine dead zone
        return 0.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _reject(reason: str, score: float) -> Dict:
        return {
            "score": score,
            "allow": False,
            "breakdown": {},
            "reason": reason,
        }
