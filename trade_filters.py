"""
trade_filters.py
----------------
Multi-layer trade filter pipeline.

Filters applied (in order):
  1. Session filter   — London / New York only (configurable)
  2. Spread filter    — skip if spread > max_spread_pips (asset-class aware)
  3. Volatility filter — skip if ATR too low (asset-class aware, price-based)

All filters return (passed: bool, reason: str).

Spread and ATR are always passed in PIP terms so the filter is
broker-agnostic (MT5, IBKR, cTrader all use the same unit after
conversion in ai_ea.py).

ATR volatility thresholds are asset-class-aware:
  - Comparison is done in PRICE terms (atr_price = atr_pips * pip_size)
    but since callers pass atr_pips we back-convert using point.
  - Each asset class has its own USD-equivalent ATR floor, eliminating
    the incorrect ×100 / ×200 pip multipliers that blocked Gold and BTC.
"""

import logging
from datetime import datetime, time, timezone
from typing import Tuple, Optional
import pytz

logger = logging.getLogger(__name__)


# ── Session definitions (UTC) ─────────────────────────────────────────────────
# SA time = UTC+2.  Midnight SA = 22:00 UTC previous day.
# Previously "new_york" ended at 22:00 UTC — this BLOCKED the 22:00–23:59 UTC
# window (midnight–01:59 SA time) and the entire Asian session (23:00–07:00 UTC).
# Trends starting at the Asian open or continuing through midnight SA time were
# silently blocked.
#
# Fix: add "asian" session and extend new_york to 23:00 UTC so no hour is
# dead-zoned unless it's truly quiet (03:00–06:00 UTC = deep Asian night, still
# allowed since BTC/metals can run).  require_session defaults to False now so
# the filter is informational / advisory rather than a hard block.  Operators
# who want hard session enforcement should pass require_session=True explicitly.
SESSIONS = {
    "asian":    {"start": time(0,  0), "end": time(9,  0)},   # 00:00–09:00 UTC (Tokyo + early London)
    "london":   {"start": time(7,  0), "end": time(16, 0)},   # 07:00–16:00 UTC
    "new_york": {"start": time(13, 0), "end": time(23, 0)},   # 13:00–23:00 UTC (extended to cover midnight SA)
    "overlap":  {"start": time(13, 0), "end": time(16, 0)},   # London/NY overlap
    "always":   {"start": time(0,  0), "end": time(23, 59)},  # full day (24h instruments: BTC, XAU)
}

# ── Spread limits (pips) per asset class ──────────────────────────────────────
# Callers pass spread in pips (broker-normalised).  Limits here are also pips.
# pip = point * 10 for all assets (MT5 convention).
#   EURUSD pip = 0.0001, so 30 pips = 0.003 spread max
#   XAUUSD pip = 0.1,    so 80 pips = $8    spread max (reasonable for gold)
#   BTCUSD pip = 10,     so 150 pips = $1500 spread max (wide weekend BTC)
SPREAD_LIMITS_PIPS = {
    "forex":   30.0,    # 30 pips max forex spread
    "metal":   80.0,    # 80 pips gold/silver (pip=0.1 → $8 max)
    "index":   60.0,    # 60 pips indices
    "energy":  60.0,    # 60 pips oil/gas
    "crypto":  500.0,   # 500 pips BTC/ETH (pip=$1 at point=0.01 → $500 max; covers wide weekend spreads)
    "stock":   50.0,
    "default": 50.0,
}

# ── ATR minimums in PRICE terms (NOT pips) ────────────────────────────────────
# These are the minimum ATR values in the instrument's native price unit.
# Checking in price terms is asset-class agnostic and always meaningful.
#   EURUSD: 0.0003 price  = 3 pips        (very quiet forex)
#   XAUUSD: 0.30 price    = $0.30          (extremely quiet gold — still tradeable)
#   BTCUSD: 20.0 price    = $20            (quiet BTC — still tradeable)
#   US100:  0.50 price    = 0.50 index pts
#   USOIL:  0.03 price    = 3 cents        (quiet oil)
ATR_MIN_PRICE = {
    "forex":   0.0003,   # $3 per standard lot if hit exactly
    "metal":   0.30,     # $0.30 gold ATR floor (in $)
    "index":   0.50,     # 0.5 index points
    "energy":  0.03,     # $0.03 oil
    "crypto":  20.0,     # $20 BTC/ETH ATR floor
    "stock":   0.05,
    "default": 0.0003,
}


def _asset_class(symbol: str) -> str:
    """Classify symbol by asset class for threshold lookup."""
    u = symbol.upper().strip("._-#")
    if any(k in u for k in ("XAU", "GOLD", "XAG", "SILVER", "XPT", "XPD")):
        return "metal"
    if any(k in u for k in ("BTC", "ETH", "LTC", "XRP", "BNB", "ADA", "SOL", "DOT", "DOGE")):
        return "crypto"
    if any(k in u for k in ("OIL", "BRENT", "WTI", "NATGAS", "GAS", "XBR", "XTI")):
        return "energy"
    if any(k in u for k in ("US30", "US500", "US100", "UK100", "GER", "DAX", "FRA",
                              "JPN", "AUS200", "HKG", "STOXX", "SPX", "NDX", "DJI",
                              "FTSE", "CAC", "NIKKEI", "NAS", "DOW")):
        return "index"
    if len(u.rstrip("0123456789")) in (5, 6, 7, 8):
        return "forex"
    return "default"


class TradeFilters:
    """
    Stateless filter pipeline.  Call `check_all()` to run every filter.

    Parameters
    ----------
    max_spread_pips  : Default max spread in pips for forex.
                       Asset-class specific limits from SPREAD_LIMITS_PIPS override this.
    min_atr_pips     : Kept for backward compat but not used internally —
                       ATR check uses price-based thresholds from ATR_MIN_PRICE.
    allowed_sessions : Session names from SESSIONS dict to allow trading in.
    require_session  : If False, skip session filter entirely.
    """

    def __init__(
        self,
        max_spread_pips: float = 30.0,
        min_atr_pips: float = 5.0,         # legacy param — kept for compat
        allowed_sessions: tuple = ("asian", "london", "new_york"),
        require_session: bool = False,   # FIX v14: False = advisory only; set True for strict enforcement
    ):
        self.max_spread_pips  = max_spread_pips
        self.min_atr_pips     = min_atr_pips   # kept for any external callers
        self.allowed_sessions = allowed_sessions
        self.require_session  = require_session

    # ── Aggregate check ───────────────────────────────────────────────────────

    def check_all(
        self,
        spread_pips: float,
        atr_pips: Optional[float],
        symbol: str = "",
        utc_now: Optional[datetime] = None,
        point: float = 0.0,         # instrument point size (needed for ATR price conversion)
    ) -> Tuple[bool, str]:
        """
        Run all filters.  Returns (passed, reason).
        If any filter fails returns (False, <reason>).

        Parameters
        ----------
        spread_pips : Spread already converted to pips (spread_points / 10 for MT5).
        atr_pips    : ATR in pips (atr_price / (point * 10)).  None = skip check.
        symbol      : Instrument name used for asset-class detection.
        utc_now     : Override for session check (default = datetime.now(timezone.utc)).
        point       : MT5/broker point size.  Used to back-convert atr_pips → price
                      for the volatility check.  If 0, the price check is skipped.
        """
        ok, reason = self.check_session(utc_now)
        if not ok:
            return False, reason

        ok, reason = self.check_spread(spread_pips, symbol)
        if not ok:
            return False, reason

        if atr_pips is not None and point > 0:
            ok, reason = self.check_volatility(atr_pips, symbol, point)
            if not ok:
                return False, reason

        return True, "all_filters_passed"

    # ── Individual filters ────────────────────────────────────────────────────

    def check_session(self, utc_now: Optional[datetime] = None) -> Tuple[bool, str]:
        """Only trade during allowed sessions.

        FIX v14: Added midnight-wrap support so sessions that cross 00:00 UTC
        are handled correctly (e.g., a session defined as 22:00–02:00 UTC).
        Also, default allowed_sessions now includes 'asian' so the 22:00–09:00
        UTC window (midnight SA onwards) is never silently blocked.
        """
        if not self.require_session:
            return True, "session_filter_disabled"

        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        current_time = utc_now.time()

        for session_name in self.allowed_sessions:
            sess = SESSIONS.get(session_name)
            if sess is None:
                continue
            start, end = sess["start"], sess["end"]
            if start <= end:
                # Normal (same-day) session
                in_sess = start <= current_time <= end
            else:
                # Midnight-crossing session (e.g., 22:00–02:00 UTC)
                in_sess = current_time >= start or current_time <= end
            if in_sess:
                return True, f"in_{session_name}_session"

        return False, f"outside_sessions_{list(self.allowed_sessions)}"

    def check_spread(self, spread_pips: float, symbol: str = "") -> Tuple[bool, str]:
        """
        Reject trades when spread is too wide.
        Limits are asset-class specific (see SPREAD_LIMITS_PIPS).
        """
        cls = _asset_class(symbol)
        limit = SPREAD_LIMITS_PIPS.get(cls, SPREAD_LIMITS_PIPS["default"])

        if spread_pips > limit:
            return False, (
                f"spread_too_wide:{spread_pips:.1f}>{limit:.1f}_pips"
                f"[{cls}:{symbol}]"
            )
        return True, f"spread_ok:{spread_pips:.1f}_pips[{cls}]"

    def check_volatility(
        self,
        atr_pips: float,
        symbol: str = "",
        point: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Reject trades in dead / low-volatility markets.

        ATR thresholds are defined in PRICE terms (asset's native currency unit)
        to be correct across all asset classes without magic multipliers.

        If point is provided, atr_pips is back-converted to price for comparison:
            atr_price = atr_pips * (point * 10)

        If point is 0, falls back to the legacy pip-based check (min_atr_pips).
        """
        cls = _asset_class(symbol)

        if point > 0:
            # Convert pips → price → compare against price-based floor
            pip_size  = point * 10
            atr_price = atr_pips * pip_size
            threshold_price = ATR_MIN_PRICE.get(cls, ATR_MIN_PRICE["default"])

            if atr_price < threshold_price:
                return False, (
                    f"low_volatility:ATR_price={atr_price:.5f}"
                    f"<{threshold_price:.5f}[{cls}:{symbol}]"
                )
            return True, (
                f"volatility_ok:ATR_price={atr_price:.5f}"
                f">={threshold_price:.5f}[{cls}]"
            )
        else:
            # Legacy fallback: pip-based comparison (used when point not provided)
            if atr_pips < self.min_atr_pips:
                return False, (
                    f"low_volatility:ATR={atr_pips:.2f}"
                    f"<{self.min_atr_pips:.2f}_pips[{cls}]"
                )
            return True, f"volatility_ok:ATR={atr_pips:.2f}_pips[{cls}]"

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def compute_spread_pips(
        symbol: str,
        ask: float,
        bid: float,
        point: float = 0.00001,
    ) -> float:
        """
        Convert ask/bid price spread to pips.
        spread_price / (point * 10) = spread_pips
        Works for all asset classes.
        """
        spread_price = ask - bid
        pip_size = point * 10
        if pip_size <= 0:
            return 0.0
        return round(spread_price / pip_size, 2)

    @staticmethod
    def compute_atr_pips(atr_price: float, point: float = 0.00001) -> float:
        """
        Convert ATR in price terms to pips.
        atr_price / (point * 10) = atr_pips
        Works for all asset classes.
        """
        pip_size = point * 10
        if pip_size <= 0:
            return 0.0
        return round(atr_price / pip_size, 2)
