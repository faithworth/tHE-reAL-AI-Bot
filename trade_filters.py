"""
trade_filters.py
----------------
Multi-layer trade filter pipeline.  v20-PROFIT

Filters applied (in order):
  1. Session filter   — symbol-aware: London/NY for forex+metals; 24h for crypto/indices
  2. Spread filter    — skip if spread > max_spread_pips (asset-class aware)
  3. Volatility filter — skip if ATR too low (asset-class aware, price-based)

Key profitability changes vs v20 original:
  - Asian session (00:00–07:00 UTC) BLOCKED for forex and metals.
    Statistically these instruments chop in Asian hours; spreads are wider
    relative to ATR and false breakouts dominate.
  - Crypto (BTC/ETH) and indices (US30/US500/US100) remain 24h — they have
    genuine volatility and volume in Asian hours.
  - is_premium_session() helper lets caller raise score threshold off-hours.
  - Spread limits tightened for metals/indices to reduce cost-drag.

All filters return (passed: bool, reason: str).
"""

import logging
import warnings
from datetime import datetime, time, timezone
from typing import Tuple, Optional
import pytz

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)


# ── Session definitions (UTC) ─────────────────────────────────────────────────
SESSIONS = {
    "pre_london": {"start": time(6,  0), "end": time(7,  59)},  # 06:00–07:59 UTC warm-up
    "london":     {"start": time(7,  0), "end": time(16,  0)},  # 07:00–16:00 UTC PRIMARY
    "new_york":   {"start": time(13, 0), "end": time(21,  0)},  # 13:00–21:00 UTC PRIMARY
    "overlap":    {"start": time(13, 0), "end": time(16,  0)},  # London/NY overlap — BEST
    "asian":      {"start": time(0,  0), "end": time(6,  59)},  # 00:00–06:59 UTC — forex/metals BLOCKED
    "always":     {"start": time(0,  0), "end": time(23, 59)},  # 24h instruments
}

# Premium = London open through NY close — tightest spreads, most volume
PREMIUM_SESSION_START = time(7,  0)   # London open UTC
PREMIUM_SESSION_END   = time(21, 0)   # NY close UTC

# Asset classes allowed in Asian session (24h instruments)
ASIAN_ALLOWED_CLASSES = {"crypto", "index"}

# ── Spread limits (pips) per asset class ──────────────────────────────────────
SPREAD_LIMITS_PIPS = {
    "forex":   25.0,    # tightened from 30 — reject wide forex spreads
    "metal":   60.0,    # tightened from 80 — gold spread > $6 is problematic
    "index":   50.0,    # kept
    "energy":  60.0,
    "crypto":  400.0,   # tightened from 500 — reject extreme crypto spreads
    "stock":   50.0,
    "default": 40.0,
}

# ── ATR minimums in PRICE terms (NOT pips) ────────────────────────────────────
ATR_MIN_PRICE = {
    "forex":   0.0003,
    "metal":   0.50,    # raised from 0.30 — require genuine gold volatility
    "index":   1.00,    # raised from 0.50 — require real index movement
    "energy":  0.05,
    "crypto":  30.0,    # raised from 20 — require genuine crypto move
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


def is_premium_session(utc_now: Optional[datetime] = None) -> bool:
    """Return True if current UTC time is within London + NY session window.

    Callers can use this to raise score/probability thresholds during off-hours
    rather than blocking entirely.
    """
    if utc_now is None:
        utc_now = datetime.now(timezone.utc)
    t = utc_now.time()
    return PREMIUM_SESSION_START <= t <= PREMIUM_SESSION_END


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
                       NOTE: 'asian' is only enforced for forex/metals — crypto/indices
                       are always allowed regardless of this setting.
    require_session  : If False, skip session filter entirely.
    """

    def __init__(
        self,
        max_spread_pips: float = 25.0,
        min_atr_pips: float = 5.0,
        allowed_sessions: tuple = ("london", "new_york", "pre_london"),
        require_session: bool = True,
    ):
        self.max_spread_pips  = max_spread_pips
        self.min_atr_pips     = min_atr_pips
        self.allowed_sessions = allowed_sessions
        self.require_session  = require_session

    # ── Aggregate check ───────────────────────────────────────────────────────

    def check_all(
        self,
        spread_pips: float,
        atr_pips: Optional[float],
        symbol: str = "",
        utc_now: Optional[datetime] = None,
        point: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Run all filters.  Returns (passed, reason).
        If any filter fails returns (False, <reason>).
        """
        ok, reason = self.check_session(utc_now, symbol=symbol)
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

    def check_session(
        self,
        utc_now: Optional[datetime] = None,
        symbol: str = "",
    ) -> Tuple[bool, str]:
        """Symbol-aware session filter.

        - Crypto and indices: always allowed (24h instruments with genuine volume).
        - Forex and metals: blocked during Asian session (00:00–07:00 UTC) to avoid
          choppy, spread-heavy, low-momentum conditions.
        - All instruments: allowed during London + NY sessions.
        """
        if not self.require_session:
            return True, "session_filter_disabled"

        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        current_time = utc_now.time()
        asset_cls = _asset_class(symbol)

        # 24h instruments — never blocked by session
        if asset_cls in ASIAN_ALLOWED_CLASSES:
            return True, f"session_ok:{asset_cls}_24h_instrument"

        # For forex/metals: check against allowed sessions only
        # Asian session is intentionally NOT in the default allowed_sessions
        for session_name in self.allowed_sessions:
            sess = SESSIONS.get(session_name)
            if sess is None:
                continue
            start, end = sess["start"], sess["end"]
            if start <= end:
                in_sess = start <= current_time <= end
            else:
                in_sess = current_time >= start or current_time <= end
            if in_sess:
                return True, f"in_{session_name}_session"

        # Explicitly log what window we're in for diagnostics
        if SESSIONS["asian"]["start"] <= current_time <= SESSIONS["asian"]["end"]:
            return False, f"asian_session_blocked_for_{asset_cls}:{symbol} (00:00-07:00 UTC)"
        return False, f"outside_active_sessions_{list(self.allowed_sessions)}:{symbol}"

    def check_spread(self, spread_pips: float, symbol: str = "") -> Tuple[bool, str]:
        """Reject trades when spread is too wide."""
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
        """Reject trades in dead / low-volatility markets."""
        cls = _asset_class(symbol)
        if point > 0:
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
        spread_price = ask - bid
        pip_size = point * 10
        if pip_size <= 0:
            return 0.0
        return round(spread_price / pip_size, 2)

    @staticmethod
    def compute_atr_pips(atr_price: float, point: float = 0.00001) -> float:
        pip_size = point * 10
        if pip_size <= 0:
            return 0.0
        return round(atr_price / pip_size, 2)
