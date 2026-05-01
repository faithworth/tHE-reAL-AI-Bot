"""
news_filter.py — Economic Calendar News Filter (AI EA v6 PRO)
=============================================================
Fetches high-impact economic events and blocks trading 30 min
before / 30 min after to avoid getting hit by NFP, FOMC, CPI etc.

Sources (in order of preference):
  1. ForexFactory calendar JSON (free, no API key)
  2. Fallback: hardcoded known recurring events as static blackout

Usage:
    nf = NewsFilter()
    blocked, reason = nf.is_blocked(symbol="EURUSD")
    if blocked:
        logger.warning(f"News blackout: {reason}")
"""

import logging
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)

# Blackout window in minutes
BLACKOUT_BEFORE_MINS = 30
BLACKOUT_AFTER_MINS  = 30

# Cache file to avoid hammering the API every tick
CACHE_FILE    = "data/news_cache.json"
CACHE_TTL_MIN = 60   # refresh calendar once per hour

# High-impact currencies per symbol class
CURRENCY_MAP = {
    "EURUSD": ["EUR", "USD"], "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"], "USDCHF": ["USD", "CHF"],
    "AUDUSD": ["AUD", "USD"], "NZDUSD": ["NZD", "USD"],
    "USDCAD": ["USD", "CAD"], "XAUUSD": ["USD"],
    "XAGUSD": ["USD"],        "BTCUSD": ["USD"],
    "ETHUSD": ["USD"],        "USOIL":  ["USD"],
}

# Static recurring high-impact events (day-of-week + UTC hour windows)
# Used as fallback when live calendar is unavailable
# Static recurring high-impact events — used ONLY when live calendar is unavailable.
# NOTE: "US CPI every Tuesday" has been intentionally removed — CPI does not fall
# on a fixed weekday and caused false positives every Tuesday even without CPI.
# The live ForexFactory calendar handles CPI correctly when network is available.
STATIC_BLACKOUT_WINDOWS = [
    # NFP first Friday of month 12:30 UTC — we only fire when day <= 7
    {"dow": 4, "hour_start": 12, "hour_end": 14, "desc": "Potential NFP (first Friday)"},
    # FOMC minutes - Wednesdays 17:00-20:00 UTC
    {"dow": 2, "hour_start": 17, "hour_end": 20, "desc": "Potential FOMC"},
]


def win_hour(start: int, end: int, hr: int) -> bool:
    """Return True if hr is within [start, end)."""
    return start <= hr < end


class NewsFilter:
    """
    Blocks trading around high-impact economic news events.
    """

    def __init__(
        self,
        blackout_before_mins: int = BLACKOUT_BEFORE_MINS,
        blackout_after_mins:  int = BLACKOUT_AFTER_MINS,
        impact_levels: tuple  = ("high",),
        use_live_calendar: bool = True,
    ):
        self.blackout_before = timedelta(minutes=blackout_before_mins)
        self.blackout_after  = timedelta(minutes=blackout_after_mins)
        self.impact_levels   = [i.lower() for i in impact_levels]
        self.use_live         = use_live_calendar
        self._cached_events: List[Dict] = []
        self._cache_loaded_at: Optional[datetime] = None
        os.makedirs("data", exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────

    def is_blocked(
        self,
        symbol: str = "",
        utc_now: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """
        Returns (blocked: bool, reason: str).
        blocked=True means do NOT trade right now.

        Logic:
          1. If live calendar loaded successfully → use ONLY live events.
             Static windows are NOT checked — they are a last-resort fallback
             for when the network is unavailable, not a supplement to live data.
          2. If live calendar failed or is disabled → fall back to static windows.
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc).replace(tzinfo=None)

        if self.use_live:
            try:
                self._refresh_cache_if_needed()
                # If we have a populated live cache, trust it completely.
                # Do NOT also run static windows — that causes false positives
                # (e.g. "Potential US CPI every Tuesday 12-14 UTC" firing even
                # when the live calendar confirms no CPI today).
                if self._cached_events is not None and self._cache_loaded_at is not None:
                    return self._check_live_events(symbol, utc_now)
            except Exception as e:
                logger.debug(f"Live news check failed: {e} -- using static fallback")

        # Static blackout fallback — only reached when live calendar is
        # unavailable (network error, first boot before any fetch, etc.)
        return self._check_static_windows(utc_now)

    def get_upcoming_events(self, hours_ahead: int = 24) -> List[Dict]:
        """Return list of upcoming high-impact events within the next N hours."""
        self._refresh_cache_if_needed()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cutoff = now + timedelta(hours=hours_ahead)
        upcoming = []
        for ev in self._cached_events:
            ev_time = ev.get("_dt")
            if ev_time and now <= ev_time <= cutoff:
                upcoming.append(ev)
        return upcoming

    # ── Internal ─────────────────────────────────────────────────────

    def _refresh_cache_if_needed(self) -> None:
        """Reload calendar from cache file or network if stale."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if (self._cache_loaded_at is not None and
                (now - self._cache_loaded_at).total_seconds() < CACHE_TTL_MIN * 60):
            return  # still fresh

        # Try disk cache first
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE) as f:
                    data = json.load(f)
                cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
                age_mins  = (now - cached_at).total_seconds() / 60
                if age_mins < CACHE_TTL_MIN:
                    self._cached_events   = self._parse_events(data.get("events", []))
                    self._cache_loaded_at = now
                    logger.debug(f"News cache: loaded {len(self._cached_events)} events from disk")
                    return
            except Exception:
                pass

        # Fetch from ForexFactory
        self._fetch_forexfactory()

    def _fetch_forexfactory(self) -> None:
        """Fetch this week's calendar from ForexFactory JSON API."""
        try:
            import urllib.request
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw = json.loads(resp.read().decode())
            events = self._parse_ff_events(raw)
            self._cached_events   = events
            self._cache_loaded_at = datetime.now(timezone.utc).replace(tzinfo=None)
            # Write to disk
            with open(CACHE_FILE, "w") as f:
                json.dump({
                    "cached_at": self._cache_loaded_at.isoformat(),
                    "events": [self._event_to_dict(e) for e in events],
                }, f, indent=2)
            logger.info(f"News filter: fetched {len(events)} events from ForexFactory")
        except Exception as e:
            logger.debug(f"ForexFactory fetch failed: {e}")
            self._cached_events   = []
            self._cache_loaded_at = datetime.now(timezone.utc).replace(tzinfo=None)

    def _parse_ff_events(self, raw: list) -> List[Dict]:
        """Parse ForexFactory JSON into internal event dicts."""
        events = []
        for item in raw:
            impact = str(item.get("impact", "")).lower()
            if impact not in self.impact_levels:
                continue
            try:
                dt_str = item.get("date", "")
                if not dt_str:
                    continue
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                dt = dt.replace(tzinfo=None)  # work in naive UTC
                events.append({
                    "_dt":     dt,
                    "title":   item.get("title", "Unknown"),
                    "country": item.get("country", ""),
                    "impact":  impact,
                })
            except Exception:
                continue
        return events

    def _parse_events(self, raw_list: list) -> List[Dict]:
        """Parse events loaded from disk cache."""
        events = []
        for item in raw_list:
            try:
                dt = datetime.fromisoformat(item["_dt"])
                item["_dt"] = dt
                events.append(item)
            except Exception:
                continue
        return events

    def _event_to_dict(self, ev: Dict) -> Dict:
        return {**ev, "_dt": ev["_dt"].isoformat()}

    def _check_live_events(
        self, symbol: str, utc_now: datetime
    ) -> Tuple[bool, str]:
        """Check if any live event falls within the blackout window."""
        currencies = self._get_currencies(symbol)
        for ev in self._cached_events:
            ev_time = ev.get("_dt")
            if not ev_time:
                continue
            country = ev.get("country", "").upper()
            # Check if event is relevant to this symbol's currencies
            relevant = (not currencies) or any(
                c[:2].upper() == country[:2].upper() or
                country.upper() in c.upper()
                for c in currencies
            )
            if not relevant:
                continue
            window_start = ev_time - self.blackout_before
            window_end   = ev_time + self.blackout_after
            if window_start <= utc_now <= window_end:
                offset = (utc_now - ev_time).total_seconds() / 60
                phase  = "before" if utc_now < ev_time else "after"
                reason = (f"NEWS_BLACKOUT: {ev.get('title','?')} ({country}) "
                          f"{abs(offset):.0f}min {phase}")
                return True, reason
        return False, "no_news_conflict"

    def _check_static_windows(self, utc_now: datetime) -> Tuple[bool, str]:
        """
        Static fallback: block known high-impact time windows.
        Only used when the live calendar is unavailable.

        NFP fires on the FIRST Friday of each month — not every Friday.
        The "US CPI every Tuesday" rule is removed because CPI does NOT fall
        on a fixed weekday and causes false positives on non-CPI Tuesdays.
        FOMC minutes are kept (Wednesdays 18-20 UTC every ~6 weeks) as a
        conservative block — low false-positive risk since it only fires
        on Wednesdays in a 2-hour window.
        """
        dow = utc_now.weekday()
        hr  = utc_now.hour

        # NFP: first Friday of month only
        if dow == 4 and win_hour(12, 14, hr):
            # First Friday = day <= 7
            if utc_now.day <= 7:
                return True, "STATIC_BLACKOUT: Potential NFP (first Friday) UTC 12-14"

        # FOMC minutes: Wednesdays 18-20 UTC
        if dow == 2 and win_hour(17, 20, hr):
            return True, "STATIC_BLACKOUT: Potential FOMC UTC 17-20"

        return False, "no_news_conflict"

    @staticmethod
    def _get_currencies(symbol: str) -> List[str]:
        """Get relevant currency codes for a symbol."""
        if not symbol:
            return []
        sym_upper = symbol.upper().strip("._-#")
        # Direct lookup
        if sym_upper in CURRENCY_MAP:
            return CURRENCY_MAP[sym_upper]
        # Try forex pair inference (e.g. EURUSD -> EUR, USD)
        if len(sym_upper) >= 6:
            return [sym_upper[:3], sym_upper[3:6]]
        return []
