"""
symbol_discovery.py — Symbol Auto-Discovery (AI EA v5)
-------------------------------------------------------
Responsibilities:
  - Dynamically retrieve all tradable instruments from the broker
  - Filter to forex, metals, indices, crypto (skip corrupted/empty)
  - Cache results with configurable TTL
  - Works with ANY broker adapter (MT5, IBKR, cTrader) via BaseBroker
  - Never rely on static symbol lists

Usage
-----
from symbol_discovery import SymbolDiscovery
disco = SymbolDiscovery(broker=my_broker)  # pass any BaseBroker instance
symbols = disco.get_tradable()             # → ["EURUSD", "XAUUSD", ...]

Legacy MT5 mode still works:
    disco = SymbolDiscovery(mt5_module=mt5)
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

CACHE_PATH     = "data/symbol_cache.json"
CACHE_TTL_SECS = 3_600          # 1 hour — refresh once per session
MIN_TICK_COUNT = 1              # symbol must have at least 1 recent tick
MAX_SPREAD_MULTIPLIER = 50      # skip symbols with absurd spread/point ratio

# ---------------------------------------------------------------------------
# Asset-class classification rules
# ---------------------------------------------------------------------------

# Each entry: (keywords_in_name, asset_class_label)
_CLASS_RULES: List[tuple] = [
    # Metals
    ({"XAU", "GOLD"},                          "metal"),
    ({"XAG", "SILVER"},                        "metal"),
    ({"XPT", "XPD"},                           "metal"),
    # Crypto
    ({"BTC", "ETH", "LTC", "XRP", "BNB", "ADA", "SOL", "DOT", "DOGE"}, "crypto"),
    # Energies
    ({"OIL", "BRENT", "WTI", "NATGAS", "GAS", "XBR", "XTI"},           "energy"),
    # Indices
    ({"US30", "US500", "US100", "UK100", "GER", "DAX", "FRA", "JPN",
      "AUS200", "HKG", "ESP", "STOXX", "SPX", "NDX", "DJI", "FTSE",
      "CAC", "NIKKEI", "NAS", "DOW"},                                   "index"),
    # Forex (6-char pairs of 3-char currency codes)
    (set(),                                                               "forex"),  # fallback
]

# Currency codes used to identify forex pairs
_CURRENCY_CODES: Set[str] = {
    "AUD", "CAD", "CHF", "CNH", "CZK", "DKK", "EUR", "GBP", "HKD",
    "HUF", "JPY", "MXN", "NOK", "NZD", "PLN", "SEK", "SGD", "TRY",
    "USD", "ZAR",
}

# Asset classes to include in discovery output
_INCLUDE_CLASSES: Set[str] = {"forex", "metal", "index", "crypto", "energy"}


def classify_symbol(name: str) -> str:
    """Return asset class string for a given symbol name."""
    upper = name.upper().strip("._-#")

    for keywords, cls in _CLASS_RULES[:-1]:   # skip the forex fallback
        if any(k in upper for k in keywords):
            return cls

    # Forex heuristic: 6-char string composed of two known 3-char codes
    clean = upper.rstrip("0123456789").strip("._-")
    if len(clean) in (6, 7, 8):
        for i in range(3, min(5, len(clean) - 2)):
            base  = clean[:i]
            quote = clean[i:i+3]
            if base in _CURRENCY_CODES and quote in _CURRENCY_CODES:
                return "forex"

    return "unknown"


class SymbolDiscovery:
    """
    Discovers and caches tradable symbols from the connected broker.
    Supports MT5, IBKR, cTrader via BaseBroker interface.
    """

    def __init__(
        self,
        broker=None,
        mt5_module=None,        # legacy — kept for backwards compat
        include_classes: Optional[Set[str]] = None,
        cache_ttl: int = CACHE_TTL_SECS,
        cache_path: str = CACHE_PATH,
    ):
        """
        Parameters
        ----------
        broker     : BaseBroker subclass instance (preferred)
        mt5_module : MetaTrader5 module (legacy fallback)
        """
        self._broker        = broker
        self._mt5           = mt5_module
        self._include       = include_classes or _INCLUDE_CLASSES
        self._cache_ttl     = cache_ttl
        self._cache_path    = cache_path
        self._cached_symbols: List[str] = []
        self._cache_ts: float = 0.0
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tradable(self, force_refresh: bool = False) -> List[str]:
        """
        Return a list of clean, tradable symbol names.
        Uses cache if within TTL; otherwise re-discovers from broker.
        """
        if not force_refresh and self._is_cache_valid():
            return list(self._cached_symbols)

        # Try disk cache before hitting the broker
        if not force_refresh:
            disk = self._load_disk_cache()
            if disk:
                self._cached_symbols = disk
                logger.info(f"SymbolDiscovery: loaded {len(disk)} symbols from disk cache")
                return list(disk)

        symbols = self._discover()
        if symbols:
            self._cached_symbols = symbols
            self._cache_ts       = time.time()
            self._save_disk_cache(symbols)
            logger.info(f"SymbolDiscovery: {len(symbols)} tradable symbols found")
        else:
            logger.warning("SymbolDiscovery: discovery returned no symbols — check connection")

        return list(self._cached_symbols)

    def get_by_class(self, asset_class: str) -> List[str]:
        """Return only symbols belonging to the given asset class."""
        all_syms = self.get_tradable()
        return [s for s in all_syms if classify_symbol(s) == asset_class]

    def invalidate(self) -> None:
        """Force next call to re-discover from broker."""
        self._cache_ts = 0.0
        try:
            if os.path.exists(self._cache_path):
                os.remove(self._cache_path)
        except Exception:
            pass

    def classify(self, symbol: str) -> str:
        return classify_symbol(symbol)

    # ------------------------------------------------------------------
    # Discovery core
    # ------------------------------------------------------------------

    def _discover(self) -> List[str]:
        """Pull symbols from broker adapter, filter, and return clean list."""

        # Prefer BaseBroker interface (MT5, IBKR, cTrader)
        if self._broker is not None:
            return self._discover_via_broker()

        # Legacy: MT5 module
        if self._mt5 is not None:
            return self._discover_via_mt5()

        logger.warning("SymbolDiscovery: no broker or MT5 module — cannot discover")
        return []

    def _discover_via_broker(self) -> List[str]:
        """Discover symbols via BaseBroker.get_symbols() — broker-agnostic."""
        try:
            raw_symbols = self._broker.get_symbols()
        except Exception as e:
            logger.error(f"SymbolDiscovery: broker.get_symbols() error: {e}")
            return []

        if not raw_symbols:
            return []

        tradable: List[str] = []
        for sym in raw_symbols:
            try:
                name      = sym.get("name", "")
                trade_mode = sym.get("trade_mode", 1)
                if not name:
                    continue
                if trade_mode == 0:   # disabled
                    continue
                cls = classify_symbol(name)
                if cls not in self._include:
                    continue
                tradable.append(name)
            except Exception:
                continue

        broker_name = getattr(self._broker, "broker_name", "?")
        logger.info(
            f"SymbolDiscovery [{broker_name}]: "
            f"{len(tradable)}/{len(raw_symbols)} tradable symbols"
        )
        return sorted(tradable)

    def _discover_via_mt5(self) -> List[str]:
        """Legacy MT5 discovery path."""
        try:
            raw_symbols = self._mt5.symbols_get()
        except Exception as e:
            logger.error(f"SymbolDiscovery: mt5.symbols_get() error: {e}")
            return []

        if not raw_symbols:
            return []

        tradable: List[str] = []
        for sym in raw_symbols:
            try:
                if not self._is_tradable_mt5(sym):
                    continue
                cls = classify_symbol(sym.name)
                if cls not in self._include:
                    continue
                tradable.append(sym.name)
            except Exception:
                continue

        return sorted(tradable)

    def _is_tradable_mt5(self, sym) -> bool:
        """Return True if MT5 symbol is suitable for live trading."""
        try:
            if not sym.visible:
                if self._mt5 and not self._mt5.symbol_select(sym.name, True):
                    return False
            if hasattr(sym, "trade_mode") and sym.trade_mode == 0:
                return False
            if not getattr(sym, "trade_contract_size", 0):
                return False
            point  = getattr(sym, "point", 0)
            spread = getattr(sym, "spread", 0)
            if point > 0 and spread > 0:
                if spread > MAX_SPREAD_MULTIPLIER * 10:
                    return False
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Disk cache helpers
    # ------------------------------------------------------------------

    def _is_cache_valid(self) -> bool:
        return (
            bool(self._cached_symbols)
            and (time.time() - self._cache_ts) < self._cache_ttl
        )

    def _save_disk_cache(self, symbols: List[str]) -> None:
        try:
            payload = {"timestamp": time.time(), "symbols": symbols}
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception as e:
            logger.warning(f"SymbolDiscovery: disk cache write failed: {e}")

    def _load_disk_cache(self) -> List[str]:
        if not os.path.exists(self._cache_path):
            return []
        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            age = time.time() - payload.get("timestamp", 0)
            if age < self._cache_ttl:
                return payload.get("symbols", [])
        except Exception as e:
            logger.warning(f"SymbolDiscovery: disk cache read failed: {e}")
        return []
