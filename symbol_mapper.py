"""
symbol_mapper.py — Dynamic Symbol Translation Layer (AI EA v5)
--------------------------------------------------------------
Responsibilities:
  - Strip broker-specific suffixes/prefixes from raw symbol names
    (e.g.  XAUUSDm → XAUUSD,  EURUSD.pro → EURUSD,  #AAPL → AAPL)
  - Apply the reverse mapping when submitting orders back to the broker
  - Zero manual configuration — rules are inferred from live symbol list
  - Works with ANY broker adapter (MT5, IBKR, cTrader) via BaseBroker
  - Unknown / untranslatable symbols pass through unchanged (safe fallback)

Usage
-----
from symbol_mapper import SymbolMapper
mapper = SymbolMapper(broker=my_broker)   # pass any BaseBroker instance
clean  = mapper.to_clean("XAUUSDm")       # → "XAUUSD"
broker = mapper.to_broker("XAUUSD")       # → "XAUUSDm"  (reverse)

Legacy MT5 mode still works:
    mapper = SymbolMapper(mt5_module=mt5)
"""

import logging
import re
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known canonical base symbols — used to validate strip results
# ---------------------------------------------------------------------------
_KNOWN_BASES: Set[str] = {
    # Forex majors / minors / exotics
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "EURAUD",
    "EURCAD", "EURCHF", "EURNZD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
    "AUDCAD", "AUDCHF", "AUDNZD", "NZDCAD", "NZDCHF", "NZDJPY",
    "USDSGD", "USDHKD", "USDMXN", "USDZAR", "USDNOK", "USDSEK", "USDDKK",
    "USDTRY", "USDHUF", "USDPLN", "USDCZK",
    # Metals
    "XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD",
    # Energies
    "USOIL", "UKOIL", "XBRUSD", "XTIUSD", "NATGAS",
    # Indices
    "US30", "US500", "US100", "UK100", "GER40", "GER30", "FRA40",
    "JPN225", "AUS200", "HKG50", "ESP35", "STOXX50",
    "SPX500", "NDX100", "DJI30", "FTSE100", "DAX40",
    # Crypto
    "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD", "BNBUSD", "ADAUSD",
    "BTCUSDT", "ETHUSDT",
}

# Regex patterns for broker decorators, ordered most-specific first
# Each entry: (compiled_pattern, description)
_STRIP_PATTERNS: List[re.Pattern] = [
    re.compile(r"^#"),                                               # CFD prefix   #AAPL
    re.compile(r"\.{2,}$"),                                          # double/triple dot suffix  XAUUSD..
    re.compile(r"\.(pro|ecn|raw|PRO|ECN|RAW|std|STD|stp|STP)$"),   # suffix .pro
    re.compile(r"[._\-](mini|MINI|micro|MICRO|nano|NANO)$"),        # size suffix
    re.compile(r"[._\-]+$"),                                         # trailing dots/underscores
    re.compile(r"[a-z]+$"),                                          # lowercase trailing chars XAUUSDm, EURUSDr
    re.compile(r"\d+$"),                                             # numeric trailing  EURUSD1
    re.compile(r"^[a-z]+"),                                          # lowercase leading  fxEURUSD
    re.compile(r"^fx_", re.IGNORECASE),                              # fx_ prefix
    re.compile(r"^(spot|cfg|cfx|cfr)_?", re.IGNORECASE),            # spot/cfg prefixes (cTrader)
]


class SymbolMapper:
    """
    Bidirectional symbol translator.
    Build once at startup; refresh if the broker symbol list changes.

    Accepts either a BaseBroker instance (preferred, broker-agnostic)
    or the legacy MT5 module (backwards compatibility).
    """

    def __init__(self, broker=None, mt5_module=None):
        """
        Parameters
        ----------
        broker     : BaseBroker subclass instance (preferred)
                     If supplied, get_symbols() is called to build the table.
        mt5_module : MetaTrader5 module (legacy — used if broker is None)
        """
        self._broker     = broker
        self._mt5        = mt5_module
        # broker_sym → clean_sym
        self._to_clean: Dict[str, str] = {}
        # clean_sym → broker_sym  (first match wins for multi-suffix brokers)
        self._to_broker: Dict[str, str] = {}

        self._build_table()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_clean(self, broker_symbol: str) -> str:
        """
        Translate broker-specific symbol to canonical form.
        Returns the original symbol unchanged if no mapping found.
        """
        if broker_symbol in self._to_clean:
            return self._to_clean[broker_symbol]
        # Fallback: try pattern strip on-the-fly
        clean = self._strip_decorators(broker_symbol)
        if clean and clean != broker_symbol:
            logger.debug(f"SymbolMapper on-the-fly: {broker_symbol} → {clean}")
            # Cache for next time
            self._to_clean[broker_symbol] = clean
            if clean not in self._to_broker:
                self._to_broker[clean] = broker_symbol
        return self._to_clean.get(broker_symbol, broker_symbol)

    def to_broker(self, clean_symbol: str) -> str:
        """
        Reverse: canonical symbol → broker-specific form.
        Returns clean_symbol unchanged if no broker mapping exists.
        """
        return self._to_broker.get(clean_symbol, clean_symbol)

    def refresh(self, broker=None) -> None:
        """Re-scan broker symbols (call after reconnect or symbol list change)."""
        if broker is not None:
            self._broker = broker
        self._to_clean.clear()
        self._to_broker.clear()
        self._build_table()
        logger.info(f"SymbolMapper refreshed — {len(self._to_clean)} mappings built")

    def get_all_mappings(self) -> Dict[str, str]:
        """Return a copy of the broker→clean mapping table."""
        return dict(self._to_clean)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_table(self) -> None:
        """Build translation table from live broker symbol list."""
        broker_symbols: List[str] = []

        # Prefer BaseBroker interface (works with MT5, IBKR, cTrader)
        if self._broker is not None:
            try:
                sym_dicts = self._broker.get_symbols()
                if sym_dicts:
                    broker_symbols = [s["name"] for s in sym_dicts]
            except Exception as e:
                logger.warning(f"SymbolMapper: could not fetch symbols from broker: {e}")

        # Legacy MT5 module fallback
        if not broker_symbols and self._mt5 is not None:
            try:
                raw = self._mt5.symbols_get()
                if raw:
                    broker_symbols = [s.name for s in raw]
            except Exception as e:
                logger.warning(f"SymbolMapper: could not fetch symbols from MT5: {e}")

        if not broker_symbols:
            logger.debug("SymbolMapper: no live symbol list — heuristic-only mode")
            return

        mapped = 0
        for sym in broker_symbols:
            clean = self._strip_decorators(sym)
            if clean and clean != sym:
                self._to_clean[sym] = clean
                if clean not in self._to_broker:
                    self._to_broker[clean] = sym
                mapped += 1
            else:
                # Symbol is already clean — map to itself
                self._to_clean[sym] = sym
                if sym not in self._to_broker:
                    self._to_broker[sym] = sym

        broker_name = getattr(self._broker, "broker_name", "MT5") if self._broker else "MT5"
        logger.info(
            f"SymbolMapper: {mapped}/{len(broker_symbols)} symbols translated "
            f"| broker={broker_name}"
        )

    def _strip_decorators(self, symbol: str) -> str:
        """
        Apply stripping rules iteratively until the result stabilises
        or matches a known base.  Returns the cleaned symbol or the
        original if stripping would produce an empty string.
        """
        s = symbol
        for _ in range(4):           # max 4 passes — prevents infinite loops
            candidate = s
            for pat in _STRIP_PATTERNS:
                candidate = pat.sub("", candidate)
            candidate = candidate.strip("._-")
            if not candidate:
                return symbol        # never return empty — safe fallback
            if candidate == s:
                break                # nothing more to strip
            if candidate.upper() in _KNOWN_BASES:
                return candidate.upper()
            s = candidate

        result = s.upper()
        return result if len(result) >= 4 else symbol   # sanity: min 4 chars

    def __repr__(self) -> str:
        return (
            f"<SymbolMapper mappings={len(self._to_clean)} "
            f"reverse={len(self._to_broker)}>"
        )
