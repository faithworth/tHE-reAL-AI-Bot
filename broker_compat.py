"""
broker_compat.py — Broker Compatibility Layer (AI EA v4)
---------------------------------------------------------
Responsibilities:
  - Detect broker type/style automatically (no manual config)
  - Infer formatting rules from live account metadata and symbol patterns
  - Expose a unified interface so the rest of the system never cares
    which broker is connected

Supported broker detection heuristics
  - Suffix pattern frequency  (m / .raw / .pro / .ecn / r / …)
  - Account currency and leverage
  - Symbol count and naming conventions
  - Server name keyword matching (known broker server substrings)

Usage
-----
from broker_compat import BrokerProfile, detect_broker
profile = detect_broker(mt5_module)   # call once after MT5 connect
print(profile.name, profile.suffix)   # e.g. "Exness", "m"
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known broker server keyword → friendly name
# ---------------------------------------------------------------------------
_SERVER_MAP: Dict[str, str] = {
    "exness":    "Exness",
    "icmarket":  "IC_Markets",
    "icm":       "IC_Markets",
    "fxgt":      "FXGT",
    "fxpro":     "FXPro",
    "pepperstone": "Pepperstone",
    "axitrader": "Axitrader",
    "xm":        "XM",
    "roboforex": "RoboForex",
    "hotforex":  "HFM",
    "hfmarket":  "HFM",
    "admiralmarket": "Admiral",
    "octafx":    "OctaFX",
    "fbs":       "FBS",
    "tickmill":  "Tickmill",
    "liteforex": "LiteForex",
    "justmarkets": "JustMarkets",
    "vantage":   "Vantage",
    "blackbull": "BlackBull",
    "fusion":    "FusionMarkets",
    "global prime": "GlobalPrime",
}

# Regex to extract trailing suffix pattern from a symbol
_SUFFIX_RE = re.compile(r"^[A-Z0-9#]+([._\-]?[a-zA-Z]+\d*|\d+)$")


@dataclass
class BrokerProfile:
    """
    Describes the connected broker's formatting conventions.
    All fields are auto-detected — never manually configured.
    """
    name: str = "Unknown"
    server: str = ""
    suffix: str = ""               # most common trailing decorator, e.g. "m"
    prefix: str = ""               # leading decorator, e.g. "#"
    uses_suffix: bool = False
    uses_prefix: bool = False
    account_currency: str = "USD"
    leverage: int = 0
    symbol_count: int = 0
    # Contract size overrides detected from live symbol info
    contract_sizes: Dict[str, float] = field(default_factory=dict)
    # Point / digit overrides
    point_values: Dict[str, float] = field(default_factory=dict)
    # Filling mode (broker-specific)
    # Default to RETURN (2) — most MT5 brokers, especially prop-firm demos, use
    # RETURN mode.  IOC (1) is the minority; the detector will override if needed.
    filling_mode: int = 2          # 2=RETURN default; overridden by _detect_filling_mode

    def describe(self) -> str:
        parts = [f"Broker={self.name}", f"server={self.server}"]
        if self.uses_suffix:
            parts.append(f"suffix='{self.suffix}'")
        if self.uses_prefix:
            parts.append(f"prefix='{self.prefix}'")
        parts += [
            f"currency={self.account_currency}",
            f"leverage=1:{self.leverage}",
            f"symbols={self.symbol_count}",
        ]
        return " | ".join(parts)


def detect_broker(mt5_module=None) -> BrokerProfile:
    """
    Auto-detect broker profile from live MT5 session data.
    Returns a populated BrokerProfile; works in offline/mock mode too.
    """
    profile = BrokerProfile()

    if mt5_module is None:
        logger.warning("BrokerCompat: no MT5 module — returning default profile")
        return profile

    # ── Account info ──────────────────────────────────────────────────────────
    try:
        acct = mt5_module.account_info()
        if acct:
            profile.server           = acct.server or ""
            profile.account_currency = acct.currency or "USD"
            profile.leverage         = int(acct.leverage) if acct.leverage else 0
            # Server name → broker name
            server_lower = profile.server.lower()
            for keyword, bname in _SERVER_MAP.items():
                if keyword in server_lower:
                    profile.name = bname
                    break
    except Exception as e:
        logger.warning(f"BrokerCompat: could not read account_info: {e}")

    # ── Symbol list analysis ──────────────────────────────────────────────────
    try:
        symbols = mt5_module.symbols_get()
        if symbols:
            profile.symbol_count = len(symbols)
            names = [s.name for s in symbols]
            suffix, prefix = _infer_decorators(names)
            profile.suffix      = suffix
            profile.prefix      = prefix
            profile.uses_suffix = bool(suffix)
            profile.uses_prefix = bool(prefix)

            # ── Per-symbol contract sizes & points ───────────────────────────
            for sym in symbols[:500]:        # sample first 500 to keep startup fast
                try:
                    profile.contract_sizes[sym.name] = float(sym.trade_contract_size)
                    profile.point_values[sym.name]   = float(sym.point)
                except Exception:
                    pass

            # ── Filling mode detection ────────────────────────────────────────
            profile.filling_mode = _detect_filling_mode(mt5_module, names)

    except Exception as e:
        logger.warning(f"BrokerCompat: symbol scan error: {e}")

    logger.info(f"BrokerCompat detected: {profile.describe()}")
    return profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_decorators(symbol_names: List[str]) -> Tuple[str, str]:
    """
    Analyse a list of broker symbol names and return the most common
    (suffix, prefix) pair.  Returns ("", "") if nothing detected.
    """
    suffix_counts: Counter = Counter()
    prefix_counts: Counter = Counter()

    for name in symbol_names:
        # Leading non-alpha-uppercase prefix (e.g. "#")
        if name and not name[0].isupper() and not name[0].isdigit():
            prefix_counts[name[0]] += 1

        # Trailing lowercase or mixed decorator
        m = re.search(r"([._]?[a-z][a-zA-Z0-9]*)$", name)
        if m:
            suffix_counts[m.group(1)] += 1

    # Accept a suffix only if it appears on >15% of symbols
    threshold = max(3, len(symbol_names) * 0.15)
    top_suffix = ""
    if suffix_counts:
        cand, cnt = suffix_counts.most_common(1)[0]
        if cnt >= threshold:
            top_suffix = cand.lstrip("._")

    top_prefix = ""
    if prefix_counts:
        cand, cnt = prefix_counts.most_common(1)[0]
        if cnt >= threshold:
            top_prefix = cand

    return top_suffix, top_prefix


def _detect_filling_mode(mt5_module, symbol_names: List[str]) -> int:
    """
    Determine the broker's supported ORDER_FILLING mode by probing symbol_info.

    CRITICAL: the symbol's ``filling_mode`` field is a BITMASK, not an enum.
    The bits map as follows:
        bit 0  (value 1) → FOK  (Fill or Kill)  is supported
        bit 1  (value 2) → IOC  (Immediate or Cancel) is supported
        bit 2  (value 4) → RETURN (partial fill) is supported

    The ORDER_FILLING constants used in type_filling are *different* values:
        ORDER_FILLING_FOK    = 0
        ORDER_FILLING_IOC    = 1
        ORDER_FILLING_RETURN = 2

    Mixing these up (testing ``fm & ORDER_FILLING_IOC`` i.e. ``fm & 1``) probes
    the FOK-supported bit, not the IOC-supported bit — a silent mapping error.

    Priority: RETURN > IOC > FOK  (RETURN is the most widely accepted on demo
    and prop-firm accounts; pure-FOK brokers are rare).
    """
    try:
        import MetaTrader5 as _mt5

        # ORDER_FILLING constants — these go into type_filling
        ORDER_FOK    = getattr(_mt5, "ORDER_FILLING_FOK",    0)
        ORDER_IOC    = getattr(_mt5, "ORDER_FILLING_IOC",    1)
        ORDER_RETURN = getattr(_mt5, "ORDER_FILLING_RETURN", 2)

        # Bitmask positions inside symbol.filling_mode
        BIT_FOK    = 1   # bit 0
        BIT_IOC    = 2   # bit 1
        BIT_RETURN = 4   # bit 2

        # Probe the first symbol whose name contains a known liquid pair
        probe_bases = ("EURUSD", "XAUUSD", "GBPUSD", "USDJPY", "BTCUSD")
        for sym_name in symbol_names[:300]:
            upper = sym_name.upper()
            if not any(base in upper for base in probe_bases):
                continue
            try:
                info = mt5_module.symbol_info(sym_name)
            except Exception:
                continue
            if info is None or not hasattr(info, "filling_mode"):
                continue
            fm = int(info.filling_mode)
            if fm == 0:
                continue   # no filling info — try next symbol
            # Prefer RETURN, then IOC, then FOK
            if fm & BIT_RETURN:
                return ORDER_RETURN
            if fm & BIT_IOC:
                return ORDER_IOC
            if fm & BIT_FOK:
                return ORDER_FOK
    except Exception:
        pass
    # Default: RETURN is the safest choice — accepted by most MT5 brokers
    # including virtually all prop-firm demo servers (e.g. GTio, FTMO, MyFx).
    return 2   # ORDER_FILLING_RETURN


def get_contract_size(profile: BrokerProfile, symbol: str, default: float = 100_000.0) -> float:
    """Retrieve contract size for a symbol from the broker profile."""
    return profile.contract_sizes.get(symbol, default)


def get_point_value(profile: BrokerProfile, symbol: str, default: float = 0.00001) -> float:
    """Retrieve point value for a symbol from the broker profile."""
    return profile.point_values.get(symbol, default)
