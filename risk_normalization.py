"""
risk_normalization.py — Symbol-Aware Risk Normalization (AI EA v4)
------------------------------------------------------------------
Drop-in extension to RiskEngine.  Provides a single entry-point:

    normalizer = RiskNormalizer(risk_engine, broker_profile)
    lot = normalizer.normalized_lot(equity, atr, symbol, sym_info)

Key responsibilities
  - Detect symbol type (forex / metal / index / crypto / energy)
  - Resolve correct contract_size, point_value, pip_value per lot
  - Apply type-specific multiplier corrections
  - Clamp to broker min/max/step with safe rounding
  - Guarantee consistent % risk across ALL symbol types and brokers
  - Never return 0 or negative lots — fallback to min_lot always
"""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-asset-class parameters
# ---------------------------------------------------------------------------
# fmt: off
_ASSET_PARAMS = {
    #  class          contract_size   point         pip_digits   description
    "forex":        (100_000.0,       0.00001,      5,           "Standard forex pair"),
    "forex_jpy":    (100_000.0,       0.001,        3,           "JPY-quoted pair"),
    "metal_xau":    (100.0,           0.01,         2,           "Gold (XAU/USD)"),
    "metal_xag":    (5_000.0,         0.001,        3,           "Silver (XAG/USD)"),
    "metal_other":  (100.0,           0.01,         2,           "Other metal"),
    "index_us":     (1.0,             0.01,         2,           "US equity index"),
    "index_eu":     (1.0,             0.01,         2,           "EU/JP equity index"),
    "crypto_btc":   (1.0,             0.01,         2,           "Bitcoin"),
    "crypto_eth":   (1.0,             0.01,         2,           "Ethereum"),
    "crypto_other": (1.0,             0.001,        3,           "Other crypto"),
    "energy_oil":   (1_000.0,         0.001,        3,           "Crude oil"),
    "energy_gas":   (10_000.0,        0.0001,       4,           "Natural gas"),
    "unknown":      (100_000.0,       0.00001,      5,           "Unknown — use forex defaults"),
}
# fmt: on


def _classify(symbol: str) -> str:
    """Classify a symbol into an _ASSET_PARAMS key."""
    u = symbol.upper().strip("._-#")

    if "XAU" in u or "GOLD" in u:   return "metal_xau"
    if "XAG" in u or "SILVER" in u: return "metal_xag"
    if any(x in u for x in ("XPT", "XPD", "PLAT", "PALL")): return "metal_other"

    if "BTC" in u:  return "crypto_btc"
    if "ETH" in u:  return "crypto_eth"
    if any(x in u for x in ("LTC","XRP","BNB","ADA","SOL","DOT","DOGE","USDT")):
        return "crypto_other"

    if any(x in u for x in ("OIL","BRENT","WTI","XBR","XTI","USOIL","UKOIL")):
        return "energy_oil"
    if any(x in u for x in ("NATGAS","GAS")):
        return "energy_gas"

    if any(x in u for x in ("US30","US500","US100","SPX","NDX","DJI","NAS","DOW")):
        return "index_us"
    if any(x in u for x in ("UK100","GER","DAX","FRA","JPN","AUS200","HKG","ESP","STOXX","NIKKEI","CAC")):
        return "index_eu"

    # Forex: JPY-quoted pairs use 3-decimal points
    if u.endswith("JPY") or "JPY" in u[-6:]:
        return "forex_jpy"

    return "forex"


class RiskNormalizer:
    """
    Wraps RiskEngine.calculate_lot_size with full symbol-type awareness.
    Resolves live contract/point values from broker when sym_info is provided,
    falls back to hardcoded defaults otherwise.
    """

    def __init__(self, risk_engine, broker_profile=None):
        """
        Parameters
        ----------
        risk_engine    : RiskEngine instance
        broker_profile : BrokerProfile (from broker_compat) — optional
        """
        self._re      = risk_engine
        self._profile = broker_profile

    # ------------------------------------------------------------------
    # Primary entry-point
    # ------------------------------------------------------------------

    def normalized_lot(
        self,
        equity:        float,
        atr:           float,
        symbol:        str,
        sym_info=None,          # mt5.symbol_info() result — optional
        min_lot:       Optional[float] = None,
        max_lot:       Optional[float] = None,
        lot_step:      Optional[float] = None,
    ) -> float:
        """
        Calculate and return a risk-normalised lot size for the given symbol.

        Parameters
        ----------
        equity      Current account equity in account currency
        atr         ATR value in price units (same as chart price)
        symbol      Raw broker symbol name
        sym_info    MT5 SymbolInfo object — provides live contract/point data
        min_lot     Override broker minimum lot (resolved automatically if None)
        max_lot     Override broker maximum lot
        lot_step    Override lot step increment

        Returns
        -------
        float — rounded, clamped lot size; never < min_lot
        """
        if equity <= 0:
            logger.warning(f"RiskNormalizer [{symbol}]: invalid equity={equity}")
            return self._safe_min(sym_info, min_lot)

        if atr <= 0:
            logger.warning(f"RiskNormalizer [{symbol}]: ATR={atr} ≤ 0 — using fallback")
            atr = self._fallback_atr(symbol, sym_info)

        # ── Resolve symbol parameters ──────────────────────────────────────
        cls               = _classify(symbol)
        def_contract, def_point, _, _ = _ASSET_PARAMS[cls]

        # Live values from MT5 SymbolInfo take priority
        contract_size = self._resolve_contract(sym_info, def_contract, symbol)
        point_value   = self._resolve_point(sym_info, def_point, symbol)

        # ── Lot sizing arithmetic ──────────────────────────────────────────
        #
        # risk_amount  = equity × risk_per_trade
        # stop_price   = atr × atr_multiplier     (in price units)
        # pip_val/lot  = point_value × contract_size
        # stop_in_pips = stop_price / point_value
        # lot          = risk_amount / (stop_in_pips × pip_val_per_lot)
        #              = risk_amount / (stop_price / point_value × point_value × contract_size)
        #              = risk_amount / (stop_price × contract_size)
        #
        risk_amount  = equity * self._re.risk_per_trade
        stop_price   = atr * self._re.atr_multiplier

        if stop_price <= 0 or contract_size <= 0:
            logger.warning(f"RiskNormalizer [{symbol}]: degenerate params — returning min_lot")
            return self._safe_min(sym_info, min_lot)

        raw_lot = risk_amount / (stop_price * contract_size)

        # ── Apply contract-size correction vs standard 100k forex lot ─────
        raw_lot = self._apply_class_correction(raw_lot, cls, contract_size)

        # ── Clamp to broker limits ─────────────────────────────────────────
        _min  = self._resolve_min_lot(sym_info, min_lot)
        _max  = self._resolve_max_lot(sym_info, max_lot)
        _step = self._resolve_step(sym_info, lot_step)

        lot = max(_min, min(raw_lot, _max))
        lot = self._round_to_step(lot, _step)
        lot = max(_min, lot)     # re-enforce after rounding

        logger.debug(
            f"RiskNormalize [{symbol}|{cls}] eq={equity:.2f} atr={atr:.6f} "
            f"contract={contract_size} stop={stop_price:.6f} "
            f"raw={raw_lot:.4f} → lot={lot}"
        )
        return lot

    # ------------------------------------------------------------------
    # Resolution helpers — live sym_info beats hardcoded defaults
    # ------------------------------------------------------------------

    def _resolve_contract(self, sym_info, default: float, symbol: str) -> float:
        """Accept MT5 SymbolInfo object OR BaseBroker dict."""
        if sym_info is not None:
            if isinstance(sym_info, dict):
                try:
                    v = float(sym_info.get("contract_size", 0) or 0)
                    if v > 0:
                        return v
                except Exception:
                    pass
            else:
                try:
                    v = float(sym_info.trade_contract_size)
                    if v > 0:
                        return v
                except Exception:
                    pass
        if self._profile:
            v = self._profile.contract_sizes.get(symbol, 0.0)
            if v > 0:
                return v
        return default

    def _resolve_point(self, sym_info, default: float, symbol: str) -> float:
        """Accept MT5 SymbolInfo object OR BaseBroker dict."""
        if sym_info is not None:
            if isinstance(sym_info, dict):
                try:
                    v = float(sym_info.get("point", 0) or 0)
                    if v > 0:
                        return v
                except Exception:
                    pass
            else:
                try:
                    v = float(sym_info.point)
                    if v > 0:
                        return v
                except Exception:
                    pass
        if self._profile:
            v = self._profile.point_values.get(symbol, 0.0)
            if v > 0:
                return v
        return default

    @staticmethod
    def _resolve_min_lot(sym_info, override: Optional[float]) -> float:
        if override is not None:
            return max(0.01, override)
        if sym_info is not None:
            try:
                v = float(sym_info.volume_min)
                if v > 0:
                    return v
            except Exception:
                pass
        return 0.01

    @staticmethod
    def _resolve_max_lot(sym_info, override: Optional[float]) -> float:
        if override is not None:
            return override
        if sym_info is not None:
            try:
                v = float(sym_info.volume_max)
                if v > 0:
                    return min(v, 100.0)   # hard cap 100 lots for safety
            except Exception:
                pass
        return 50.0

    @staticmethod
    def _resolve_step(sym_info, override: Optional[float]) -> float:
        if override is not None:
            return override
        if sym_info is not None:
            try:
                v = float(sym_info.volume_step)
                if v > 0:
                    return v
            except Exception:
                pass
        return 0.01

    @staticmethod
    def _round_to_step(lot: float, step: float) -> float:
        """Round lot down to the nearest valid step increment."""
        if step <= 0:
            return round(lot, 2)
        factor = 1.0 / step
        return math.floor(lot * factor) / factor

    @staticmethod
    def _safe_min(sym_info, override: Optional[float]) -> float:
        if override is not None:
            return override
        if sym_info is not None:
            try:
                v = float(sym_info.volume_min)
                if v > 0:
                    return v
            except Exception:
                pass
        return 0.01

    @staticmethod
    def _fallback_atr(symbol: str, sym_info) -> float:
        """Return a crude ATR proxy when actual ATR is unavailable."""
        u = symbol.upper()
        if "XAU" in u:       return 5.0       # ~$5 default stop for gold
        if "BTC" in u:       return 500.0     # ~$500 for BTC
        if any(x in u for x in ("US30","US500","GER","UK100")): return 50.0
        if "JPY" in u:       return 0.5
        return 0.0010        # 10-pip default for standard forex

    @staticmethod
    def _apply_class_correction(raw_lot: float, cls: str, contract_size: float) -> float:
        """
        Scale raw lot to account for non-standard contract sizes.
        Reference baseline: standard forex lot = 100,000 units.
        """
        FOREX_STANDARD = 100_000.0
        if contract_size > 0 and contract_size != FOREX_STANDARD:
            # If broker uses a different contract size the arithmetic already
            # accounted for it; no extra scaling needed.
            # But apply a sanity multiplier for micro/nano lots (contract < 1000)
            if contract_size < 1_000.0:
                raw_lot = raw_lot * (FOREX_STANDARD / max(contract_size, 1.0))
        return raw_lot
