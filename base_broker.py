"""
base_broker.py — Universal Broker Interface (AI EA v5)
=======================================================
Abstract base class that every broker adapter must implement.
All downstream components (executor, data_fetcher, ai_ea) depend ONLY
on this interface — never on a broker-specific implementation.

Standardised data structures returned by all adapters:

  AccountInfo   dict with keys: balance, equity, margin, free_margin,
                                currency, leverage, login
  SymbolInfo    dict with keys: name, contract_size, point, digits,
                                min_lot, max_lot, lot_step, spread,
                                trade_mode, asset_class
  Candle row    dict / DataFrame row with: open, high, low, close,
                                           tick_volume, real_volume, time
  Position      dict with keys: ticket, symbol, type, volume,
                                 open_price, sl, tp, profit, magic
  OrderResult   dict with keys: ticket, symbol, type, volume, price,
                                 sl, tp, comment, retcode, success
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class BrokerError(Exception):
    """Raised when a broker adapter encounters an unrecoverable error."""


class OrderRejected(BrokerError):
    """Raised when a broker rejects an order placement."""


class BaseBroker(ABC):
    """
    Universal broker interface.  Every adapter (MT5, IBKR, cTrader)
    must subclass this and implement every abstract method.

    Concrete adapters must call super().__init__() and set self.connected.
    """

    def __init__(self):
        self.connected: bool = False
        self.broker_name: str = "Unknown"
        self._account_cache: Optional[Dict] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Connection lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to broker.
        Returns True on success, False on failure.
        Sets self.connected = True if successful.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Gracefully close connection. Sets self.connected = False."""

    def ensure_connected(self) -> bool:
        """Return True if connected; attempt reconnect if not."""
        if self.connected:
            return True
        logger.warning(f"[{self.broker_name}] Not connected — reconnecting...")
        return self.connect()

    # ─────────────────────────────────────────────────────────────────────────
    # Symbol discovery
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_symbols(self) -> List[Dict]:
        """
        Return list of all available tradable instruments.
        Each entry is a SymbolInfo dict:
          { name, contract_size, point, digits, min_lot, max_lot,
            lot_step, spread, trade_mode, asset_class }
        """

    def get_symbol_names(self) -> List[str]:
        """Convenience: return just the symbol name strings."""
        return [s["name"] for s in self.get_symbols()]

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Return SymbolInfo dict for a single symbol, or None."""
        for s in self.get_symbols():
            if s["name"] == symbol:
                return s
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Market data
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        bars: int = 500,
    ) -> Optional[pd.DataFrame]:
        """
        Return OHLCV DataFrame with DatetimeIndex.
        Columns: open, high, low, close, tick_volume, real_volume
        timeframe strings: 'm1','m5','m15','m30','h1','h2','h4','d1','w1'
        Returns None on failure.
        """

    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        """
        Return latest bid/ask/spread for a symbol.
        Returns { bid, ask, spread, time } or None.
        Default impl: uses last candle from 1-bar h1 request.
        Adapters should override with native tick fetch.
        """
        df = self.get_market_data(symbol, "m1", 1)
        if df is None or df.empty:
            return None
        row = df.iloc[-1]
        price = float(row["close"])
        return {"bid": price, "ask": price, "spread": 0.0, "time": str(df.index[-1])}

    # ─────────────────────────────────────────────────────────────────────────
    # Order execution
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        order_type: str,       # "buy" | "sell"
        volume: float,
        price: Optional[float] = None,   # None = market order
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "AI_EA_v5",
        magic: int = 20250424,
    ) -> Optional[Dict]:
        """
        Place a market or limit order.
        Returns OrderResult dict on success, None on failure.
        OrderResult: { ticket, symbol, type, volume, price, sl, tp,
                       comment, retcode, success }
        """

    @abstractmethod
    def close_order(self, ticket: int, symbol: str = "", volume: float = 0.0) -> bool:
        """
        Close an open position by ticket.
        symbol and volume are hints for brokers that need them (e.g. cTrader).
        Returns True on success.
        """

    @abstractmethod
    def modify_order(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        """
        Modify SL/TP of an existing open position.
        Returns True on success.
        """

    # ─────────────────────────────────────────────────────────────────────────
    # Account information
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_account_info(self) -> Optional[Dict]:
        """
        Return account state.
        Keys: balance, equity, margin, free_margin, currency, leverage, login
        Returns None on failure.
        """

    def get_equity(self) -> float:
        """Return current equity; 0.0 on failure."""
        info = self.get_account_info()
        return float(info.get("equity", 0.0)) if info else 0.0

    def get_balance(self) -> float:
        """Return current balance; 0.0 on failure."""
        info = self.get_account_info()
        return float(info.get("balance", 0.0)) if info else 0.0

    def get_free_margin(self) -> float:
        """Return free margin; 0.0 on failure."""
        info = self.get_account_info()
        return float(info.get("free_margin", 0.0)) if info else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Position management
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_open_positions(self, symbol: str = "") -> List[Dict]:
        """
        Return list of open positions.
        If symbol provided, filter to that symbol only.
        Each position dict: { ticket, symbol, type, volume, open_price,
                               sl, tp, profit, magic, comment }
        """

    def count_open_positions(self, symbol: str = "") -> int:
        """Return count of open positions."""
        return len(self.get_open_positions(symbol))

    def get_position_by_ticket(self, ticket: int) -> Optional[Dict]:
        """Return a specific position by ticket number."""
        for pos in self.get_open_positions():
            if pos.get("ticket") == ticket:
                return pos
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Trade history
    # ─────────────────────────────────────────────────────────────────────────

    def get_trade_history(self, days: int = 365, symbol: str = "") -> List[Dict]:
        """
        Return closed trade history (last N days).
        Default implementation returns empty list — override in adapters
        that support history retrieval.
        """
        return []

    # ─────────────────────────────────────────────────────────────────────────
    # Validation helpers (shared across all adapters)
    # ─────────────────────────────────────────────────────────────────────────

    def validate_volume(self, volume: float, sym_info: Optional[Dict]) -> float:
        """Clamp and round volume to broker constraints."""
        if sym_info is None:
            return round(max(0.01, volume), 2)
        min_lot  = float(sym_info.get("min_lot",  0.01))
        max_lot  = float(sym_info.get("max_lot",  100.0))
        lot_step = float(sym_info.get("lot_step", 0.01))
        volume   = max(min_lot, min(float(volume), max_lot))
        if lot_step > 0:
            import math
            # Use integer arithmetic to avoid float imprecision.
            # e.g. floor(0.7 / 0.1) * 0.1 → 0.6000000000000001 (float error)
            # Scaling to int: floor(70 / 10) * 0.1 = 7 * 0.1 = 0.7 (exact)
            scale  = round(1.0 / lot_step)
            volume = math.floor(round(volume * scale)) / scale
        return float(round(max(min_lot, volume), 8))

    def validate_price(self, price: float, digits: int) -> float:
        """Round price to broker digit precision."""
        return round(price, digits)

    # ─────────────────────────────────────────────────────────────────────────
    # Asset classification helper
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def classify_asset(symbol: str) -> str:
        """Return asset class string for a symbol name."""
        u = symbol.upper().strip("._-#")
        if any(x in u for x in ("XAU", "GOLD")): return "metal"
        if any(x in u for x in ("XAG", "SILVER")): return "metal"
        if any(x in u for x in ("BTC", "ETH", "LTC", "XRP", "BNB", "ADA", "SOL")): return "crypto"
        if any(x in u for x in ("OIL", "BRENT", "WTI", "XBR", "XTI", "NATGAS", "GAS")): return "energy"
        if any(x in u for x in ("US30", "US500", "US100", "SPX", "NDX", "DJI",
                                  "UK100", "GER", "DAX", "FRA", "JPN", "AUS200",
                                  "HKG", "STOXX", "NIKKEI", "CAC", "NAS", "DOW")): return "index"
        if u.endswith("JPY"): return "forex"
        # Generic: check for 6-char forex pair
        if 5 <= len(u) <= 8:
            return "forex"
        return "stock"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} broker={self.broker_name} connected={self.connected}>"
