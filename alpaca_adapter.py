"""
alpaca_adapter.py — Alpaca Markets Broker Adapter (AI EA v20)
=============================================================
Implements BaseBroker for Alpaca Markets (stocks, crypto, options).

Supports:
  - Alpaca REST API v2 (live + paper trading)
  - Alpaca Crypto API
  - WebSocket streaming (optional, for real-time bars)
  - Full offline/paper mode with simulated fills

Assets:   US Stocks, Crypto (BTC, ETH, …), ETFs
Accounts: Paper (free) and Live (funded)

Setup
-----
Set in .env:
  BROKER_TYPE=alpaca
  ALPACA_API_KEY=your_key
  ALPACA_SECRET_KEY=your_secret
  ALPACA_PAPER=true          # false for live trading
  ALPACA_DATA_FEED=iex       # iex (free) or sip (paid)
  ALPACA_BASE_URL=https://paper-api.alpaca.markets   # auto-set from ALPACA_PAPER

Requirements:
  pip install alpaca-py

Offline/Simulation mode:
  ALPACA_OFFLINE=true        # fully simulated, no API calls
"""

import json
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from base_broker import BaseBroker, OrderRejected

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TRADE_LOG_PATH   = "data/trade_log_alpaca.jsonl"
MAGIC_NUMBER     = 20250424
MAX_RETRIES      = 3
RETRY_DELAY      = 2.0

# Default Alpaca universe — stocks + crypto
DEFAULT_SYMBOLS = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL",
    "BTC/USD", "ETH/USD", "SOL/USD",
]

# Timeframe string → Alpaca TimeFrame mapping
_TF_MAP = {
    "m1":  "1Min",
    "m5":  "5Min",
    "m15": "15Min",
    "m30": "30Min",
    "h1":  "1Hour",
    "h2":  "2Hour",
    "h4":  "4Hour",
    "d1":  "1Day",
}

# ── Optional alpaca-py import ─────────────────────────────────────────────────
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest,
        GetOrdersRequest, GetAssetsRequest,
    )
    from alpaca.trading.enums import (
        OrderSide, TimeInForce, AssetClass, AssetStatus,
        OrderStatus,
    )
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import (
        StockBarsRequest, CryptoBarsRequest,
        StockLatestQuoteRequest, CryptoLatestQuoteRequest,
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.common.exceptions import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning(
        "alpaca-py not installed. Run: pip install alpaca-py\n"
        "Falling back to offline/simulation mode."
    )


def _parse_tf(tf_str: str):
    """Convert timeframe string to Alpaca TimeFrame."""
    if not ALPACA_AVAILABLE:
        return None
    mapping = {
        "m1":  TimeFrame(1,  TimeFrameUnit.Minute),
        "m5":  TimeFrame(5,  TimeFrameUnit.Minute),
        "m15": TimeFrame(15, TimeFrameUnit.Minute),
        "m30": TimeFrame(30, TimeFrameUnit.Minute),
        "h1":  TimeFrame(1,  TimeFrameUnit.Hour),
        "h2":  TimeFrame(2,  TimeFrameUnit.Hour),
        "h4":  TimeFrame(4,  TimeFrameUnit.Hour),
        "d1":  TimeFrame(1,  TimeFrameUnit.Day),
    }
    return mapping.get(tf_str.lower(), TimeFrame(1, TimeFrameUnit.Hour))


def _is_crypto(symbol: str) -> bool:
    """Return True if symbol is a crypto pair."""
    return "/" in symbol or any(
        c in symbol.upper() for c in ("BTC", "ETH", "SOL", "ADA", "XRP", "LTC", "DOGE")
    )


class AlpacaSimulator:
    """
    Offline simulator — generates synthetic OHLCV and simulates order fills.
    Activated when ALPACA_OFFLINE=true or alpaca-py is not installed.
    """

    def __init__(self):
        self._positions: Dict[str, Dict] = {}
        self._trade_counter = 1000
        self._equity = float(os.getenv("SIM_EQUITY", "10000"))
        self._balance = self._equity
        logger.info("[AlpacaSim] Offline simulation mode active")

    def get_bars(self, symbol: str, tf: str, n_bars: int) -> pd.DataFrame:
        """Generate synthetic OHLCV data with realistic price movement."""
        np.random.seed(abs(hash(symbol)) % 2**31)
        prices = [100.0]
        for _ in range(n_bars - 1):
            ret = np.random.normal(0, 0.001)
            prices.append(prices[-1] * (1 + ret))

        end = datetime.now(timezone.utc)
        minutes = {"m1": 1, "m5": 5, "m15": 15, "m30": 30,
                   "h1": 60, "h2": 120, "h4": 240, "d1": 1440}
        freq_m = minutes.get(tf, 60)
        timestamps = [end - timedelta(minutes=freq_m * i) for i in range(n_bars, 0, -1)]

        opens  = prices
        highs  = [p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices]
        lows   = [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices]
        closes = [p * (1 + np.random.normal(0, 0.0003)) for p in prices]
        vols   = [int(np.random.randint(100, 10000)) for _ in prices]

        df = pd.DataFrame({
            "open":        opens,
            "high":        highs,
            "low":         lows,
            "close":       closes,
            "tick_volume": vols,
            "real_volume": vols,
        }, index=pd.DatetimeIndex(timestamps, tz="UTC"))
        return df

    def place_order(self, symbol, side, qty, sl=None, tp=None) -> Dict:
        ticket = self._trade_counter
        self._trade_counter += 1
        price = 100.0 + np.random.uniform(-1, 1)
        self._positions[str(ticket)] = {
            "ticket": ticket, "symbol": symbol, "type": side,
            "volume": qty, "open_price": price,
            "sl": sl or price * 0.99, "tp": tp or price * 1.01,
            "profit": 0.0, "magic": MAGIC_NUMBER,
        }
        logger.info(f"[AlpacaSim] Order filled: {side} {qty} {symbol} @ {price:.4f}")
        return {"ticket": ticket, "success": True, "price": price}

    def close_order(self, ticket) -> bool:
        pos = self._positions.pop(str(ticket), None)
        if pos:
            pnl = np.random.uniform(-50, 100)
            self._equity += pnl
            logger.info(f"[AlpacaSim] Closed ticket={ticket} pnl={pnl:.2f}")
            return True
        return False

    def get_positions(self) -> List[Dict]:
        return list(self._positions.values())

    def get_account(self) -> Dict:
        return {
            "balance": self._balance, "equity": self._equity,
            "margin": 0.0, "free_margin": self._equity,
            "currency": "USD", "leverage": 1, "login": 0,
        }


class AlpacaAdapter(BaseBroker):
    """
    Alpaca Markets adapter — live, paper, and offline simulation.
    Drop-in replacement: set BROKER_TYPE=alpaca in .env.
    """

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        paper: bool = True,
        data_feed: str = "iex",
        offline: bool = False,
        risk_engine=None,
    ):
        super().__init__()
        self.broker_name = "Alpaca" + (" [Paper]" if paper else " [Live]")
        self._api_key    = api_key
        self._secret_key = secret_key
        self._paper      = paper
        self._data_feed  = data_feed
        self._offline    = offline or not ALPACA_AVAILABLE
        self._risk_engine = risk_engine

        # Alpaca client handles (set on connect)
        self._trading: Optional[object] = None
        self._stock_data: Optional[object] = None
        self._crypto_data: Optional[object] = None

        # Offline simulator fallback
        self._sim: Optional[AlpacaSimulator] = AlpacaSimulator() if self._offline else None

        # Symbol cache
        self._symbols_cache: List[Dict] = []
        self._symbols_ts: float = 0.0

        os.makedirs("data", exist_ok=True)

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        if self._offline:
            logger.info("[Alpaca] Offline/simulation mode — no API connection needed.")
            self.connected = True
            return True

        if not ALPACA_AVAILABLE:
            logger.error("[Alpaca] alpaca-py not installed. Run: pip install alpaca-py")
            logger.info("[Alpaca] Switching to offline simulation mode.")
            self._offline = True
            self._sim = AlpacaSimulator()
            self.connected = True
            return True

        if not self._api_key or not self._secret_key:
            logger.error("[Alpaca] API key or secret key not set. Check ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")
            return False

        try:
            self._trading = TradingClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper=self._paper,
            )
            self._stock_data = StockHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )
            self._crypto_data = CryptoHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )
            # Verify connection
            acct = self._trading.get_account()
            logger.info(
                f"[Alpaca] Connected | Account={acct.id} | "
                f"Equity=${float(acct.equity):.2f} | Paper={self._paper}"
            )
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"[Alpaca] Connection failed: {e}")
            logger.info("[Alpaca] Switching to offline simulation mode.")
            self._offline = True
            self._sim = AlpacaSimulator()
            self.connected = True
            return True

    def disconnect(self) -> None:
        self._trading = None
        self._stock_data = None
        self._crypto_data = None
        self.connected = False
        logger.info("[Alpaca] Disconnected.")

    # ── Symbols ───────────────────────────────────────────────────────────────

    def get_symbols(self) -> List[Dict]:
        if time.time() - self._symbols_ts < 300:
            return self._symbols_cache

        if self._offline:
            syms = [self._mock_sym_info(s) for s in DEFAULT_SYMBOLS]
            self._symbols_cache = syms
            self._symbols_ts = time.time()
            return syms

        try:
            req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
            assets = self._trading.get_all_assets(req)
            syms = []
            for a in assets:
                if a.tradable:
                    syms.append(self._asset_to_dict(a))
            # Add crypto
            for c in ["BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD"]:
                syms.append(self._mock_sym_info(c, is_crypto=True))
            self._symbols_cache = syms
            self._symbols_ts = time.time()
            return syms
        except Exception as e:
            logger.warning(f"[Alpaca] get_symbols error: {e} — using defaults")
            return [self._mock_sym_info(s) for s in DEFAULT_SYMBOLS]

    def _asset_to_dict(self, asset) -> Dict:
        return {
            "name":          asset.symbol,
            "contract_size": 1.0,
            "point":         0.01,
            "digits":        2,
            "min_lot":       1.0,
            "max_lot":       100000.0,
            "lot_step":      1.0,
            "spread":        0.0,
            "trade_mode":    "full",
            "asset_class":   "stock",
        }

    def _mock_sym_info(self, symbol: str, is_crypto: bool = False) -> Dict:
        crypto = is_crypto or _is_crypto(symbol)
        return {
            "name":          symbol,
            "contract_size": 1.0,
            "point":         0.01 if not crypto else 1.0,
            "digits":        2 if not crypto else 2,
            "min_lot":       1.0 if not crypto else 0.001,
            "max_lot":       100000.0,
            "lot_step":      1.0 if not crypto else 0.001,
            "spread":        0.0,
            "trade_mode":    "full",
            "asset_class":   "crypto" if crypto else "stock",
        }

    # ── Market Data ───────────────────────────────────────────────────────────

    def get_market_data(
        self,
        symbol: str,
        timeframe: str = "h1",
        bars: int = 500,
    ) -> Optional[pd.DataFrame]:

        if self._offline:
            return self._sim.get_bars(symbol, timeframe, bars)

        try:
            tf = _parse_tf(timeframe)
            end   = datetime.now(timezone.utc)
            # Estimate start based on bars requested
            minutes = {"m1": 1, "m5": 5, "m15": 15, "m30": 30,
                       "h1": 60, "h2": 120, "h4": 240, "d1": 1440}
            freq_m = minutes.get(timeframe.lower(), 60)
            # Buffer by 1.5x to account for market closed hours
            start = end - timedelta(minutes=int(bars * freq_m * 1.5))

            if _is_crypto(symbol):
                req = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start,
                    end=end,
                    limit=bars,
                )
                bars_data = self._crypto_data.get_crypto_bars(req)
            else:
                req = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start,
                    end=end,
                    limit=bars,
                    feed=self._data_feed,
                )
                bars_data = self._stock_data.get_stock_bars(req)

            df_raw = bars_data[symbol].df if hasattr(bars_data, '__getitem__') else bars_data.df
            if df_raw is None or df_raw.empty:
                return None

            df = pd.DataFrame({
                "open":        df_raw["open"].astype(float),
                "high":        df_raw["high"].astype(float),
                "low":         df_raw["low"].astype(float),
                "close":       df_raw["close"].astype(float),
                "tick_volume": df_raw["volume"].astype(float),
                "real_volume": df_raw["volume"].astype(float),
            }, index=df_raw.index)

            # Ensure UTC timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            return df.tail(bars)

        except Exception as e:
            logger.error(f"[Alpaca] get_market_data {symbol} {timeframe} error: {e}")
            if self._sim:
                return self._sim.get_bars(symbol, timeframe, bars)
            return None

    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        if self._offline:
            price = 100.0 + np.random.uniform(-5, 5)
            return {"bid": price * 0.9999, "ask": price * 1.0001, "spread": price * 0.0002, "time": str(datetime.now())}

        try:
            if _is_crypto(symbol):
                req = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
                quote = self._crypto_data.get_crypto_latest_quote(req)[symbol]
            else:
                req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quote = self._stock_data.get_stock_latest_quote(req)[symbol]

            return {
                "bid":    float(quote.bid_price),
                "ask":    float(quote.ask_price),
                "spread": float(quote.ask_price - quote.bid_price),
                "time":   str(quote.timestamp),
            }
        except Exception as e:
            logger.warning(f"[Alpaca] get_latest_price {symbol}: {e}")
            return None

    # ── Order Execution ───────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "AI_EA_v20",
        magic: int = MAGIC_NUMBER,
    ) -> Optional[Dict]:

        if self._offline:
            return self._sim.place_order(symbol, order_type, volume, sl, tp)

        side = OrderSide.BUY if order_type.lower() == "buy" else OrderSide.SELL

        # Alpaca uses fractional shares for crypto, whole for stocks
        qty = volume
        if not _is_crypto(symbol):
            qty = max(1, int(volume))

        for attempt in range(MAX_RETRIES):
            try:
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.GTC if _is_crypto(symbol) else TimeInForce.DAY,
                    client_order_id=f"aiea_{magic}_{int(time.time())}",
                )
                order = self._trading.submit_order(req)
                ticket = str(order.id)
                fill_price = float(order.filled_avg_price) if order.filled_avg_price else 0.0

                result = {
                    "ticket":  ticket,
                    "symbol":  symbol,
                    "type":    order_type.lower(),
                    "volume":  float(qty),
                    "price":   fill_price,
                    "sl":      sl,
                    "tp":      tp,
                    "comment": comment,
                    "retcode": "TRADE_RETCODE_DONE",
                    "success": True,
                }
                self._log_trade(result)
                logger.info(
                    f"[Alpaca] Order placed: {order_type.upper()} {qty} {symbol} "
                    f"ticket={ticket} price={fill_price:.4f}"
                )
                return result

            except Exception as e:
                logger.warning(f"[Alpaca] place_order attempt {attempt+1}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)

        logger.error(f"[Alpaca] place_order failed after {MAX_RETRIES} attempts")
        return None

    def close_order(self, ticket: int, symbol: str = "", volume: float = 0.0) -> bool:
        if self._offline:
            return self._sim.close_order(ticket)

        try:
            self._trading.close_position(str(ticket))
            logger.info(f"[Alpaca] Closed position ticket={ticket}")
            return True
        except Exception as e:
            logger.error(f"[Alpaca] close_order {ticket}: {e}")
            # Try closing by symbol if ticket fails
            if symbol:
                try:
                    self._trading.close_position(symbol)
                    return True
                except Exception as e2:
                    logger.error(f"[Alpaca] close_order by symbol {symbol}: {e2}")
            return False

    def modify_order(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        # Alpaca doesn't support native SL/TP modification on market orders.
        # AI EA handles SL/TP via manual monitoring in the main loop.
        logger.debug(f"[Alpaca] modify_order: SL/TP managed by EA (ticket={ticket})")
        return True

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account_info(self) -> Optional[Dict]:
        if self._offline:
            return self._sim.get_account()

        try:
            acct = self._trading.get_account()
            return {
                "balance":     float(acct.cash),
                "equity":      float(acct.equity),
                "margin":      float(acct.initial_margin) if acct.initial_margin else 0.0,
                "free_margin": float(acct.buying_power),
                "currency":    "USD",
                "leverage":    1,
                "login":       str(acct.id)[:8],
            }
        except Exception as e:
            logger.error(f"[Alpaca] get_account_info: {e}")
            return None

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_open_positions(self, symbol: str = "") -> List[Dict]:
        if self._offline:
            positions = self._sim.get_positions()
            if symbol:
                return [p for p in positions if p["symbol"] == symbol]
            return positions

        try:
            positions = self._trading.get_all_positions()
            result = []
            for pos in positions:
                if symbol and pos.symbol != symbol:
                    continue
                result.append({
                    "ticket":     str(pos.asset_id),
                    "symbol":     pos.symbol,
                    "type":       "buy" if float(pos.qty) > 0 else "sell",
                    "volume":     abs(float(pos.qty)),
                    "open_price": float(pos.avg_entry_price),
                    "sl":         None,
                    "tp":         None,
                    "profit":     float(pos.unrealized_pl),
                    "magic":      MAGIC_NUMBER,
                    "comment":    "AI_EA",
                })
            return result
        except Exception as e:
            logger.error(f"[Alpaca] get_open_positions: {e}")
            return []

    # ── Trade History (for ML learning) ──────────────────────────────────────

    def _load_local_trade_log(self, symbol: str = "") -> List[Dict]:
        """Load trades from local JSONL log (written by _log_trade). Works in all modes."""
        if not os.path.exists(TRADE_LOG_PATH):
            return []
        trades = []
        try:
            with open(TRADE_LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                        if symbol and t.get("symbol") != symbol:
                            continue
                        trades.append(t)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"[Alpaca] _load_local_trade_log: {e}")
        return trades

    def get_trade_history(self, days: int = 90, symbol: str = "") -> List[Dict]:
        """
        Fetch ALL closed orders from Alpaca with accurate FIFO PnL.  (v20)

        v20 FIX: Alpaca market orders don't carry PnL on the order object.
        We reconstruct PnL by matching BUY → SELL pairs per symbol (FIFO).
        Also merges local trade_log_alpaca.jsonl so paper/offline trades are
        included and TradeHistoryLearner always has data to learn from.
        """
        local_trades = self._load_local_trade_log(symbol)

        if self._offline:
            return local_trades

        try:
            since = datetime.now(timezone.utc) - timedelta(days=days)
            req = GetOrdersRequest(
                status=OrderStatus.CLOSED,
                after=since,
                limit=500,
                symbols=[symbol] if symbol else None,
            )
            orders = self._trading.get_orders(req)

            # Keep only filled orders
            filled = [o for o in orders
                      if o.filled_at is not None and float(o.filled_qty or 0) > 0]

            # Group by symbol, sort by fill time
            by_sym: Dict[str, list] = {}
            for o in filled:
                by_sym.setdefault(o.symbol, []).append(o)

            history = []
            for sym, sym_orders in by_sym.items():
                sym_orders.sort(key=lambda o: o.filled_at)
                buys: List[Dict] = []   # FIFO queue of open buy lots
                for o in sym_orders:
                    qty   = float(o.filled_qty or 0)
                    price = float(o.filled_avg_price or 0)
                    if o.side == OrderSide.BUY:
                        buys.append({"qty": qty, "price": price,
                                     "ts": o.created_at, "id": str(o.id)})
                    else:
                        # SELL — match against earliest BUYs (FIFO)
                        remaining = qty
                        pnl = 0.0
                        entry_price = price
                        entry_ts    = o.created_at
                        while remaining > 1e-9 and buys:
                            b       = buys[0]
                            matched = min(b["qty"], remaining)
                            pnl    += matched * (price - b["price"])
                            entry_price = b["price"]
                            entry_ts    = b["ts"]
                            b["qty"]   -= matched
                            remaining  -= matched
                            if b["qty"] <= 1e-9:
                                buys.pop(0)

                        history.append({
                            "ticket":      str(o.id),
                            "symbol":      sym,
                            "type":        "buy",
                            "volume":      qty,
                            "open_price":  round(entry_price, 6),
                            "close_price": round(price, 6),
                            "profit":      round(pnl, 4),
                            "open_time":   str(entry_ts),
                            "close_time":  str(o.filled_at),
                            "strategy":    "AI_EA",
                        })

            logger.info(
                f"[Alpaca] Loaded {len(history)} paired trades (last {days}d) "
                f"+ {len(local_trades)} from local log"
            )
            # Merge, deduplicate by ticket
            merged = {t["ticket"]: t for t in local_trades}
            for t in history:
                merged[t["ticket"]] = t
            return list(merged.values())

        except Exception as e:
            logger.error(f"[Alpaca] get_trade_history: {e}")
            return local_trades

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log_trade(self, result: Dict) -> None:
        try:
            with open(TRADE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({**result, "ts": datetime.utcnow().isoformat()}) + "\n")
        except Exception:
            pass

    def __repr__(self) -> str:
        mode = "OFFLINE" if self._offline else ("PAPER" if self._paper else "LIVE")
        return f"<AlpacaAdapter mode={mode} connected={self.connected}>"
