"""
ctrader_adapter.py — Full Spotware cTrader Open API Adapter (AI EA v5)
=======================================================================
Implements BaseBroker for cTrader brokers via the Spotware Open API.

Supports all Spotware cTrader brokers (IC Markets cTrader, Pepperstone
cTrader, FxPro cTrader, etc.)

Requirements:
    pip install ctrader-open-api

Spotware Open API endpoints:
    Live:  tradeserver.ctraderaccessory.com:5035  (SSL)
    Demo:  demo.ctraderaccessory.com:5035         (SSL)

Authentication flow:
    1. TCP/SSL connection to Spotware server
    2. ProtoOAApplicationAuthReq  (clientId + clientSecret)
    3. ProtoOAAccountAuthReq      (accessToken per trading account)
    4. Trading is now live

All methods return standardised dicts from BaseBroker — no cTrader-specific
proto objects leak outside this module.
"""

import logging
import math
import os
import time
import threading
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Callable, Any

import pandas as pd
import numpy as np

from base_broker import BaseBroker, OrderRejected

logger = logging.getLogger(__name__)

# ── Optional cTrader Open API import ─────────────────────────────────────────
try:
    from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAApplicationAuthReq,
        ProtoOAAccountAuthReq,
        ProtoOASymbolsListReq,
        ProtoOASymbolByIdReq,
        ProtoOAGetTrendbarsReq,
        ProtoOANewOrderReq,
        ProtoOAAmendPositionSLTPReq,
        ProtoOAClosePositionReq,
        ProtoOAReconcileReq,
        ProtoOATraderReq,
        ProtoOASubscribeLiveTrendbarReq,
        ProtoOAGetTickDataReq,
    )
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import (
        ProtoOAOrderType,
        ProtoOATradeSide,
        ProtoOATrendbarPeriod,
    )
    CTRADER_AVAILABLE = True
except ImportError:
    CTRADER_AVAILABLE = False
    logger.warning(
        "[cTrader] ctrader-open-api not installed. "
        "Run: pip install ctrader-open-api"
    )

# ── Timeframe string → ProtoOA period mapping ─────────────────────────────────
_CT_TF_MAP = {
    "m1":  1,   # ProtoOATrendbarPeriod.M1
    "m2":  2,
    "m3":  3,
    "m4":  4,
    "m5":  5,   # M5
    "m10": 6,
    "m15": 7,   # M15
    "m30": 8,   # M30
    "h1":  9,   # H1
    "h4":  10,  # H4
    "h12": 11,
    "d1":  12,  # D1
    "w1":  13,  # W1
    "mn1": 14,  # MN1
}

# Seconds per period (for duration calculations)
_PERIOD_SECS = {
    1: 60, 2: 120, 3: 180, 4: 240, 5: 300, 6: 600,
    7: 900, 8: 1800, 9: 3600, 10: 14400, 11: 43200,
    12: 86400, 13: 604800, 14: 2592000,
}

MAGIC_NUMBER  = 20250424
TRADE_LOG_PATH = "data/trade_log.jsonl"


class CTraderAdapter(BaseBroker):
    """
    Full Spotware cTrader Open API adapter.

    Uses synchronous blocking wrappers around the async Twisted/asyncio
    event loop so the rest of the AI EA v5 engine (which is synchronous)
    can call all methods without change.
    """

    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        access_token: str = "",
        account_id: int = 0,
        host: str = "",
        port: int = 5035,
        demo: bool = True,
        risk_engine=None,
    ):
        super().__init__()
        self.broker_name   = "cTrader"
        self.client_id     = client_id or os.getenv("CTRADER_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("CTRADER_CLIENT_SECRET", "")
        self.access_token  = access_token or os.getenv("CTRADER_ACCESS_TOKEN", "")
        self.account_id    = account_id or int(os.getenv("CTRADER_ACCOUNT_ID", "0"))
        self.demo          = demo
        self.risk_engine   = risk_engine

        # Pick endpoint
        if host:
            self._host = host
            self._port = port
        else:
            if demo:
                self._host = "demo.ctraderaccessory.com"
            else:
                self._host = "tradeserver.ctraderaccessory.com"
            self._port = 5035

        # Internal state
        self._client: Optional[Any] = None
        self._symbols_cache: List[Dict] = []
        self._symbols_ts: float = 0.0
        self._symbols_map: Dict[str, int] = {}   # symbol_name → symbolId
        self._pending_responses: Dict[int, Any] = {}
        self._response_events: Dict[int, threading.Event] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._app_authenticated = False
        self._account_authenticated = False
        self._account_info_cache: Optional[Dict] = None
        self._lock = threading.Lock()
        self._req_id = 1

        os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Connection
    # ─────────────────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        if not CTRADER_AVAILABLE:
            logger.error(
                "[cTrader] ctrader-open-api package not installed. "
                "Install with: pip install ctrader-open-api"
            )
            return False

        if not self.client_id or not self.client_secret:
            logger.error("[cTrader] client_id and client_secret required.")
            return False

        if not self.access_token or not self.account_id:
            logger.error("[cTrader] access_token and account_id required.")
            return False

        try:
            # Build synchronous event loop in a background thread
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._run_loop, daemon=True, name="ctrader-loop"
            )
            self._loop_thread.start()

            # Create client and connect
            future = asyncio.run_coroutine_threadsafe(
                self._async_connect(), self._loop
            )
            result = future.result(timeout=30)
            if result:
                self.connected = True
                logger.info(
                    f"[cTrader] Connected to {self._host}:{self._port} "
                    f"| account={self.account_id}"
                )
            return result
        except Exception as e:
            logger.error(f"[cTrader] connect() failed: {e}", exc_info=True)
            return False

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _async_connect(self) -> bool:
        try:
            self._client = Client(
                self._host, self._port, TcpProtocol
            )

            # Application auth
            app_auth = ProtoOAApplicationAuthReq()
            app_auth.clientId     = self.client_id
            app_auth.clientSecret = self.client_secret
            await self._async_send_recv(app_auth, timeout=15)
            self._app_authenticated = True
            logger.info("[cTrader] Application authenticated.")

            # Account auth
            acc_auth = ProtoOAAccountAuthReq()
            acc_auth.ctidTraderAccountId = self.account_id
            acc_auth.accessToken         = self.access_token
            await self._async_send_recv(acc_auth, timeout=15)
            self._account_authenticated = True
            logger.info(f"[cTrader] Account {self.account_id} authenticated.")

            # Pre-load symbol table
            await self._async_load_symbols()
            return True
        except Exception as e:
            logger.error(f"[cTrader] _async_connect: {e}")
            return False

    async def _async_send_recv(self, request, timeout: float = 10.0) -> Any:
        """Send a proto request and await response (non-blocking)."""
        if self._client is None:
            raise ConnectionError("[cTrader] Client not initialised")
        # The ctrader-open-api library uses callbacks; wrap in asyncio Future
        future = self._loop.create_future()

        def on_message(client, msg, ident):
            if not future.done():
                self._loop.call_soon_threadsafe(future.set_result, msg)

        def on_error(failure):
            if not future.done():
                self._loop.call_soon_threadsafe(
                    future.set_exception, Exception(str(failure))
                )

        d = self._client.send(request)
        if hasattr(d, "addCallback"):
            d.addCallback(on_message, None)
            d.addErrback(on_error)
        else:
            # Fallback for plain coroutine-based clients
            return await asyncio.wait_for(asyncio.shield(d), timeout=timeout)

        return await asyncio.wait_for(future, timeout=timeout)

    def disconnect(self) -> None:
        try:
            if self._client is not None:
                try:
                    self._client.stopService()
                except Exception:
                    pass
            if self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception as e:
            logger.warning(f"[cTrader] disconnect: {e}")
        finally:
            self.connected = False
            self._app_authenticated = False
            self._account_authenticated = False
            logger.info("[cTrader] Disconnected.")

    # ─────────────────────────────────────────────────────────────────────────
    # Symbol discovery
    # ─────────────────────────────────────────────────────────────────────────

    async def _async_load_symbols(self) -> None:
        """Fetch full symbol list from cTrader and populate cache."""
        req = ProtoOASymbolsListReq()
        req.ctidTraderAccountId = self.account_id
        req.includeArchivedSymbols = False
        try:
            resp = await self._async_send_recv(req, timeout=20)
            symbols_out = []
            for sym in resp.symbol:
                sym_id   = sym.symbolId
                sym_name = sym.symbolName
                self._symbols_map[sym_name.upper()] = sym_id

                # Classify asset class
                asset_cls = BaseBroker.classify_asset(sym_name)

                entry = {
                    "name":          sym_name,
                    "symbol_id":     sym_id,
                    "contract_size": float(getattr(sym, "lotSize", 100000)),
                    "point":         pow(10, -int(getattr(sym, "digits", 5))),
                    "digits":        int(getattr(sym, "digits", 5)),
                    "min_lot":       float(getattr(sym, "minVolume", 1000)) / 100.0,
                    "max_lot":       float(getattr(sym, "maxVolume", 1000000)) / 100.0,
                    "lot_step":      float(getattr(sym, "stepVolume", 100)) / 100.0,
                    "spread":        float(getattr(sym, "spread", 0)),
                    "trade_mode":    1,
                    "asset_class":   asset_cls,
                }
                symbols_out.append(entry)

            self._symbols_cache = symbols_out
            self._symbols_ts = time.time()
            logger.info(f"[cTrader] Loaded {len(symbols_out)} symbols.")
        except Exception as e:
            logger.error(f"[cTrader] _async_load_symbols: {e}")

    def get_symbols(self) -> List[Dict]:
        if not self.connected:
            return []
        # Refresh if cache is stale (>1hr)
        if time.time() - self._symbols_ts > 3600:
            future = asyncio.run_coroutine_threadsafe(
                self._async_load_symbols(), self._loop
            )
            try:
                future.result(timeout=20)
            except Exception as e:
                logger.warning(f"[cTrader] get_symbols refresh: {e}")
        return list(self._symbols_cache)

    def _get_symbol_id(self, symbol: str) -> Optional[int]:
        """Resolve symbol name → cTrader symbolId."""
        # Try exact match
        sid = self._symbols_map.get(symbol.upper())
        if sid:
            return sid
        # Fuzzy: strip suffix
        clean = symbol.upper().rstrip("M.").rstrip("_")
        for name, sid in self._symbols_map.items():
            if name.startswith(clean):
                return sid
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Market data
    # ─────────────────────────────────────────────────────────────────────────

    def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        bars: int = 500,
    ) -> Optional[pd.DataFrame]:
        if not self.ensure_connected():
            return None
        future = asyncio.run_coroutine_threadsafe(
            self._async_get_market_data(symbol, timeframe, bars), self._loop
        )
        try:
            return future.result(timeout=30)
        except Exception as e:
            logger.error(f"[cTrader] get_market_data({symbol}): {e}")
            return None

    async def _async_get_market_data(
        self,
        symbol: str,
        timeframe: str,
        bars: int,
    ) -> Optional[pd.DataFrame]:
        sym_id = self._get_symbol_id(symbol)
        if sym_id is None:
            logger.error(f"[cTrader] Unknown symbol: {symbol}")
            return None

        period = _CT_TF_MAP.get(timeframe.lower(), 9)  # default H1
        period_secs = _PERIOD_SECS.get(period, 3600)

        # cTrader uses Unix timestamps in milliseconds
        to_ts   = int(time.time() * 1000)
        from_ts = to_ts - bars * period_secs * 1000

        req = ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId            = sym_id
        req.period              = period
        req.fromTimestamp       = from_ts
        req.toTimestamp         = to_ts
        req.count               = min(bars, 5000)

        try:
            resp = await self._async_send_recv(req, timeout=30)
            if not resp.trendbar:
                logger.warning(f"[cTrader] No trendbar data: {symbol} {timeframe}")
                return None

            rows = []
            for bar in resp.trendbar:
                # cTrader prices are in relative ticks (divide by 1e5 for most)
                divisor = 100000.0
                rows.append({
                    "time":        datetime.utcfromtimestamp(bar.utcTimestampInMinutes * 60),
                    "open":        (bar.low + bar.deltaOpen) / divisor,
                    "high":        (bar.low + bar.deltaHigh) / divisor,
                    "low":         bar.low / divisor,
                    "close":       (bar.low + bar.deltaClose) / divisor,
                    "tick_volume": int(bar.volume),
                    "real_volume": int(bar.volume),
                })

            if not rows:
                return None

            df = pd.DataFrame(rows)
            df.set_index("time", inplace=True)
            df.sort_index(inplace=True)
            df = df.tail(bars)
            df["symbol"] = symbol
            df = self._add_indicators(df)
            return df
        except Exception as e:
            logger.error(f"[cTrader] _async_get_market_data({symbol}): {e}")
            return None

    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        """Get latest bid/ask from a 1-bar M1 pull."""
        df = self.get_market_data(symbol, "m1", 1)
        if df is None or df.empty:
            return None
        close = float(df.iloc[-1]["close"])
        # Estimate spread from symbol info
        spread = 0.0
        for s in self._symbols_cache:
            if s["name"].upper() == symbol.upper():
                spread = s["spread"] * s["point"]
                break
        return {
            "bid":    close,
            "ask":    close + spread,
            "spread": spread,
            "time":   datetime.now(timezone.utc).isoformat(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Order placement
    # ─────────────────────────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        order_type: str,       # "buy" | "sell"
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "AI_EA_v5",
        magic: int = MAGIC_NUMBER,
        atr: float = 0.0,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 2.5,
        signal_prob: float = 0.65,
    ) -> Optional[Dict]:
        if not self.ensure_connected():
            return None

        # Risk engine approval
        if self.risk_engine:
            equity   = self.get_equity()
            open_cnt = self.count_open_positions()
            approved, reason = self.risk_engine.approve_trade(
                equity=equity, open_positions=open_cnt,
                symbol=symbol, signal_prob=signal_prob,
            )
            if not approved:
                logger.warning(f"[cTrader] Trade BLOCKED [{symbol}]: {reason}")
                return None

        sym_id = self._get_symbol_id(symbol)
        if sym_id is None:
            logger.error(f"[cTrader] place_order: unknown symbol {symbol}")
            return None

        # Validate volume (cTrader uses units × 100 internally)
        sym_info = self.get_symbol_info(symbol)
        vol = self.validate_volume(volume, sym_info)

        # Auto-calculate SL/TP from ATR if not supplied
        if (sl is None or tp is None) and atr > 0:
            latest = self.get_latest_price(symbol)
            if latest:
                exec_price = latest["ask"] if order_type.lower() == "buy" else latest["bid"]
                direction  = 1 if order_type.lower() == "buy" else -1
                if sl is None:
                    sl = round(exec_price - direction * atr * sl_atr_mult, 5)
                if tp is None:
                    tp = round(exec_price + direction * atr * tp_atr_mult, 5)

        future = asyncio.run_coroutine_threadsafe(
            self._async_place_order(
                sym_id, symbol, order_type, vol, price, sl, tp, comment
            ),
            self._loop,
        )
        try:
            result = future.result(timeout=15)
            # NOTE: record_trade_open() is called by ai_ea.py — do not call here.
            return result
        except Exception as e:
            logger.error(f"[cTrader] place_order({symbol}): {e}")
            return None

    async def _async_place_order(
        self,
        sym_id: int,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float],
        sl: Optional[float],
        tp: Optional[float],
        comment: str,
    ) -> Optional[Dict]:
        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId            = sym_id
        req.orderType           = (
            ProtoOAOrderType.MARKET if price is None
            else ProtoOAOrderType.LIMIT
        )
        req.tradeSide = (
            ProtoOATradeSide.BUY if order_type.lower() == "buy"
            else ProtoOATradeSide.SELL
        )
        # cTrader volumes are in units × 100 (e.g. 1 lot = 100000 units)
        req.volume = int(volume * 100)
        req.comment = comment[:64]

        if price is not None:
            req.limitPrice = int(price * 100000)

        if sl is not None:
            req.relativeStopLoss = 0   # use absolute
            req.stopLoss = int(sl * 100000)

        if tp is not None:
            req.relativeStopLoss = 0
            req.takeProfit = int(tp * 100000)

        try:
            resp = await self._async_send_recv(req, timeout=15)
            # Extract position ID from response
            position_id = getattr(resp, "positionId", 0) or getattr(resp, "orderId", 0)
            latest = self.get_latest_price(symbol)
            exec_price = latest["ask"] if order_type.lower() == "buy" else latest["bid"] if latest else 0.0

            record = {
                "ticket":      position_id,
                "symbol":      symbol,
                "type":        order_type.lower(),
                "volume":      volume,
                "price":       exec_price,
                "sl":          sl or 0.0,
                "tp":          tp or 0.0,
                "comment":     comment,
                "signal_prob": 0.0,
                "retcode":     0,
                "success":     True,
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "broker":      "ctrader",
            }
            self._log_trade(record)
            logger.info(
                f"[cTrader] ORDER ▶ {symbol} {order_type.upper()} {volume}L "
                f"| ticket={position_id}"
            )
            return record
        except Exception as e:
            logger.error(f"[cTrader] _async_place_order: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Close order
    # ─────────────────────────────────────────────────────────────────────────

    def close_order(self, ticket: int, symbol: str = "", volume: float = 0.0) -> bool:
        if not self.ensure_connected():
            return False
        future = asyncio.run_coroutine_threadsafe(
            self._async_close_order(ticket, symbol, volume), self._loop
        )
        try:
            return future.result(timeout=15)
        except Exception as e:
            logger.error(f"[cTrader] close_order({ticket}): {e}")
            return False

    async def _async_close_order(
        self, ticket: int, symbol: str, volume: float
    ) -> bool:
        req = ProtoOAClosePositionReq()
        req.ctidTraderAccountId = self.account_id
        req.positionId          = ticket
        if volume > 0:
            req.volume = int(volume * 100)
        try:
            await self._async_send_recv(req, timeout=15)
            logger.info(f"[cTrader] Closed position {ticket}")
            return True
        except Exception as e:
            logger.error(f"[cTrader] _async_close_order({ticket}): {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Modify order
    # ─────────────────────────────────────────────────────────────────────────

    def modify_order(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        if not self.ensure_connected():
            return False
        future = asyncio.run_coroutine_threadsafe(
            self._async_modify_order(ticket, sl, tp), self._loop
        )
        try:
            return future.result(timeout=10)
        except Exception as e:
            logger.error(f"[cTrader] modify_order({ticket}): {e}")
            return False

    async def _async_modify_order(
        self, ticket: int, sl: Optional[float], tp: Optional[float]
    ) -> bool:
        req = ProtoOAAmendPositionSLTPReq()
        req.ctidTraderAccountId = self.account_id
        req.positionId          = ticket
        if sl is not None:
            req.stopLoss           = int(sl * 100000)
            req.guaranteedStopLoss = False
        if tp is not None:
            req.takeProfit = int(tp * 100000)
        try:
            await self._async_send_recv(req, timeout=10)
            logger.info(f"[cTrader] Modified ticket={ticket} sl={sl} tp={tp}")
            return True
        except Exception as e:
            logger.error(f"[cTrader] _async_modify_order: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Account information
    # ─────────────────────────────────────────────────────────────────────────

    def get_account_info(self) -> Optional[Dict]:
        if not self.ensure_connected():
            return None
        future = asyncio.run_coroutine_threadsafe(
            self._async_get_account_info(), self._loop
        )
        try:
            return future.result(timeout=15)
        except Exception as e:
            logger.error(f"[cTrader] get_account_info: {e}")
            return self._account_info_cache  # return stale cache on error

    async def _async_get_account_info(self) -> Optional[Dict]:
        req = ProtoOATraderReq()
        req.ctidTraderAccountId = self.account_id
        try:
            resp = await self._async_send_recv(req, timeout=15)
            trader = resp.trader
            balance    = float(getattr(trader, "balance", 0)) / 100.0
            equity     = float(getattr(trader, "balance", 0)) / 100.0  # no unrealised available without reconcile
            currency   = getattr(trader, "depositAssetId", "USD")
            leverage   = int(getattr(trader, "leverageInCents", 50) / 100) or 50

            result = {
                "login":       self.account_id,
                "balance":     balance,
                "equity":      equity,
                "margin":      0.0,
                "free_margin": balance,
                "currency":    str(currency),
                "leverage":    leverage,
                "broker":      "ctrader",
            }
            # Reconcile for accurate equity
            rec_req = ProtoOAReconcileReq()
            rec_req.ctidTraderAccountId = self.account_id
            rec_resp = await self._async_send_recv(rec_req, timeout=15)
            used_margin = 0.0
            floating    = 0.0
            for pos in rec_resp.position:
                used_margin += float(getattr(pos, "usedMargin", 0)) / 100.0
                floating    += float(getattr(pos, "swap", 0)) / 100.0
            result["margin"]      = used_margin
            result["free_margin"] = balance - used_margin
            result["equity"]      = balance + floating

            self._account_info_cache = result
            return result
        except Exception as e:
            logger.error(f"[cTrader] _async_get_account_info: {e}")
            return self._account_info_cache

    # ─────────────────────────────────────────────────────────────────────────
    # Open positions
    # ─────────────────────────────────────────────────────────────────────────

    def get_open_positions(self, symbol: str = "") -> List[Dict]:
        if not self.ensure_connected():
            return []
        future = asyncio.run_coroutine_threadsafe(
            self._async_get_open_positions(symbol), self._loop
        )
        try:
            return future.result(timeout=15)
        except Exception as e:
            logger.error(f"[cTrader] get_open_positions: {e}")
            return []

    async def _async_get_open_positions(self, symbol: str) -> List[Dict]:
        req = ProtoOAReconcileReq()
        req.ctidTraderAccountId = self.account_id
        try:
            resp = await self._async_send_recv(req, timeout=15)
            result = []
            for pos in resp.position:
                sym_name = self._resolve_sym_name(pos.tradeData.symbolId)
                if symbol and sym_name.upper() != symbol.upper():
                    continue
                pos_type = "buy" if pos.tradeData.tradeSide == ProtoOATradeSide.BUY else "sell"
                vol      = float(pos.tradeData.volume) / 100.0
                result.append({
                    "ticket":     pos.positionId,
                    "symbol":     sym_name,
                    "type":       pos_type,
                    "volume":     vol,
                    "open_price": float(pos.price) / 100000.0,
                    "current":    0.0,
                    "sl":         float(getattr(pos, "stopLoss", 0)) / 100000.0,
                    "tp":         float(getattr(pos, "takeProfit", 0)) / 100000.0,
                    "profit":     float(getattr(pos, "swap", 0)) / 100.0,
                    "magic":      MAGIC_NUMBER,
                    "comment":    getattr(pos.tradeData, "comment", ""),
                    "broker":     "ctrader",
                })
            return result
        except Exception as e:
            logger.error(f"[cTrader] _async_get_open_positions: {e}")
            return []

    def _resolve_sym_name(self, symbol_id: int) -> str:
        """Reverse lookup symbolId → symbol name."""
        for name, sid in self._symbols_map.items():
            if sid == symbol_id:
                return name
        return str(symbol_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Trade history
    # ─────────────────────────────────────────────────────────────────────────

    def get_trade_history(self, days: int = 365, symbol: str = "") -> List[Dict]:
        """
        Pull closed deal history from cTrader via ProtoOADealListReq.  (v20)

        Was a stub returning []. Now fully implemented with date pagination
        (cTrader limits each request to 1 week, so we paginate over `days`).
        PnL is taken directly from deal.closePositionDetail.grossProfit.
        365d default (was 7d) so TradeHistoryLearner gets full context.
        """
        if not self.ensure_connected():
            return []
        if not CTRADER_AVAILABLE:
            return []

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._async_get_deal_history(days=days, symbol=symbol),
                self._loop,
            )
            return future.result(timeout=60)
        except Exception as e:
            logger.error(f"[cTrader] get_trade_history: {e}")
            return []

    async def _async_get_deal_history(
        self, days: int = 365, symbol: str = ""
    ) -> List[Dict]:
        """
        cTrader ProtoOADealListReq paginated deal pull.
        API allows max 604800000 ms (7 days) per request — we loop weekly chunks.
        """
        try:
            # Dynamic import — only needed here
            try:
                from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                    ProtoOADealListReq,
                )
            except ImportError:
                logger.warning("[cTrader] ProtoOADealListReq not available in installed version")
                return []

            end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
            start_ms = end_ms - int(days * 86400 * 1000)
            WEEK_MS  = 7 * 24 * 3600 * 1000   # 7-day chunks (API limit)

            all_deals = []
            chunk_start = start_ms
            while chunk_start < end_ms:
                chunk_end = min(chunk_start + WEEK_MS, end_ms)
                req = ProtoOADealListReq()
                req.ctidTraderAccountId = self.account_id
                req.fromTimestamp = chunk_start
                req.toTimestamp   = chunk_end
                try:
                    resp = await self._async_send_recv(req, timeout=20)
                    if hasattr(resp, "deal"):
                        all_deals.extend(resp.deal)
                except Exception as chunk_err:
                    logger.debug(f"[cTrader] deal chunk error: {chunk_err}")
                chunk_start = chunk_end

            result = []
            for deal in all_deals:
                # Only closing deals have gross profit
                if not hasattr(deal, "closePositionDetail"):
                    continue
                cpd = deal.closePositionDetail
                sym_name = self._resolve_sym_name(deal.symbolId)
                if symbol and sym_name.upper() != symbol.upper():
                    continue

                # cTrader stores prices as integers × 100000
                close_price = float(deal.executionPrice) / 100000.0
                open_price  = float(cpd.entryPrice) / 100000.0 if hasattr(cpd, "entryPrice") else 0.0
                pnl         = float(cpd.grossProfit) / 100.0   # stored in cents
                volume      = float(deal.volume) / 100.0

                side = "buy" if getattr(deal, "tradeSide", 1) == ProtoOATradeSide.BUY else "sell"

                close_time = datetime.fromtimestamp(
                    deal.executionTimestamp / 1000, tz=timezone.utc
                ).isoformat() if hasattr(deal, "executionTimestamp") else ""

                open_time = datetime.fromtimestamp(
                    cpd.entryTimestamp / 1000, tz=timezone.utc
                ).isoformat() if hasattr(cpd, "entryTimestamp") else ""

                result.append({
                    "ticket":      deal.dealId,
                    "symbol":      sym_name,
                    "type":        side,
                    "volume":      volume,
                    "open_price":  round(open_price, 6),
                    "close_price": round(close_price, 6),
                    "profit":      round(pnl, 4),
                    "open_time":   open_time,
                    "close_time":  close_time,
                    "broker":      "ctrader",
                })

            logger.info(
                f"[cTrader] get_trade_history: {len(result)} deals (last {days}d)"
            )
            return result

        except Exception as e:
            logger.error(f"[cTrader] _async_get_deal_history: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # Compatibility helpers
    # ─────────────────────────────────────────────────────────────────────────

    def get_candles(self, symbol: str, timeframe: str, bars: int = 500) -> Optional[pd.DataFrame]:
        """Alias for get_market_data."""
        return self.get_market_data(symbol, timeframe, bars)

    def count_open_positions(self) -> int:
        return len(self.get_open_positions())

    def update_sl(self, ticket: int, new_sl: float) -> bool:
        return self.modify_order(ticket, sl=new_sl)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if len(df) < 20:
                return df
            df["body"]       = df["close"] - df["open"]
            df["range"]      = df["high"]  - df["low"]
            df["body_pct"]   = df["body"].abs() / df["range"].replace(0, np.nan)
            df["sma20"]      = df["close"].rolling(20).mean()
            df["ema50"]      = df["close"].ewm(span=50,  adjust=False).mean()
            df["ema200"]     = df["close"].ewm(span=200, adjust=False).mean()
            delta = df["close"].diff()
            gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
            df["rsi"]        = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
            fast = df["close"].ewm(span=12, adjust=False).mean()
            slow = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"]        = fast - slow
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            hl = df["high"] - df["low"]
            hc = (df["high"] - df["close"].shift()).abs()
            lc = (df["low"]  - df["close"].shift()).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            df["atr"]        = tr.ewm(alpha=1/14, adjust=False).mean()
            df["volatility"] = df["close"].pct_change().rolling(14).std() * 100
            if "tick_volume" in df.columns and df["tick_volume"].sum() > 0:
                df["volume_ma"]    = df["tick_volume"].rolling(20).mean()
                df["volume_ratio"] = df["tick_volume"] / df["volume_ma"].replace(0, np.nan)
        except Exception as e:
            logger.debug(f"[cTrader] _add_indicators error: {e}")
        return df

    @staticmethod
    def _log_trade(record: Dict) -> None:
        try:
            os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
            with open(TRADE_LOG_PATH, "a", encoding="utf-8") as f:
                import json
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"[cTrader] Trade log write failed: {e}")

    def __repr__(self) -> str:
        return (
            f"<CTraderAdapter account={self.account_id} "
            f"connected={self.connected} demo={self.demo}>"
        )
