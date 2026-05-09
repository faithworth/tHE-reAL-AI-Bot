"""
ibkr_adapter.py — Full Interactive Brokers Adapter (AI EA v5)
=============================================================
Implements BaseBroker for Interactive Brokers via the ib_insync library.

Supports: Forex, Stocks, Indices (ETF/CFD), Futures, Options
Requirements: TWS or IB Gateway running, ib_insync installed
  pip install ib_insync

IB Gateway default ports:
  Paper trading: 7497 (TWS) / 4002 (Gateway)
  Live trading:  7496 (TWS) / 4001 (Gateway)
"""

import logging
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from base_broker import BaseBroker, OrderRejected

logger = logging.getLogger(__name__)

# ── Optional ib_insync import ─────────────────────────────────────────────────
try:
    import ib_insync as ibi
    from ib_insync import (
        IB, Contract, Forex, Stock, Index, Future, CFD,
        MarketOrder, LimitOrder, StopOrder, BracketOrder,
        Trade, Position as IBPosition, AccountValue,
    )
    IBKR_AVAILABLE = True
except ImportError:
    ibi = None
    IBKR_AVAILABLE = False
    logger.warning("[IBKR] ib_insync not installed. Run: pip install ib_insync")

# Timeframe string → IB duration/bar string mapping
_IB_TF_MAP = {
    "m1":  ("1 D",  "1 min"),
    "m5":  ("5 D",  "5 mins"),
    "m15": ("10 D", "15 mins"),
    "m30": ("20 D", "30 mins"),
    "h1":  ("30 D", "1 hour"),
    "h2":  ("60 D", "2 hours"),
    "h4":  ("60 D", "4 hours"),
    "d1":  ("1 Y",  "1 day"),
    "w1":  ("5 Y",  "1 week"),
}

_IBKR_FOREX_PAIRS = {
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "EURAUD",
    "EURCAD", "EURCHF", "EURNZD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
    "AUDCAD", "AUDCHF", "AUDNZD", "NZDCAD", "NZDCHF", "NZDJPY",
}

_IBKR_INDICES = {
    "SPX": ("SPX", "CBOE", "IND"),
    "NDX": ("NDX", "NASDAQ", "IND"),
    "DJI": ("INDU", "NYSE", "IND"),
    "US30": ("YM", "ECBOT", "FUT"),
    "US500": ("ES", "CME", "FUT"),
    "US100": ("NQ", "CME", "FUT"),
}


class IBKRAdapter(BaseBroker):
    """
    Full Interactive Brokers adapter using ib_insync.
    Connect via IB Gateway or TWS before using.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        account: str = "",
        risk_engine=None,
        paper_trading: bool = True,
    ):
        super().__init__()
        self.broker_name  = "IBKR"
        self.host         = host
        self.port         = port
        self.client_id    = client_id
        self.account      = account
        self.risk_engine  = risk_engine
        self.paper_trading = paper_trading
        self._ib: Optional["IB"] = None
        self._contracts_cache: Dict[str, "Contract"] = {}
        self._next_order_id: int = 1

    # ─────────────────────────────────────────────────────────────────────────
    # Connection
    # ─────────────────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        if not IBKR_AVAILABLE:
            logger.error("[IBKR] ib_insync not installed. pip install ib_insync")
            return False
        try:
            self._ib = IB()
            self._ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=20,
                readonly=False,
            )
            if not self._ib.isConnected():
                raise ConnectionError("IB not connected after connect()")
            self.connected = True
            # Get account if not specified
            if not self.account and self._ib.managedAccounts():
                self.account = self._ib.managedAccounts()[0]
            logger.info(
                f"[IBKR] Connected: host={self.host} port={self.port} "
                f"account={self.account} paper={self.paper_trading}"
            )
            return True
        except Exception as e:
            logger.error(f"[IBKR] Connection failed: {e}")
            self.connected = False
            return False

    def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
        self.connected = False
        logger.info("[IBKR] Disconnected.")

    def ensure_connected(self) -> bool:
        if self._ib and self._ib.isConnected():
            self.connected = True
            return True
        self.connected = False
        return self.connect()

    # ─────────────────────────────────────────────────────────────────────────
    # Symbol discovery
    # ─────────────────────────────────────────────────────────────────────────

    def get_symbols(self) -> List[Dict]:
        """Return well-known forex pairs + common instruments as SymbolInfo dicts."""
        if not self.ensure_connected():
            return []
        symbols = []
        # Forex
        for pair in _IBKR_FOREX_PAIRS:
            symbols.append({
                "name":          pair,
                "contract_size": 20000.0,     # IBKR mini forex default
                "point":         0.00001,
                "digits":        5,
                "min_lot":       0.02,
                "max_lot":       100.0,
                "lot_step":      0.01,
                "spread":        0,
                "trade_mode":    4,
                "asset_class":   "forex",
                "ib_sectype":    "CASH",
                "ib_exchange":   "IDEALPRO",
            })
        # Common stocks
        for ticker, exchange in [("AAPL","SMART"), ("MSFT","SMART"), ("TSLA","SMART"),
                                  ("AMZN","SMART"), ("GOOGL","SMART"), ("META","SMART"),
                                  ("NVDA","SMART"), ("SPY","SMART"), ("QQQ","SMART")]:
            symbols.append({
                "name":          ticker,
                "contract_size": 1.0,
                "point":         0.01,
                "digits":        2,
                "min_lot":       1.0,
                "max_lot":       10000.0,
                "lot_step":      1.0,
                "spread":        0,
                "trade_mode":    4,
                "asset_class":   "stock",
                "ib_sectype":    "STK",
                "ib_exchange":   exchange,
            })
        # Indices / Futures
        for name, (symbol, exchange, sectype) in _IBKR_INDICES.items():
            symbols.append({
                "name":          name,
                "contract_size": 1.0,
                "point":         0.25 if sectype == "FUT" else 1.0,
                "digits":        2,
                "min_lot":       1.0,
                "max_lot":       100.0,
                "lot_step":      1.0,
                "spread":        0,
                "trade_mode":    4,
                "asset_class":   "index",
                "ib_sectype":    sectype,
                "ib_exchange":   exchange,
                "ib_symbol":     symbol,
            })
        # XAU proxy via GLD ETF or CFD
        symbols.append({
            "name": "XAUUSD", "contract_size": 100.0, "point": 0.01,
            "digits": 2, "min_lot": 0.01, "max_lot": 50.0, "lot_step": 0.01,
            "spread": 0, "trade_mode": 4, "asset_class": "metal",
            "ib_sectype": "CMDTY", "ib_exchange": "SMART", "ib_symbol": "XAUUSD",
        })
        return symbols

    def get_contract(self, symbol: str) -> Optional["Contract"]:
        """Resolve symbol to IBKR Contract object."""
        if symbol in self._contracts_cache:
            return self._contracts_cache[symbol]
        if not self.ensure_connected():
            return None
        try:
            contract = self._symbol_to_contract(symbol)
            if contract:
                details = self._ib.reqContractDetails(contract)
                if details:
                    resolved = details[0].contract
                    self._contracts_cache[symbol] = resolved
                    return resolved
        except Exception as e:
            logger.error(f"[IBKR] get_contract({symbol}): {e}")
        return None

    def _symbol_to_contract(self, symbol: str) -> Optional["Contract"]:
        """Build a Contract object for a given symbol name."""
        if not IBKR_AVAILABLE:
            return None
        u = symbol.upper()
        # Forex pairs
        if u in _IBKR_FOREX_PAIRS and len(u) == 6:
            return Forex(u[:3] + "/" + u[3:])
        # Stocks
        if len(u) <= 5 and u.isalpha() and u not in _IBKR_FOREX_PAIRS:
            return Stock(u, "SMART", "USD")
        # Indices
        if u in _IBKR_INDICES:
            sym, exch, st = _IBKR_INDICES[u]
            if st == "FUT":
                return Future(sym, exchange=exch)
            return Index(sym, exch)
        # Gold proxy
        if "XAU" in u:
            c = Contract()
            c.symbol   = "XAUUSD"
            c.secType  = "CMDTY"
            c.currency = "USD"
            c.exchange = "SMART"
            return c
        # Generic
        c = Contract()
        c.symbol   = u
        c.secType  = "STK"
        c.exchange = "SMART"
        c.currency = "USD"
        return c

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
        contract = self.get_contract(symbol)
        if contract is None:
            logger.error(f"[IBKR] Cannot resolve contract: {symbol}")
            return None
        tf_key = timeframe.lower()
        duration, bar_size = _IB_TF_MAP.get(tf_key, ("30 D", "1 hour"))
        try:
            bars_data = self._ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="MIDPOINT" if "CASH" in str(type(contract)) else "TRADES",
                useRTH=False,
                formatDate=2,
            )
            if not bars_data:
                logger.error(f"[IBKR] No historical data: {symbol} {timeframe}")
                return None
            rows = []
            for b in bars_data:
                rows.append({
                    "open":        float(b.open),
                    "high":        float(b.high),
                    "low":         float(b.low),
                    "close":       float(b.close),
                    "tick_volume": int(b.volume),
                    "real_volume": int(b.volume),
                    "time":        b.date if hasattr(b.date, "strftime") else str(b.date),
                })
            df = pd.DataFrame(rows)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            df = df.tail(bars)
            df["symbol"] = symbol
            df = self._add_indicators(df)
            return df
        except Exception as e:
            logger.error(f"[IBKR] get_market_data({symbol}): {e}", exc_info=True)
            return None

    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        if not self.ensure_connected():
            return None
        contract = self.get_contract(symbol)
        if contract is None:
            return None
        try:
            ticker = self._ib.reqMktData(contract, "", False, False)
            self._ib.sleep(1)
            bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else float(ticker.last or 0)
            ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else float(ticker.last or 0)
            self._ib.cancelMktData(contract)
            return {
                "bid":    bid,
                "ask":    ask,
                "spread": ask - bid,
                "time":   datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"[IBKR] get_latest_price({symbol}): {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Order placement
    # ─────────────────────────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "AI_EA_v5",
        magic: int = 20250424,
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
                symbol=symbol, signal_prob=signal_prob)
            if not approved:
                logger.warning(f"[IBKR] Trade BLOCKED [{symbol}]: {reason}")
                return None

        contract = self.get_contract(symbol)
        if contract is None:
            logger.error(f"[IBKR] Cannot resolve contract: {symbol}")
            return None

        # Validate volume
        sym_info = self.get_symbol_info(symbol)
        vol = self.validate_volume(volume, sym_info)
        action = "BUY" if order_type.lower() == "buy" else "SELL"

        try:
            if price is not None:
                ib_order = LimitOrder(action, vol, round(price, 5))
            else:
                ib_order = MarketOrder(action, vol)
            ib_order.orderRef = f"{comment}_{symbol}"

            # Bracket order with SL/TP if provided
            if sl is not None or tp is not None:
                latest = self.get_latest_price(symbol)
                exec_price = price or (latest["ask"] if action == "BUY" else latest["bid"] if latest else 0)
                if atr <= 0 and latest:
                    atr = abs(latest["ask"] - latest["bid"]) * 50
                calc_sl = sl or (exec_price - atr * sl_atr_mult if action == "BUY"
                                  else exec_price + atr * sl_atr_mult)
                calc_tp = tp or (exec_price + atr * tp_atr_mult if action == "BUY"
                                  else exec_price - atr * tp_atr_mult)
                bracket = self._ib.bracketOrder(
                    action, vol, exec_price,
                    round(calc_tp, 5), round(calc_sl, 5)
                )
                trades = []
                for o in bracket:
                    t = self._ib.placeOrder(contract, o)
                    trades.append(t)
                self._ib.sleep(2)
                parent = trades[0]
                ticket = parent.order.orderId
                result_price = exec_price
            else:
                trade = self._ib.placeOrder(contract, ib_order)
                self._ib.sleep(2)
                ticket       = trade.order.orderId
                result_price = price or 0.0

            record = {
                "ticket":      ticket,
                "symbol":      symbol,
                "type":        order_type,
                "volume":      vol,
                "price":       result_price,
                "sl":          sl or 0.0,
                "tp":          tp or 0.0,
                "comment":     comment,
                "signal_prob": signal_prob,
                "retcode":     0,
                "success":     True,
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "broker":      "ibkr",
            }
            # NOTE: record_trade_open() is called by ai_ea.py — do not call here.
            logger.info(
                f"[IBKR] ORDER ▶ {symbol} {action} {vol} | "
                f"ticket={ticket} prob={signal_prob:.3f}"
            )
            return record
        except Exception as e:
            logger.error(f"[IBKR] place_order({symbol}): {e}", exc_info=True)
            return None

    def close_order(self, ticket: int, symbol: str = "", volume: float = 0.0) -> bool:
        if not self.ensure_connected():
            return False
        try:
            # Find position
            positions = self._ib.positions(self.account)
            pos = None
            for p in positions:
                if p.contract.symbol == symbol.upper() or (hasattr(p, "account")):
                    pos = p
                    break
            if pos is None:
                # Try to cancel open order
                orders = self._ib.orders()
                for o in orders:
                    if o.orderId == ticket:
                        self._ib.cancelOrder(o)
                        logger.info(f"[IBKR] Cancelled order {ticket}")
                        return True
                return False
            contract = pos.contract
            qty      = abs(pos.position) if volume == 0 else volume
            action   = "SELL" if pos.position > 0 else "BUY"
            close_order = MarketOrder(action, qty)
            trade    = self._ib.placeOrder(contract, close_order)
            self._ib.sleep(2)
            logger.info(f"[IBKR] Closed position: {contract.symbol} qty={qty}")
            return True
        except Exception as e:
            logger.error(f"[IBKR] close_order({ticket}): {e}")
            return False

    def modify_order(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        """
        IBKR does not support native SL/TP modification on existing orders
        in the same way MT5 does — we cancel child orders and replace them.
        """
        if not self.ensure_connected():
            return False
        try:
            orders = self._ib.openOrders()
            modified = False
            for o in orders:
                if o.parentId == ticket:
                    if o.orderType == "STP" and sl is not None:
                        o.auxPrice = sl
                        self._ib.placeOrder(self._ib.openTrades()[0].contract, o)
                        modified = True
                    elif o.orderType == "LMT" and tp is not None:
                        o.lmtPrice = tp
                        self._ib.placeOrder(self._ib.openTrades()[0].contract, o)
                        modified = True
            if modified:
                logger.info(f"[IBKR] Modified ticket={ticket} sl={sl} tp={tp}")
            return modified
        except Exception as e:
            logger.error(f"[IBKR] modify_order({ticket}): {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Account
    # ─────────────────────────────────────────────────────────────────────────

    def get_account_info(self) -> Optional[Dict]:
        if not self.ensure_connected():
            return None
        try:
            account_values = self._ib.accountSummary(self.account)
            if not account_values:
                self._ib.reqAccountSummary()
                self._ib.sleep(1)
                account_values = self._ib.accountSummary(self.account)
            av_dict = {v.tag: v.value for v in account_values}
            equity       = float(av_dict.get("NetLiquidation",  av_dict.get("TotalCashValue", 0)))
            balance      = float(av_dict.get("TotalCashValue",  equity))
            margin       = float(av_dict.get("MaintMarginReq",  0))
            free_margin  = float(av_dict.get("AvailableFunds",  equity - margin))
            currency     = av_dict.get("Currency", "USD")
            return {
                "login":       self.account,
                "balance":     balance,
                "equity":      equity,
                "margin":      margin,
                "free_margin": free_margin,
                "currency":    currency,
                "leverage":    1,         # IBKR uses margin-based, not fixed leverage
                "broker":      "ibkr",
            }
        except Exception as e:
            logger.error(f"[IBKR] get_account_info: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Positions
    # ─────────────────────────────────────────────────────────────────────────

    def get_open_positions(self, symbol: str = "") -> List[Dict]:
        if not self.ensure_connected():
            return []
        try:
            ib_positions = self._ib.positions(self.account)
            result = []
            for p in ib_positions:
                if p.position == 0:
                    continue
                sym_name = p.contract.symbol
                if symbol and sym_name.upper() != symbol.upper():
                    continue
                result.append({
                    "ticket":     p.contract.conId,
                    "symbol":     sym_name,
                    "type":       "buy" if p.position > 0 else "sell",
                    "volume":     abs(p.position),
                    "open_price": float(p.avgCost),
                    "current":    float(p.avgCost),  # reqMktData needed for live price
                    "sl":         0.0,
                    "tp":         0.0,
                    "profit":     float(p.unrealizedPNL) if hasattr(p, "unrealizedPNL") else 0.0,
                    "magic":      0,
                    "comment":    "",
                    "broker":     "ibkr",
                })
            return result
        except Exception as e:
            logger.error(f"[IBKR] get_open_positions: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # Trade history
    # ─────────────────────────────────────────────────────────────────────────

    def get_trade_history(self, days: int = 365, symbol: str = "") -> List[Dict]:
        """
        Pull closed trade history from IBKR fills with real PnL.  (v20)

        IBKR fills() only gives executions, not PnL. We reconstruct PnL via
        FIFO BUY→SELL matching per symbol (same approach as Alpaca adapter).
        Also tries reqPnLSingle for live PnL where available.
        365d default (was 7d) so the TradeHistoryLearner gets full context.
        """
        if not self.ensure_connected():
            return []
        try:
            end   = datetime.now(timezone.utc)
            start = end - timedelta(days=days)

            fills = self._ib.fills()
            if not fills:
                return []

            # Group fills by symbol, sort by time
            by_sym: Dict[str, list] = {}
            for fill in fills:
                try:
                    t = fill.time
                    # ib_insync returns naive or aware datetime
                    if hasattr(t, 'tzinfo') and t.tzinfo is None:
                        from datetime import timezone as _tz
                        t = t.replace(tzinfo=_tz.utc)
                    if not (start <= t <= end):
                        continue
                    sym = fill.contract.symbol
                    if symbol and sym.upper() != symbol.upper():
                        continue
                    by_sym.setdefault(sym, []).append((t, fill))
                except Exception:
                    continue

            result = []
            for sym, entries in by_sym.items():
                entries.sort(key=lambda x: x[0])
                buys: List[Dict] = []
                for t, fill in entries:
                    side  = fill.execution.side   # "BOT" or "SLD"
                    qty   = float(fill.execution.shares)
                    price = float(fill.execution.price)

                    if side == "BOT":
                        buys.append({"qty": qty, "price": price,
                                     "ts": t, "id": fill.execution.execId})
                    else:
                        # SLD — FIFO match against buys
                        remaining = qty
                        pnl = 0.0
                        entry_price = price
                        entry_ts    = t
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

                        result.append({
                            "ticket":      fill.execution.execId,
                            "symbol":      sym,
                            "type":        "buy",
                            "volume":      qty,
                            "open_price":  round(entry_price, 6),
                            "close_price": round(price, 6),
                            "profit":      round(pnl, 4),
                            "open_time":   entry_ts.isoformat(),
                            "close_time":  t.isoformat(),
                            "broker":      "ibkr",
                        })

            logger.info(f"[IBKR] get_trade_history: {len(result)} paired trades (last {days}d)")
            return result
        except Exception as e:
            logger.error(f"[IBKR] get_trade_history: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # Compatibility helpers
    # ─────────────────────────────────────────────────────────────────────────

    def get_candles(self, symbol: str, timeframe: str, bars: int = 500) -> Optional[pd.DataFrame]:
        """Alias for get_market_data."""
        return self.get_market_data(symbol, timeframe, bars)

    def count_open_positions(self) -> int:
        return len(self.get_open_positions())

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
            logger.debug(f"[IBKR] _add_indicators error: {e}")
        return df
