"""
executor.py — Rebuilt MT5 execution engine (AI EA v4)
"""
import json, logging, os, time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from risk_engine import RiskEngine

logger = logging.getLogger(__name__)

TRADE_LOG_PATH       = "data/trade_log.jsonl"
MAX_CONNECT_ATTEMPTS = 5
CONNECT_RETRY_DELAY  = 5
MAX_ORDER_RETRIES    = 3
ORDER_RETRY_DELAY    = 1
DEVIATION_POINTS     = 20


class MT5Executor:
    def __init__(self, login: int, password: str, server: str,
                 risk_engine: Optional[RiskEngine] = None):
        self.login = login
        self.password = password
        self.server = server
        self.risk_engine = risk_engine or RiskEngine()
        self.connected = False
        os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
        self.connect()

    # ── connection ──────────────────────────────────────────────────────
    def connect(self) -> bool:
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 not available — mock mode.")
            return False
        for attempt in range(1, MAX_CONNECT_ATTEMPTS + 1):
            try:
                if not mt5.initialize():
                    raise ConnectionError(f"initialize failed: {mt5.last_error()}")
                if not mt5.login(self.login, self.password, self.server):
                    raise ConnectionError(f"login failed: {mt5.last_error()}")
                self.connected = True
                logger.info(f"Connected to MT5: {self.server} account={self.login}")
                return True
            except Exception as e:
                logger.warning(f"Connect attempt {attempt}/{MAX_CONNECT_ATTEMPTS}: {e}")
                time.sleep(CONNECT_RETRY_DELAY)
        logger.critical("All MT5 connect attempts failed.")
        return False

    def ensure_connected(self) -> bool:
        if self.connected and MT5_AVAILABLE:
            try:
                if mt5.account_info() is not None:
                    return True
            except Exception:
                pass
        self.connected = False
        return self.connect()

    def close_connection(self) -> None:
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False

    # ── account ─────────────────────────────────────────────────────────
    def get_account_info(self) -> Optional[Dict]:
        if not self.ensure_connected():
            return None
        i = mt5.account_info()
        if i is None:
            return None
        return {"login": i.login, "balance": i.balance, "equity": i.equity,
                "margin": i.margin, "free_margin": i.margin_free,
                "margin_level": i.margin_level, "currency": i.currency}

    def get_equity(self) -> float:
        info = self.get_account_info()
        return info["equity"] if info else 0.0

    # ── positions ────────────────────────────────────────────────────────
    def get_position(self, symbol: str):
        if not self.ensure_connected(): return None
        p = mt5.positions_get(symbol=symbol)
        return p[0] if p else None

    def get_all_positions(self) -> List:
        if not self.ensure_connected(): return []
        p = mt5.positions_get()
        return list(p) if p else []

    def count_open_positions(self) -> int:
        return len(self.get_all_positions())

    # ── order placement ──────────────────────────────────────────────────
    def place_order(self, symbol: str, lot: float, order_type: str,
                    atr: float = 0.0, sl_atr_mult: float = 1.5,
                    tp_atr_mult: float = 2.5, signal_prob: float = 0.65,
                    comment: str = "AI_EA_v4") -> Optional[Dict]:
        if not self.ensure_connected():
            return None

        equity   = self.get_equity()
        open_cnt = self.count_open_positions()
        approved, reason = self.risk_engine.approve_trade(
            equity=equity, open_positions=open_cnt,
            symbol=symbol, signal_prob=signal_prob)
        if not approved:
            logger.warning(f"Trade BLOCKED [{symbol}]: {reason}")
            return None

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            logger.error(f"Symbol {symbol} not found"); return None
        if not sym_info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Cannot select {symbol}"); return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"No tick for {symbol}"); return None

        # spread validation
        spread_pips = (tick.ask - tick.bid) / sym_info.point / 10
        max_sp = 50.0
        if any(x in symbol.upper() for x in ["BTC","ETH"]): max_sp = 400.0
        elif any(x in symbol.upper() for x in ["XAU","GOLD"]): max_sp = 200.0
        elif "OIL" in symbol.upper(): max_sp = 100.0
        if spread_pips > max_sp:
            logger.warning(f"Spread too wide [{symbol}]: {spread_pips:.1f}>{max_sp} pips")
            return None

        # margin validation
        acct = self.get_account_info()
        margin_req = mt5.order_calc_margin(
            mt5.ORDER_TYPE_BUY if order_type=="buy" else mt5.ORDER_TYPE_SELL,
            symbol, lot, tick.ask)
        if margin_req is None or (acct and margin_req > acct["free_margin"] * 0.9):
            logger.error(f"Insufficient margin for {symbol} {lot} lots")
            return None

        digits = sym_info.digits
        point  = sym_info.point
        if atr <= 0:
            atr = (tick.ask - tick.bid) * 50

        if order_type.lower() == "buy":
            price    = round(tick.ask, digits)
            sl       = round(price - atr * sl_atr_mult, digits)
            tp       = round(price + atr * tp_atr_mult, digits)
            mt5_type = mt5.ORDER_TYPE_BUY
        else:
            price    = round(tick.bid, digits)
            sl       = round(price + atr * sl_atr_mult, digits)
            tp       = round(price - atr * tp_atr_mult, digits)
            mt5_type = mt5.ORDER_TYPE_SELL

        min_stop = sym_info.trade_stops_level * point
        if order_type == "buy":
            sl = min(sl, round(price - max(min_stop + point, atr * sl_atr_mult), digits))
            tp = max(tp, round(price + max(min_stop + point, atr * tp_atr_mult), digits))
        else:
            sl = max(sl, round(price + max(min_stop + point, atr * sl_atr_mult), digits))
            tp = min(tp, round(price - max(min_stop + point, atr * tp_atr_mult), digits))

        # Detect broker-supported filling mode from live symbol info.
        # Many brokers (incl. GTio Markets) use RETURN or FOK, not IOC.
        filling_mode = mt5.ORDER_FILLING_IOC   # safe default
        try:
            fm_bits = getattr(sym_info, "filling_mode", 0)
            FOK    = getattr(mt5, "ORDER_FILLING_FOK",    0)
            IOC    = getattr(mt5, "ORDER_FILLING_IOC",    1)
            RETURN = getattr(mt5, "ORDER_FILLING_RETURN", 2)
            if fm_bits & RETURN:
                filling_mode = RETURN
            elif fm_bits & IOC:
                filling_mode = IOC
            elif fm_bits & FOK:
                filling_mode = FOK
        except Exception:
            pass

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       lot,
            "type":         mt5_type,
            "price":        price,
            "sl":           sl,
            "tp":           tp,
            "deviation":    DEVIATION_POINTS,
            "magic":        20250424,
            "comment":      f"{comment}_p{signal_prob:.2f}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        result = self._send_with_retry(request)
        if result is None:
            return None

        record = {
            "timestamp": datetime.now().isoformat(), "symbol": symbol,
            "type": order_type, "lot": lot, "price": price, "sl": sl, "tp": tp,
            "spread_pips": round(spread_pips, 2), "signal_prob": signal_prob,
            "ticket": result.order, "atr": atr,
        }
        self._log_trade(record)
        # record_trade_open() is called by ai_ea.py after place_order() succeeds
        logger.info(f"ORDER ▶ {symbol} {order_type.upper()} {lot}L | "
                    f"price={price} sl={sl} tp={tp} | ticket={result.order} | "
                    f"spread={spread_pips:.1f}pip | prob={signal_prob:.3f}")
        return record

    def _send_with_retry(self, request: Dict):
        for attempt in range(1, MAX_ORDER_RETRIES + 1):
            result = mt5.order_send(request)
            if result is None:
                time.sleep(ORDER_RETRY_DELAY); continue
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return result
            if result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                tick = mt5.symbol_info_tick(request["symbol"])
                if tick:
                    request["price"] = tick.ask if request["type"]==mt5.ORDER_TYPE_BUY else tick.bid
                time.sleep(ORDER_RETRY_DELAY); continue
            logger.error(f"Order failed retcode={result.retcode}: {result.comment}")
            return None
        return None

    # ── position management ──────────────────────────────────────────────
    def close_position(self, ticket: int) -> bool:
        if not self.ensure_connected(): return False
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.error(f"Position {ticket} not found"); return False
        pos = positions[0]
        close_type = mt5.ORDER_TYPE_BUY if pos.type==mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None: return False
        price = tick.ask if close_type==mt5.ORDER_TYPE_BUY else tick.bid
        close_sym_info = mt5.symbol_info(pos.symbol)
        close_filling  = mt5.ORDER_FILLING_IOC
        try:
            fm_bits = getattr(close_sym_info, "filling_mode", 0)
            FOK    = getattr(mt5, "ORDER_FILLING_FOK",    0)
            IOC    = getattr(mt5, "ORDER_FILLING_IOC",    1)
            RETURN = getattr(mt5, "ORDER_FILLING_RETURN", 2)
            if fm_bits & RETURN: close_filling = RETURN
            elif fm_bits & IOC:  close_filling = IOC
            elif fm_bits & FOK:  close_filling = FOK
        except Exception:
            pass
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol,
            "volume": pos.volume, "type": close_type, "position": ticket,
            "price": price, "deviation": DEVIATION_POINTS,
            "magic": 20250424, "comment": "AI_EA_v4_close",
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": close_filling,
        }
        result = self._send_with_retry(request)
        if result:
            equity = self.get_equity()
            self.risk_engine.record_trade_close(pos.profit, equity)
            self._log_trade({"timestamp": datetime.now().isoformat(), "event": "close",
                             "ticket": ticket, "symbol": pos.symbol,
                             "profit": pos.profit, "equity_after": equity})
            logger.info(f"Closed ticket={ticket} | P&L={pos.profit:.2f}")
            return True
        return False

    def update_sl(self, ticket: int, new_sl: float) -> bool:
        if not self.ensure_connected(): return False
        positions = mt5.positions_get(ticket=ticket)
        if not positions: return False
        pos = positions[0]
        r = mt5.order_send({
            "action": mt5.TRADE_ACTION_SLTP, "position": ticket,
            "sl": new_sl, "tp": pos.tp, "symbol": pos.symbol,
            "deviation": DEVIATION_POINTS,
        })
        ok = r and r.retcode == mt5.TRADE_RETCODE_DONE
        if ok: logger.info(f"SL updated ticket={ticket} → {new_sl:.5f}")
        else:  logger.error(f"SL update failed ticket={ticket}: {r.comment if r else 'None'}")
        return ok

    # ── history ───────────────────────────────────────────────────────────
    def get_trade_history(self, days: int = 7, symbol: Optional[str] = None) -> List[Dict]:
        if not self.ensure_connected(): return []
        end = datetime.now()
        start = end - timedelta(days=days)
        deals = mt5.history_deals_get(start, end)
        if not deals: return []
        out = []
        for d in deals:
            if d.entry != mt5.DEAL_ENTRY_OUT: continue
            if symbol and d.symbol != symbol: continue
            out.append({"ticket": d.ticket, "symbol": d.symbol,
                        "time": datetime.fromtimestamp(d.time).isoformat(),
                        "type": "buy" if d.type==mt5.DEAL_TYPE_BUY else "sell",
                        "volume": d.volume, "price": d.price,
                        "profit": d.profit, "comment": d.comment})
        return out

    # ── pip helper ────────────────────────────────────────────────────────
    @staticmethod
    def get_pip_multiplier(symbol: str) -> float:
        s = symbol.upper()
        if "XAU" in s: return 0.01
        if "BTC" in s or "ETH" in s: return 1.0
        if "OIL" in s: return 0.001
        if "JPY" in s: return 0.01
        return 0.0001

    # ── logging ──────────────────────────────────────────────────────────
    def _log_trade(self, record: Dict) -> None:
        try:
            with open(TRADE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Trade log write failed: {e}")


# =============================================================================
# BrokerExecutor — Universal execution wrapper around BaseBroker
# =============================================================================
# Provides the same interface as MT5Executor but works with ANY broker adapter
# (MT5Adapter, IBKRAdapter, CTraderAdapter) via the BaseBroker interface.
# Use this class everywhere ai_ea.py / other modules need execution without
# caring which broker is active.
# =============================================================================

import logging as _exec_log
from typing import Dict, List, Optional

_bex_logger = _exec_log.getLogger(__name__ + ".BrokerExecutor")


class BrokerExecutor:
    """
    Universal order executor that delegates to a connected BaseBroker instance.

    Wraps place_order / close_order / modify_order / position queries with the
    same method signatures as MT5Executor so existing call-sites need minimal
    changes.

    Parameters
    ----------
    broker     : BaseBroker — connected adapter from BrokerRouter.
    risk_engine: RiskEngine — optional; used for pre-trade approval checks.
    """

    def __init__(self, broker, risk_engine=None):
        self._broker      = broker
        self._risk_engine = risk_engine
        self.connected    = broker.connected
        _bex_logger.info(
            f"[BrokerExecutor] Attached to {broker.broker_name}"
        )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def ensure_connected(self) -> bool:
        return self._broker.ensure_connected()

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account_info(self) -> Optional[Dict]:
        return self._broker.get_account_info()

    def get_equity(self) -> float:
        return self._broker.get_equity()

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_position(self, symbol: str):
        positions = self._broker.get_open_positions(symbol)
        return positions[0] if positions else None

    def get_all_positions(self) -> List[Dict]:
        return self._broker.get_open_positions()

    def count_open_positions(self) -> int:
        return self._broker.count_open_positions()

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        lot: float,
        order_type: str,
        atr: float = 0.0,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 2.5,
        signal_prob: float = 0.65,
        comment: str = "AI_EA_v5",
    ) -> Optional[Dict]:
        """
        Place a market order via the connected broker adapter.

        Runs a risk-engine pre-approval check if risk_engine was provided,
        then delegates to broker.place_order().
        """
        if not self.ensure_connected():
            return None

        # NOTE: approve_trade() is intentionally NOT called here.
        # ai_ea.py calls risk_engine.approve_trade() before reaching this
        # executor, so calling it again here causes double-gating that can
        # silently block valid trades.  The MT5Adapter also used to call it a
        # third time — that has been removed too.

        try:
            result = self._broker.place_order(
                symbol=symbol,
                order_type=order_type,
                volume=lot,
                atr=atr,
                sl_atr_mult=sl_atr_mult,
                tp_atr_mult=tp_atr_mult,
                signal_prob=signal_prob,
                comment=comment,
            )
            if result:
                # record_trade_open() is called by ai_ea.py after place_order() succeeds
                _bex_logger.info(
                    f"ORDER ▶ {symbol} {order_type.upper()} {lot}L | "
                    f"prob={signal_prob:.3f} | ticket={result.get('ticket', '?')}"
                )
            return result
        except Exception as exc:
            _bex_logger.error(
                f"[BrokerExecutor] place_order({symbol}) error: {exc}",
                exc_info=True,
            )
            return None

    def close_position(self, ticket: int, symbol: str = "") -> bool:
        """Close a position by ticket ID."""
        try:
            ok = self._broker.close_order(ticket, symbol=symbol)
            if ok and self._risk_engine is not None:
                # Try to get profit from open positions list first
                self._risk_engine.record_trade_close(0.0, self.get_equity())
            return bool(ok)
        except Exception as exc:
            _bex_logger.error(
                f"[BrokerExecutor] close_position({ticket}) error: {exc}",
                exc_info=True,
            )
            return False

    def update_sl(self, ticket: int, new_sl: float) -> bool:
        """Modify the stop-loss of an open position."""
        try:
            return bool(self._broker.modify_order(ticket, sl=new_sl))
        except Exception as exc:
            _bex_logger.error(
                f"[BrokerExecutor] update_sl({ticket}) error: {exc}",
                exc_info=True,
            )
            return False
