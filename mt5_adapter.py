"""
mt5_adapter.py — Full MetaTrader 5 Broker Adapter (AI EA v5)
=============================================================
Implements BaseBroker for the MetaTrader 5 platform.

Supports: Exness, FXGT, IC Markets, Pepperstone, XM, and any MT5 broker.
Assets:   Forex, Metals, Indices, Crypto, Energies, Stocks

All methods return standardised dicts from BaseBroker — no MT5-specific
objects leak outside this module.
"""

import json
import logging
import math
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from base_broker import BaseBroker, OrderRejected

logger = logging.getLogger(__name__)

# ── Optional MT5 import ───────────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

# Constants
TRADE_LOG_PATH       = "data/trade_log.jsonl"
MAX_CONNECT_ATTEMPTS = 5
CONNECT_RETRY_DELAY  = 5
MAX_ORDER_RETRIES    = 5    # was 3 — extra retries for GTio demo latency
ORDER_RETRY_DELAY    = 2.0  # was 1.5 — slightly longer wait between attempts
DEVIATION_POINTS     = 50   # was 20 — GTio demo needs wider price tolerance
MAGIC_NUMBER         = 20250424

# Timeframe string → MT5 constant mapping
_TF_MAP: Dict[str, int] = {}

def _build_tf_map():
    global _TF_MAP
    if not MT5_AVAILABLE or mt5 is None:
        return
    _TF_MAP = {
        "m1":  mt5.TIMEFRAME_M1,
        "m5":  mt5.TIMEFRAME_M5,
        "m15": mt5.TIMEFRAME_M15,
        "m30": mt5.TIMEFRAME_M30,
        "h1":  mt5.TIMEFRAME_H1,
        "h2":  mt5.TIMEFRAME_H2,
        "h4":  mt5.TIMEFRAME_H4,
        "d1":  mt5.TIMEFRAME_D1,
        "w1":  mt5.TIMEFRAME_W1,
        "mn1": mt5.TIMEFRAME_MN1,
    }


def _get_digits(sym_info) -> int:
    """
    Return digit precision for a symbol's prices.
    Extracts from a live mt5 symbol_info object OR a dict (from _sym_to_dict).
    Falls back to 5 (standard forex precision) if unavailable.
    Centralised so every price-rounding call uses the same logic.
    """
    if sym_info is None:
        return 5
    if isinstance(sym_info, dict):
        return int(sym_info.get("digits", 5))
    return int(getattr(sym_info, "digits", 5))


class MT5Adapter(BaseBroker):
    """
    Full MetaTrader 5 adapter.  Can be used as a drop-in replacement for
    the original MT5Executor + MT5DataFetcher combination.
    """

    def __init__(
        self,
        login: int = 0,
        password: str = "",
        server: str = "",
        risk_engine=None,
    ):
        super().__init__()
        self.broker_name   = "MT5"
        self.login_id      = login
        self.password      = password
        self.server        = server
        self.risk_engine   = risk_engine
        self._symbols_cache: List[Dict] = []
        self._symbols_ts: float = 0.0
        os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)

        if MT5_AVAILABLE:
            _build_tf_map()

    # ─────────────────────────────────────────────────────────────────────────
    # Connection
    # ─────────────────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        if not MT5_AVAILABLE:
            logger.error("[MT5] MetaTrader5 Python package not installed.")
            return False
        for attempt in range(1, MAX_CONNECT_ATTEMPTS + 1):
            try:
                if not mt5.initialize():
                    raise ConnectionError(f"mt5.initialize() failed: {mt5.last_error()}")
                if self.login_id and self.password and self.server:
                    if not mt5.login(self.login_id, self.password, self.server):
                        raise ConnectionError(f"mt5.login() failed: {mt5.last_error()}")
                acct = mt5.account_info()
                if acct is None:
                    raise ConnectionError("No account info after login")
                self.connected   = True
                self.broker_name = f"MT5/{acct.server}"
                logger.info(
                    f"[MT5] Connected: server={acct.server} "
                    f"account={acct.login} equity={acct.equity:.2f} {acct.currency}"
                )
                return True
            except Exception as e:
                logger.warning(f"[MT5] Connect attempt {attempt}/{MAX_CONNECT_ATTEMPTS}: {e}")
                time.sleep(CONNECT_RETRY_DELAY)
        logger.critical("[MT5] All connect attempts failed.")
        return False

    def disconnect(self) -> None:
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("[MT5] Disconnected.")

    def ensure_connected(self) -> bool:
        if self.connected and MT5_AVAILABLE:
            try:
                if mt5.account_info() is not None:
                    return True
            except Exception:
                pass
        self.connected = False
        return self.connect()

    # ─────────────────────────────────────────────────────────────────────────
    # Symbol discovery
    # ─────────────────────────────────────────────────────────────────────────

    def get_symbols(self) -> List[Dict]:
        """Return all available MT5 symbols as standardised SymbolInfo dicts."""
        if not self.ensure_connected():
            return []
        # Use cache (60 min TTL)
        if self._symbols_cache and (time.time() - self._symbols_ts) < 3600:
            return self._symbols_cache
        try:
            raw = mt5.symbols_get()
            if not raw:
                return []
            result = []
            for sym in raw:
                try:
                    info = self._sym_to_dict(sym)
                    result.append(info)
                except Exception:
                    pass
            self._symbols_cache = result
            self._symbols_ts    = time.time()
            logger.info(f"[MT5] Loaded {len(result)} symbols")
            return result
        except Exception as e:
            logger.error(f"[MT5] get_symbols error: {e}")
            return []

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        if not self.ensure_connected():
            return None
        try:
            sym = mt5.symbol_info(symbol)
            if sym is None:
                # Try to select and retry
                mt5.symbol_select(symbol, True)
                sym = mt5.symbol_info(symbol)
            return self._sym_to_dict(sym) if sym else None
        except Exception as e:
            logger.error(f"[MT5] get_symbol_info({symbol}): {e}")
            return None

    def select_symbol(self, symbol: str) -> bool:
        if not self.ensure_connected():
            return False
        info = mt5.symbol_info(symbol)
        if info is None:
            return False
        if not info.visible:
            return mt5.symbol_select(symbol, True)
        return True

    def _sym_to_dict(self, sym) -> Dict:
        return {
            "name":          sym.name,
            "contract_size": float(getattr(sym, "trade_contract_size", 100000)),
            "point":         float(getattr(sym, "point", 0.00001)),
            "digits":        int(getattr(sym, "digits", 5)),
            "min_lot":       float(getattr(sym, "volume_min", 0.01)),
            "max_lot":       float(getattr(sym, "volume_max", 100.0)),
            "lot_step":      float(getattr(sym, "volume_step", 0.01)),
            "spread":        int(getattr(sym, "spread", 0)),
            "trade_mode":    int(getattr(sym, "trade_mode", 4)),
            "filling_mode":  int(getattr(sym, "filling_mode", 1)),
            "stops_level":   int(getattr(sym, "trade_stops_level", 0)),
            "asset_class":   self.classify_asset(sym.name),
        }

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
        if not _TF_MAP:
            _build_tf_map()
        tf_int = _TF_MAP.get(timeframe.lower(), mt5.TIMEFRAME_H1)
        try:
            # Ensure symbol is visible
            sym_info = mt5.symbol_info(symbol)
            if sym_info is None:
                logger.error(f"[MT5] Symbol not found: {symbol}")
                return None
            if not sym_info.visible:
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"[MT5] Cannot select symbol: {symbol}")
                    return None

            rates = mt5.copy_rates_from_pos(symbol, tf_int, 0, bars)
            if rates is None or len(rates) == 0:
                logger.error(f"[MT5] No data: {symbol} {timeframe}")
                return None

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            # Normalise column names
            rename = {
                "tick_volume": "tick_volume",
                "real_volume": "real_volume",
                "spread":      "spread",
            }
            for old, new in rename.items():
                if old in df.columns and old != new:
                    df.rename(columns={old: new}, inplace=True)
            df["symbol"] = symbol
            df = self._add_indicators(df)
            return df

        except Exception as e:
            logger.error(f"[MT5] get_market_data({symbol},{timeframe}): {e}", exc_info=True)
            return None

    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        if not self.ensure_connected():
            return None
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            return {
                "bid":    float(tick.bid),
                "ask":    float(tick.ask),
                "spread": float(tick.ask - tick.bid),
                "time":   datetime.fromtimestamp(tick.time).isoformat(),
            }
        except Exception as e:
            logger.error(f"[MT5] get_latest_price({symbol}): {e}")
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
        magic: int = MAGIC_NUMBER,
        atr: float = 0.0,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 2.5,
        signal_prob: float = 0.65,
    ) -> Optional[Dict]:
        if not self.ensure_connected():
            return None

        # NOTE: approve_trade() is intentionally NOT called here.
        # It is already called upstream in ai_ea.py and BrokerExecutor.place_order().
        # Calling it a second (or third) time against the same shared risk_engine
        # state causes silent triple-gating that blocks valid trades.

        # ── 1. Force-select the symbol FIRST ─────────────────────────────────
        # Always call symbol_select even if visible — GTio and some prop-firm
        # demo servers require an explicit select before each order or MT5
        # silently returns None from order_send with no last_error detail.
        if not mt5.symbol_select(symbol, True):
            logger.error(f"[MT5] Cannot select symbol: {symbol}")
            return None

        # Re-fetch info after select so filling_mode/stops_level are fresh
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            logger.error(f"[MT5] Symbol info unavailable after select: {symbol}")
            return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"[MT5] No tick data for {symbol}")
            return None

        # ── 2. Market open guard ─────────────────────────────────────────────
        # trade_mode: 0=disabled, 1=longonly, 2=shortonly, 3=closeonly, 4=full
        # If not 4 (full) or 1/2 for the requested direction, skip silently.
        trade_mode = int(getattr(sym_info, "trade_mode", 4))
        if trade_mode == 0:
            logger.info(f"[MT5] {symbol}: market closed (trade_mode=0) — skipping order")
            return None
        if trade_mode == 3:
            logger.info(f"[MT5] {symbol}: close-only mode — skipping new order")
            return None

        # ── 3. Spread and margin guards ───────────────────────────────────────
        if not self._spread_ok(symbol, sym_info, tick):
            return None

        acct = self.get_account_info()
        if acct and not self._margin_ok(symbol, volume, order_type, tick, acct):
            return None

        # ── 4. Extract symbol parameters — all cast to native Python types ────
        # MT5 C-extension objects return values that can silently be numpy or
        # ctypes scalars.  order_send rejects requests containing non-native
        # Python types, which causes a silent None response with no error detail.
        digits      = int(sym_info.digits)
        point       = float(sym_info.point)
        stops_level = int(sym_info.trade_stops_level)

        # ── 4. Ensure ATR is a usable value ───────────────────────────────────
        atr = float(atr)
        if atr <= 0.0:
            atr = float(tick.ask - tick.bid) * 50.0
        if atr <= 0.0:
            atr = point * 100   # absolute fallback: 100 points

        # ── 5. Determine execution price and order type ───────────────────────
        if order_type.lower() == "buy":
            exec_price = round(float(tick.ask), digits)
            calc_sl    = round(exec_price - atr * sl_atr_mult, digits)
            calc_tp    = round(exec_price + atr * tp_atr_mult, digits)
            mt5_type   = int(mt5.ORDER_TYPE_BUY)
        else:
            exec_price = round(float(tick.bid), digits)
            calc_sl    = round(exec_price + atr * sl_atr_mult, digits)
            calc_tp    = round(exec_price - atr * tp_atr_mult, digits)
            mt5_type   = int(mt5.ORDER_TYPE_SELL)

        # Use caller's SL/TP if provided, otherwise ATR-derived values
        final_sl = float(sl) if sl is not None else calc_sl
        final_tp = float(tp) if tp is not None else calc_tp

        # ── 6. Enforce minimum stop distance ─────────────────────────────────
        # MT5 requires SL/TP at least (stops_level + DEVIATION_POINTS) ticks away.
        # We use max(broker_min, half_atr) as the floor so very tight stops on
        # low-volatility instruments still clear the broker's minimum.
        min_stop_dist = max(
            (stops_level + DEVIATION_POINTS) * point,
            atr * sl_atr_mult * 0.5,
        )
        if order_type.lower() == "buy":
            sl_floor = round(exec_price - max(min_stop_dist, atr * sl_atr_mult), digits)
            tp_ceil  = round(exec_price + max(min_stop_dist, atr * tp_atr_mult), digits)
            if final_sl > exec_price - min_stop_dist:
                logger.warning(
                    f"[MT5] SL {final_sl} too close — adjusting to {sl_floor}"
                )
                final_sl = sl_floor
            if final_tp < exec_price + min_stop_dist:
                logger.warning(
                    f"[MT5] TP {final_tp} too close — adjusting to {tp_ceil}"
                )
                final_tp = tp_ceil
        else:
            sl_floor = round(exec_price + max(min_stop_dist, atr * sl_atr_mult), digits)
            tp_ceil  = round(exec_price - max(min_stop_dist, atr * tp_atr_mult), digits)
            if final_sl < exec_price + min_stop_dist:
                logger.warning(
                    f"[MT5] SL {final_sl} too close — adjusting to {sl_floor}"
                )
                final_sl = sl_floor
            if final_tp > exec_price - min_stop_dist:
                logger.warning(
                    f"[MT5] TP {final_tp} too close — adjusting to {tp_ceil}"
                )
                final_tp = tp_ceil

        # ── 7. Validate and clamp volume — cast to native float ───────────────
        vol = float(self.validate_volume(volume, self._sym_to_dict(sym_info)))

        # ── 8. Detect filling mode ────────────────────────────────────────────
        filling = int(self._detect_filling(sym_info))

        # ── 9. Build request — every value explicitly cast to native Python ────
        # MT5's C extension is strict: numpy int64, numpy float64, ctypes values
        # all silently produce a None from order_send with no last_error detail.
        #
        # COMMENT FIX (GTio / prop-firm MT5 servers):
        # These servers validate comments at the C level and reject strings that are
        # longer than ~16 chars OR contain underscores, dots, or special chars — even
        # though the MT5 spec allows them. The consistent symptom is error
        # (-2, 'Invalid "comment" argument') on every attempt regardless of filling mode.
        #
        # Workaround: use a short, pure-alphanumeric comment (no underscores, no dots).
        # Magic number already encodes EA identity, so the comment only needs to carry
        # a compact signal tag for visual identification in the MT5 terminal.
        import re as _re
        _prob_tag   = f"{int(signal_prob * 100):02d}"          # e.g. 0.566 → "56"
        _dir_tag    = "B" if order_type.lower() == "buy" else "S"
        safe_comment = f"AIEA{_dir_tag}{_prob_tag}"[:15]       # e.g. "AIAES56" — 7 chars, pure alphanum
        request = {
            "action":       int(mt5.TRADE_ACTION_DEAL),
            "symbol":       str(symbol),
            "volume":       float(vol),
            "type":         int(mt5_type),
            "price":        float(exec_price),
            "sl":           float(round(final_sl, digits)),
            "tp":           float(round(final_tp, digits)),
            "deviation":    int(DEVIATION_POINTS),
            "magic":        int(magic),
            "comment":      safe_comment,
            "type_time":    int(mt5.ORDER_TIME_GTC),
            "type_filling": int(filling),
        }

        logger.info(
            f"[MT5] Sending order: {symbol} {order_type.upper()} vol={vol} "
            f"price={exec_price} sl={round(final_sl,digits)} tp={round(final_tp,digits)} "
            f"filling={filling} dev={DEVIATION_POINTS}"
        )

        result = self._send_with_retry(request)
        if result is None:
            return None

        spread_pips = (tick.ask - tick.bid) / point / 10.0

        record = {
            "ticket":      int(result.order),
            "symbol":      symbol,
            "type":        order_type,
            "volume":      vol,
            "price":       exec_price,
            "sl":          round(final_sl, digits),
            "tp":          round(final_tp, digits),
            "spread_pips": round(spread_pips, 2),
            "signal_prob": signal_prob,
            "atr":         atr,
            "comment":     comment,
            "retcode":     result.retcode,
            "success":     True,
            "timestamp":   datetime.now().isoformat(),
            "broker":      "mt5",
        }
        self._log_trade(record)
        # NOTE: record_trade_open() is called by ai_ea.py after place_order() returns.
        # Do NOT call it here — that would double-count every trade in daily_trades.
        logger.info(
            f"[MT5] ORDER ▶ {symbol} {order_type.upper()} {vol}L | "
            f"price={exec_price} sl={final_sl:.{digits}f} tp={final_tp:.{digits}f} | "
            f"ticket={result.order} | prob={signal_prob:.3f}"
        )
        return record

    def _send_with_retry(self, request: Dict):
        # Retcodes that are fatal — no point retrying
        FATAL_CODES = {
            10006,  # Request rejected
            10007,  # Request canceled by trader
            10010,  # Only part of the request was completed
            10011,  # Request processing error
            10014,  # Invalid volume in the request
            10015,  # Market closed
            10016,  # Invalid stops — SL/TP too close to price
            10017,  # Trade disabled
            10018,  # Market closed
            10019,  # No money
        }
        # Retcodes where we should rotate filling mode and retry
        # NOTE: 10022 = "Unsupported filling mode" — MUST be in FILLING_CODES,
        # not FATAL_CODES, so the retry loop can switch to the correct mode.
        FILLING_CODES = {10022, 10026, 10030}

        last_result = None
        for attempt in range(1, MAX_ORDER_RETRIES + 1):
            result = mt5.order_send(request)
            last_result = result

            if result is None:
                # None means MT5 rejected the request at the API validation level
                # (e.g. invalid filling mode, symbol not selected, bad parameter).
                # Always log last_error so the problem is diagnosable.
                err = mt5.last_error()
                logger.warning(
                    f"[MT5] order_send returned None "
                    f"(attempt {attempt}/{MAX_ORDER_RETRIES}) last_error={err} "
                    f"filling={request.get('type_filling')} symbol={request.get('symbol')}"
                )
                # Rotate filling mode on EVERY None attempt (not just attempt 1).
                # Staying on the same rejected filling mode wastes all remaining retries.
                # Cycle: FOK(0) → RETURN(2) → IOC(1) → FOK(0)
                current = request.get("type_filling", -1)
                FOK  = getattr(mt5, "ORDER_FILLING_FOK",    0)
                IOC  = getattr(mt5, "ORDER_FILLING_IOC",    1)
                RTRN = getattr(mt5, "ORDER_FILLING_RETURN", 2)
                alt  = {FOK: RTRN, RTRN: IOC, IOC: FOK}.get(current, RTRN)
                logger.warning(
                    f"[MT5] None response — rotating filling {current} → {alt} for retry"
                )
                request["type_filling"] = alt
                # On second None attempt, also try stripping comment entirely.
                # Some prop-firm/demo brokers (e.g. GTio) reject ANY comment string
                # from automated systems via broker-side policy.  An empty comment
                # bypasses that validation entirely; the magic number still identifies the EA.
                if attempt == 2 and request.get("comment", "") != "":
                    logger.warning(
                        "[MT5] Persistent None — stripping comment to '' for remaining retries"
                    )
                    request["comment"] = ""
                time.sleep(ORDER_RETRY_DELAY)
                continue

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return result

            if result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                tick = mt5.symbol_info_tick(request["symbol"])
                if tick:
                    request["price"] = (
                        tick.ask if request["type"] == mt5.ORDER_TYPE_BUY else tick.bid
                    )
                logger.warning(
                    f"[MT5] Requote on {request['symbol']}, "
                    f"retrying with new price={request['price']}"
                )
                time.sleep(ORDER_RETRY_DELAY)
                continue

            if result.retcode in FILLING_CODES:
                # Rotate filling mode: RETURN → IOC → FOK → RETURN
                current = request.get("type_filling", -1)
                FOK  = getattr(mt5, "ORDER_FILLING_FOK",    0)
                IOC  = getattr(mt5, "ORDER_FILLING_IOC",    1)
                RTRN = getattr(mt5, "ORDER_FILLING_RETURN", 2)
                alt  = {RTRN: IOC, IOC: FOK, FOK: RTRN}.get(current, RTRN)
                logger.warning(
                    f"[MT5] Filling mode {current} rejected "
                    f"(retcode={result.retcode} '{result.comment}'), switching to {alt}"
                )
                request["type_filling"] = alt
                time.sleep(ORDER_RETRY_DELAY)
                continue

            if result.retcode in FATAL_CODES or result.retcode in (
                mt5.TRADE_RETCODE_MARKET_CLOSED, mt5.TRADE_RETCODE_NO_MONEY
            ):
                logger.error(
                    f"[MT5] FATAL order error on {request['symbol']}: "
                    f"retcode={result.retcode} comment='{result.comment}' — not retrying"
                )
                return None

            logger.warning(
                f"[MT5] Order attempt {attempt}/{MAX_ORDER_RETRIES} on {request['symbol']}: "
                f"retcode={result.retcode} comment='{result.comment}'"
            )
            time.sleep(ORDER_RETRY_DELAY)

        # All retries exhausted
        if last_result is not None:
            logger.error(
                f"[MT5] ORDER FAILED after {MAX_ORDER_RETRIES} attempts on {request['symbol']}: "
                f"retcode={last_result.retcode} comment='{last_result.comment}' "
                f"| price={request.get('price')} sl={request.get('sl')} tp={request.get('tp')} "
                f"vol={request.get('volume')} filling={request.get('type_filling')}"
            )
        else:
            logger.error(
                f"[MT5] ORDER FAILED — order_send returned None for {request['symbol']} "
                f"after {MAX_ORDER_RETRIES} attempts. "
                f"last_error={mt5.last_error()} "
                f"filling={request.get('type_filling')} — Check MT5 connection."
            )
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Order management
    # ─────────────────────────────────────────────────────────────────────────

    def close_order(self, ticket: int, symbol: str = "", volume: float = 0.0) -> bool:
        if not self.ensure_connected():
            return False
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.error(f"[MT5] Position {ticket} not found")
            return False
        pos = positions[0]
        close_type = mt5.ORDER_TYPE_BUY if pos.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return False
        close_price = tick.ask if close_type == mt5.ORDER_TYPE_BUY else tick.bid
        sym_info    = mt5.symbol_info(pos.symbol)
        filling     = self._detect_filling(sym_info) if sym_info else getattr(mt5, "ORDER_FILLING_RETURN", 2)
        request = {
            "action":       int(mt5.TRADE_ACTION_DEAL),
            "symbol":       str(pos.symbol),
            "volume":       float(pos.volume),
            "type":         int(close_type),
            "position":     int(ticket),
            "price":        float(close_price),
            "deviation":    int(DEVIATION_POINTS),
            "magic":        int(MAGIC_NUMBER),
            "comment":      "AI_EA_v5_close",
            "type_time":    int(mt5.ORDER_TIME_GTC),
            "type_filling": int(filling),
        }
        result = self._send_with_retry(request)
        if result:
            equity = self.get_equity()
            if self.risk_engine:
                self.risk_engine.record_trade_close(pos.profit, equity)
            self._log_trade({
                "event": "close", "ticket": ticket, "symbol": pos.symbol,
                "profit": pos.profit, "equity_after": equity,
                "timestamp": datetime.now().isoformat(),
            })
            logger.info(f"[MT5] Closed ticket={ticket} | P&L={pos.profit:.2f}")
            return True
        return False

    def modify_order(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        if not self.ensure_connected():
            return False
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        pos = positions[0]
        r = mt5.order_send({
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl":       sl if sl is not None else pos.sl,
            "tp":       tp if tp is not None else pos.tp,
            "symbol":   pos.symbol,
            "deviation": DEVIATION_POINTS,
        })
        ok = r and r.retcode == mt5.TRADE_RETCODE_DONE
        if ok:
            logger.info(f"[MT5] Modified ticket={ticket} sl={sl} tp={tp}")
        else:
            logger.error(f"[MT5] Modify failed ticket={ticket}: {r.comment if r else 'None'}")
        return ok

    # ─────────────────────────────────────────────────────────────────────────
    # Account
    # ─────────────────────────────────────────────────────────────────────────

    def get_account_info(self) -> Optional[Dict]:
        if not self.ensure_connected():
            return None
        i = mt5.account_info()
        if i is None:
            return None
        return {
            "login":        i.login,
            "balance":      i.balance,
            "equity":       i.equity,
            "margin":       i.margin,
            "free_margin":  i.margin_free,
            "margin_level": i.margin_level,
            "currency":     i.currency,
            "leverage":     i.leverage,
            "server":       i.server,
            "broker":       "mt5",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Positions
    # ─────────────────────────────────────────────────────────────────────────

    def get_open_positions(self, symbol: str = "") -> List[Dict]:
        if not self.ensure_connected():
            return []
        try:
            if symbol:
                raw = mt5.positions_get(symbol=symbol)
            else:
                raw = mt5.positions_get()
            if not raw:
                return []
            result = []
            for p in raw:
                result.append({
                    "ticket":     p.ticket,
                    "symbol":     p.symbol,
                    "type":       "buy" if p.type == mt5.ORDER_TYPE_BUY else "sell",
                    "volume":     p.volume,
                    "open_price": p.price_open,
                    "current":    p.price_current,
                    "sl":         p.sl,
                    "tp":         p.tp,
                    "profit":     p.profit,
                    "magic":      p.magic,
                    "comment":    p.comment,
                    "open_time":  datetime.fromtimestamp(p.time).isoformat(),
                    "broker":     "mt5",
                })
            return result
        except Exception as e:
            logger.error(f"[MT5] get_open_positions: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # Trade history
    # ─────────────────────────────────────────────────────────────────────────

    def get_trade_history(self, days: int = 365, symbol: str = "") -> List[Dict]:
        """
        Pull full closed trade history from MT5.  (v20 — was 7d, now 365d)

        Uses mt5.history_deals_get() with DEAL_ENTRY_OUT to get every closed
        position. Also pulls the matching IN deal to reconstruct open_time and
        open_price so the TradeHistoryLearner gets accurate entry data.
        """
        if not self.ensure_connected():
            return []
        try:
            end   = datetime.now()
            start = end - timedelta(days=days)
            deals = mt5.history_deals_get(start, end)
            if not deals:
                return []

            # Build a map of position_id → IN deal for open_price / open_time
            in_deals: Dict[int, object] = {}
            for d in deals:
                if d.entry == mt5.DEAL_ENTRY_IN:
                    in_deals[d.position_id] = d

            out = []
            for d in deals:
                if d.entry != mt5.DEAL_ENTRY_OUT:
                    continue
                if symbol and d.symbol != symbol:
                    continue

                # Reconstruct open side from matching IN deal
                in_d = in_deals.get(d.position_id)
                open_price = float(in_d.price)                             if in_d else 0.0
                open_time  = datetime.fromtimestamp(in_d.time).isoformat() if in_d else ""
                close_time = datetime.fromtimestamp(d.time).isoformat()

                out.append({
                    "ticket":      d.ticket,
                    "symbol":      d.symbol,
                    "type":        "buy" if d.type == mt5.DEAL_TYPE_BUY else "sell",
                    "volume":      float(d.volume),
                    "open_price":  open_price,
                    "close_price": float(d.price),
                    "profit":      float(d.profit),
                    "open_time":   open_time,
                    "close_time":  close_time,
                    "comment":     d.comment,
                    "broker":      "mt5",
                })
            logger.info(f"[MT5] get_trade_history: {len(out)} closed deals (last {days}d)")
            return out
        except Exception as e:
            logger.error(f"[MT5] get_trade_history: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # Compatibility helpers (used by existing executor/data_fetcher callers)
    # ─────────────────────────────────────────────────────────────────────────

    def get_candles(self, symbol: str, timeframe: str, bars: int = 500) -> Optional[pd.DataFrame]:
        """Alias for get_market_data — keeps old API compatible."""
        return self.get_market_data(symbol, timeframe, bars)

    def get_equity(self) -> float:
        info = self.get_account_info()
        return float(info["equity"]) if info else 0.0

    def count_open_positions(self) -> int:
        return len(self.get_open_positions())

    def get_position(self, symbol: str):
        """Legacy compatibility — returns first position for symbol or None."""
        positions = self.get_open_positions(symbol)
        return positions[0] if positions else None

    def get_all_positions(self) -> List[Dict]:
        return self.get_open_positions()

    def update_sl(self, ticket: int, new_sl: float) -> bool:
        return self.modify_order(ticket, sl=new_sl)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_filling(self, sym_info) -> int:
        """
        Return the correct ORDER_FILLING constant for this symbol.

        BITMASK vs CONSTANT — the critical distinction:
          sym_info.filling_mode is a BITMASK:
            bit 0 (value 1) → FOK  supported
            bit 1 (value 2) → IOC  supported
            bit 2 (value 4) → RETURN supported

          ORDER_FILLING constants for type_filling:
            ORDER_FILLING_FOK    = 0
            ORDER_FILLING_IOC    = 1
            ORDER_FILLING_RETURN = 2

        Using the ORDER_FILLING constants as bitmask test values was the
        original bug: ``fm & IOC`` (i.e. ``fm & 1``) tests the FOK-supported
        bit, not the IOC-supported bit.  ``fm & FOK`` (``fm & 0``) is always
        falsy and never selects FOK.

        Priority: RETURN > IOC > FOK.  RETURN is the most permissive mode and
        is accepted by virtually every demo/prop-firm MT5 server.
        """
        if sym_info is None:
            return getattr(mt5, "ORDER_FILLING_RETURN", 2)
        try:
            fm = int(getattr(sym_info, "filling_mode", 0))

            # Bitmask positions inside filling_mode field
            BIT_FOK    = 1   # bit 0
            BIT_IOC    = 2   # bit 1
            BIT_RETURN = 4   # bit 2

            # ORDER_FILLING constants (what goes into type_filling)
            ORDER_FOK    = getattr(mt5, "ORDER_FILLING_FOK",    0)
            ORDER_IOC    = getattr(mt5, "ORDER_FILLING_IOC",    1)
            ORDER_RETURN = getattr(mt5, "ORDER_FILLING_RETURN", 2)

            if fm & BIT_RETURN: return ORDER_RETURN
            if fm & BIT_IOC:    return ORDER_IOC
            if fm & BIT_FOK:    return ORDER_FOK
        except Exception:
            pass
        # Default: RETURN — accepted by most MT5 brokers including all prop-firm demos
        return getattr(mt5, "ORDER_FILLING_RETURN", 2)

    def _spread_ok(self, symbol: str, sym_info, tick) -> bool:
        try:
            spread_pips = (tick.ask - tick.bid) / sym_info.point / 10
            max_sp = 50.0
            u = symbol.upper()
            if any(x in u for x in ["BTC", "ETH"]):  max_sp = 400.0
            elif any(x in u for x in ["XAU", "GOLD"]): max_sp = 200.0
            elif "OIL" in u:                             max_sp = 100.0
            elif "JPY" in u:                             max_sp = 30.0
            if spread_pips > max_sp:
                logger.warning(f"[MT5] Spread too wide [{symbol}]: {spread_pips:.1f}>{max_sp} pips")
                return False
        except Exception:
            pass
        return True

    def _margin_ok(self, symbol: str, volume: float, order_type: str, tick, acct: Dict) -> bool:
        try:
            mt5_type   = mt5.ORDER_TYPE_BUY if order_type == "buy" else mt5.ORDER_TYPE_SELL
            margin_req = mt5.order_calc_margin(mt5_type, symbol, volume, tick.ask)
            if margin_req is None:
                return True
            free = acct.get("free_margin", 0)
            if margin_req > free * 0.9:
                logger.error(f"[MT5] Insufficient margin: need {margin_req:.2f}, free={free:.2f}")
                return False
        except Exception:
            pass
        return True

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add standard technical indicators to a candle DataFrame."""
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
            logger.debug(f"[MT5] _add_indicators error: {e}")
        return df

    @staticmethod
    def _log_trade(record: Dict) -> None:
        try:
            os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
            with open(TRADE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Trade log write failed: {e}")

    @staticmethod
    def get_pip_multiplier(symbol: str) -> float:
        s = symbol.upper()
        if "XAU" in s: return 0.01
        if "BTC" in s or "ETH" in s: return 1.0
        if "OIL" in s: return 0.001
        if "JPY" in s: return 0.01
        return 0.0001
