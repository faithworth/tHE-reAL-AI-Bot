"""
data_fetcher.py — Data Fetcher layer (AI EA v5)
================================================
Contains two classes:

  MT5DataFetcher      — original MT5-specific fetcher (backward compat).
                        Only usable when MetaTrader5 package is installed
                        and MT5Executor has already called initialize()+login().

  BrokerDataFetcher   — universal fetcher that wraps any BaseBroker adapter
                        (MT5Adapter, IBKRAdapter, CTraderAdapter).
                        Use this class for all new code.

The MT5 import is conditional — importing this module will NOT crash on
systems where MetaTrader5 is not installed (e.g. IBKR / cTrader machines).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time, timezone
import logging
import pytz
from typing import Dict, List, Optional, Tuple, Union

# Optional MT5 import — only available on Windows with MT5 installed
try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    _MT5_AVAILABLE = False


class MT5DataFetcher:
    """
    MT5-specific data fetcher.  Requires MetaTrader5 to be installed and
    MT5Executor to have already called initialize()+login().
    On non-MT5 systems use BrokerDataFetcher instead.
    """

    def __init__(self):
        if not _MT5_AVAILABLE:
            raise ImportError(
                "MT5DataFetcher requires the MetaTrader5 package. "
                "Use BrokerDataFetcher for IBKR / cTrader."
            )
        self.connected = False
        self.timeframe_map = {
            'm1':  mt5.TIMEFRAME_M1,
            'm5':  mt5.TIMEFRAME_M5,
            'm15': mt5.TIMEFRAME_M15,
            'm30': mt5.TIMEFRAME_M30,
            'h1':  mt5.TIMEFRAME_H1,
            'h2':  mt5.TIMEFRAME_H2,
            'h4':  mt5.TIMEFRAME_H4,
            'd1':  mt5.TIMEFRAME_D1,
            'w1':  mt5.TIMEFRAME_W1,
        }
        self.ict_config = {
            'fvg_lookback':            3,
            'ob_wick_ratio':           0.7,
            'liquidity_window':        20,
            'volume_spike_threshold':  2.0,
            'killzones': {
                'london':   (7, 9),
                'new_york': (8, 10),
                'tokyo':    (0, 2),
            },
        }
        self.connect()

    def connect(self) -> bool:
        """
        Attach to the MT5 session.  initialize() is idempotent — if
        MT5Executor already initialised and logged in, this just confirms
        the connection is alive without overwriting credentials.
        """
        if not mt5.initialize():
            raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")
        # Verify we have an active account (means executor already logged in)
        acct = mt5.account_info()
        if acct is None:
            raise ConnectionError(
                "MT5DataFetcher: no active MT5 account after initialize(). "
                "Make sure MT5Executor.connect() runs before MT5DataFetcher()."
            )
        self.connected = True
        logging.info(
            f"Connected to MT5 for data fetching (account={acct.login})"
        )
        return True

    def shutdown(self):
        """Shutdown MT5 connection — only call when the whole EA exits."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logging.info("Disconnected from MT5 (data fetcher)")

    def get_candles(
        self,
        symbol: str,
        timeframe: Union[str, int],
        bars: int = 1000,
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data for *symbol*.
        symbol must already be the broker's exact symbol name
        (e.g. 'XAUUSDm', not 'XAUUSD..').
        """
        if not self.connected:
            try:
                self.connect()
            except Exception as e:
                logging.error(f"Cannot reconnect to MT5: {e}")
                return None

        if isinstance(timeframe, str):
            timeframe = self._parse_timeframe(timeframe)

        # Ensure the symbol is visible/selectable before requesting data
        info = mt5.symbol_info(symbol)
        if info is None:
            logging.error(f"Symbol not found in MT5: {symbol}")
            return None
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                logging.error(f"Cannot select symbol {symbol}")
                return None

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            logging.error(f"No data returned for {symbol} {timeframe}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df['symbol'] = symbol

        df = self._add_technical_indicators(df)
        return df

    def get_multi_timeframe_data(
        self,
        symbol: str,
        timeframes: List[Union[str, int]],
    ) -> Dict[str, pd.DataFrame]:
        data = {}
        for tf in timeframes:
            tf_str = tf if isinstance(tf, str) else self._get_timeframe_str(tf)
            data[tf_str] = self.get_candles(symbol, tf)
        return data

    def is_in_killzone(self, killzone_type: str = "london") -> bool:
        if killzone_type not in self.ict_config['killzones']:
            raise ValueError(f"Unknown killzone type: {killzone_type}")
        tz_map = {
            'london':   'Europe/London',
            'new_york': 'America/New_York',
            'tokyo':    'Asia/Tokyo',
        }
        tz = pytz.timezone(tz_map[killzone_type])
        now = datetime.now(tz).time()
        start_hour, end_hour = self.ict_config['killzones'][killzone_type]
        return time(start_hour, 0) <= now <= time(end_hour, 0)

    # ── Timeframe helpers ─────────────────────────────────────────────────────

    def _parse_timeframe(self, tf_str: str) -> int:
        return self.timeframe_map.get(tf_str.lower(), mt5.TIMEFRAME_H1)

    def _get_timeframe_str(self, tf_int: int) -> str:
        reverse_map = {v: k for k, v in self.timeframe_map.items()}
        return reverse_map.get(tf_int, 'h1')

    # ── Technical indicators ──────────────────────────────────────────────────

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta   = series.diff()
        gain    = delta.where(delta > 0, 0)
        loss    = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high_low   = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close  = np.abs(df['low']  - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    def _calculate_macd(
        self,
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series]:
        ema_fast   = close.ewm(span=fast, adjust=False).mean()
        ema_slow   = close.ewm(span=slow, adjust=False).mean()
        macd       = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or len(df) < 20:
            return df
        try:
            df['body']       = df['close'] - df['open']
            df['range']      = df['high']  - df['low']
            df['body_pct']   = abs(df['body']) / df['range'].replace(0, np.nan)
            df['sma20']      = df['close'].rolling(20).mean()
            df['ema50']      = df['close'].ewm(span=50,  adjust=False).mean()
            df['ema200']     = df['close'].ewm(span=200, adjust=False).mean()
            df['rsi']        = self._calculate_rsi(df['close'], 14)
            df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
            df['atr']        = self._calculate_atr(df, 14)
            df['volatility'] = df['close'].pct_change().rolling(14).std() * 100

            if 'real_volume' in df.columns:
                df['volume_ma']    = df['real_volume'].rolling(20).mean()
                df['volume_ratio'] = df['real_volume'] / df['volume_ma']

            df['fvg_bullish'],  df['fvg_bearish'] = self._calculate_enhanced_fvg(df)
            df['ob_bullish'],   df['ob_bearish']  = self._calculate_order_blocks(df)
            df['liquidity_pools']  = self._calculate_liquidity_pools(df)
            df['mitigation_blocks'] = self._calculate_mitigation_blocks(df)
            df['liquidity_grabs']  = self._calculate_liquidity_grabs(df)
            df['trend_strength']   = self._calculate_trend_strength(df)
        except Exception as e:
            logging.error(f"Indicator calculation error: {e}")
        return df

    # ── ICT / SMC helpers ─────────────────────────────────────────────────────

    def _calculate_enhanced_fvg(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        lb = self.ict_config['fvg_lookback']
        fvg_bullish = (df['low'] > df['high'].shift(lb)) & (df['close'] > df['high'].shift(1))
        fvg_bearish = (df['high'] < df['low'].shift(lb))  & (df['close'] < df['low'].shift(1))
        return fvg_bullish.fillna(False), fvg_bearish.fillna(False)

    def _calculate_order_blocks(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        wr           = self.ict_config['ob_wick_ratio']
        rng          = df['high'] - df['low']
        body         = df['close'] - df['open']
        next_close   = df['close'].shift(-1)
        bull_cond = (
            (df['close'] < df['open']) &
            ((-body) > rng * wr) &
            (next_close > df['high'])
        )
        bear_cond = (
            (df['close'] > df['open']) &
            (body > rng * wr) &
            (next_close < df['low'])
        )
        if 'volume_ratio' in df.columns:
            vol_ok    = df['volume_ratio'] > 1.0
            bull_cond = bull_cond & vol_ok
            bear_cond = bear_cond & vol_ok
        return bull_cond.fillna(False), bear_cond.fillna(False)

    def _calculate_liquidity_pools(self, df: pd.DataFrame) -> pd.Series:
        w = self.ict_config['liquidity_window']
        is_swing_high = df['high'] == df['high'].rolling(w, center=True).max()
        is_swing_low  = df['low']  == df['low'].rolling(w,  center=True).min()
        pools = is_swing_high | is_swing_low
        if 'volume_ratio' in df.columns:
            thr = self.ict_config['volume_spike_threshold']
            pools = pools & (df['volume_ratio'] > thr)
        return pools

    def _calculate_mitigation_blocks(self, df: pd.DataFrame) -> pd.Series:
        prev_close = df['close'].shift(1)
        prev_open  = df['open'].shift(1)
        prev_low   = df['low'].shift(1)
        prev_high  = df['high'].shift(1)
        bullish_mb = (prev_close < prev_open) & (df['close'] > prev_low)
        bearish_mb = (prev_close > prev_open) & (df['close'] < prev_high)
        return (bullish_mb | bearish_mb).fillna(False)

    def _calculate_liquidity_grabs(self, df: pd.DataFrame) -> pd.Series:
        low2  = df['low'].shift(2)
        high2 = df['high'].shift(2)
        bullish_sweep = (df['low'] < low2)   & (df['close'] > low2)
        bearish_sweep = (df['high'] > high2) & (df['close'] < high2)
        return (bullish_sweep | bearish_sweep).fillna(False)

    def _calculate_trend_strength(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        plus_dm  =  df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm   < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr       = self._calculate_atr(df, period)
        plus_di  = 100 * (plus_dm.ewm(alpha=1/period).mean()  / tr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        return dx.ewm(alpha=1/period).mean()

    def get_ict_market_state(
        self, symbol: str, timeframe: Union[str, int] = 'h1'
    ) -> Dict:
        df = self.get_candles(symbol, timeframe, 100)
        if df is None or len(df) < 20:
            return {}
        return {
            'trend': {
                'direction': 'up' if df['close'].iloc[-1] > df['ema50'].iloc[-1] else 'down',
                'strength':  float(df['trend_strength'].iloc[-1]),
            },
            'liquidity': {
                'pools': bool(df['liquidity_pools'].iloc[-1]),
                'grabs': bool(df['liquidity_grabs'].iloc[-1]),
            },
            'value': {
                'fvg_bullish': bool(df['fvg_bullish'].iloc[-1]),
                'fvg_bearish': bool(df['fvg_bearish'].iloc[-1]),
                'ob_bullish':  bool(df['ob_bullish'].iloc[-1]),
                'ob_bearish':  bool(df['ob_bearish'].iloc[-1]),
                'mitigation':  bool(df['mitigation_blocks'].iloc[-1]),
            },
            'volatility': float(df['volatility'].iloc[-1]),
            'volume': float(df['volume_ratio'].iloc[-1]) if 'volume_ratio' in df.columns else 0.0,
            'killzones': {
                'london':   self.is_in_killzone('london'),
                'new_york': self.is_in_killzone('new_york'),
                'tokyo':    self.is_in_killzone('tokyo'),
            },
        }

    def detect_order_blocks(self, df: pd.DataFrame) -> pd.Series:
        """Compatibility method: returns combined bullish+bearish OB mask."""
        bull, bear = self._calculate_order_blocks(df)
        return bull | bear


# =============================================================================
# BrokerDataFetcher — Universal wrapper around BaseBroker
# =============================================================================
# Provides the same get_candles() / get_multi_timeframe_data() interface as
# MT5DataFetcher but delegates all calls to a connected BaseBroker instance.
# Works with MT5Adapter, IBKRAdapter, and CTraderAdapter transparently.
# =============================================================================

import logging as _logging
from typing import Dict, List, Optional, Union

_bdf_logger = _logging.getLogger(__name__ + ".BrokerDataFetcher")


class BrokerDataFetcher:
    """
    Universal data fetcher.  Drop-in replacement for MT5DataFetcher when
    running outside of MT5 (IBKR, cTrader) or when you want a single object
    that works across all brokers.

    Parameters
    ----------
    broker : BaseBroker
        A connected broker adapter instance obtained from BrokerRouter.

    Usage
    -----
        from broker_router import BrokerRouter
        from data_fetcher  import BrokerDataFetcher

        router  = BrokerRouter()
        broker  = router.get_broker()
        fetcher = BrokerDataFetcher(broker)
        df      = fetcher.get_candles("EURUSD", "h1", 500)
    """

    def __init__(self, broker):
        self._broker = broker
        self.connected = broker.connected
        _bdf_logger.info(
            f"[BrokerDataFetcher] Attached to {broker.broker_name}"
        )

    # ------------------------------------------------------------------
    # Primary interface (same signatures as MT5DataFetcher)
    # ------------------------------------------------------------------

    def get_candles(
        self,
        symbol: str,
        timeframe: Union[str, int] = "h1",
        bars: int = 1000,
    ) -> Optional["pd.DataFrame"]:
        """
        Fetch OHLCV data for *symbol* via the connected broker adapter.

        Parameters
        ----------
        symbol    : Exact broker symbol name (already resolved).
        timeframe : Timeframe string ('m1', 'm5', 'h1', 'h4', 'd1' …) or
                    integer (MT5 constant).  Non-string integers are
                    converted to a string representation before forwarding.
        bars      : Number of historical bars to retrieve.

        Returns
        -------
        pd.DataFrame or None on error.
        """
        if not self._broker.ensure_connected():
            _bdf_logger.error(
                f"[BrokerDataFetcher] Not connected — cannot fetch {symbol}"
            )
            return None

        if isinstance(timeframe, int):
            # Map common MT5 integer constants to string equivalents
            _tf_int_map = {
                1: "m1", 5: "m5", 15: "m15", 30: "m30",
                16385: "h1", 16386: "h2", 16388: "h4",
                16408: "d1", 32769: "w1",
            }
            timeframe = _tf_int_map.get(timeframe, "h1")

        try:
            df = self._broker.get_market_data(symbol, str(timeframe).lower(), bars)
            if df is not None:
                _bdf_logger.debug(
                    f"[BrokerDataFetcher] {symbol} {timeframe}: {len(df)} bars"
                )
            return df
        except Exception as exc:
            _bdf_logger.error(
                f"[BrokerDataFetcher] get_candles({symbol}) error: {exc}",
                exc_info=True,
            )
            return None

    def get_multi_timeframe_data(
        self,
        symbol: str,
        timeframes: List[Union[str, int]],
        bars: int = 500,
    ) -> Dict[str, "pd.DataFrame"]:
        """
        Fetch data for multiple timeframes in one call.

        Returns dict keyed by timeframe string.
        """
        result: Dict[str, "pd.DataFrame"] = {}
        for tf in timeframes:
            tf_str = str(tf) if isinstance(tf, str) else self._tf_int_to_str(tf)
            df = self.get_candles(symbol, tf_str, bars)
            if df is not None:
                result[tf_str] = df
        return result

    def shutdown(self) -> None:
        """No-op — lifecycle managed by BrokerRouter / BaseBroker."""
        _bdf_logger.debug("[BrokerDataFetcher] shutdown() called (no-op).")

    def is_in_killzone(self, killzone_type: str = "london") -> bool:
        """Killzone check based on UTC wall-clock time (no broker call needed)."""
        import datetime as _dt  # timezone via _dt.timezone.utc
        now_h = _dt.datetime.now(timezone.utc).hour
        zones = {
            "london":   range(7, 10),
            "new_york": range(12, 17),
            "tokyo":    range(0, 3),
        }
        return now_h in zones.get(killzone_type, range(0, 0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tf_int_to_str(tf_int: int) -> str:
        _map = {
            1: "m1", 5: "m5", 15: "m15", 30: "m30",
            16385: "h1", 16386: "h2", 16388: "h4",
            16408: "d1", 32769: "w1",
        }
        return _map.get(tf_int, "h1")
