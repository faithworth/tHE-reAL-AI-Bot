"""
ai_ea.py — AI EA v17  UNIVERSAL MULTI-BROKER ENGINE + RANGING SCALP
====================================================

v17 upgrades
-------------
  KEY FIX: ML probability gates in REGIME_CONFIGS were set to 0.65–0.90 which
  are unreachable by a temperature-scaled (T=1.5) 3-class model where live probs
  peak at 0.45–0.65. This permanently blocked ALL trades. Thresholds recalibrated
  to 0.38–0.48 range matching the actual model output distribution.

  All version strings updated from v7/v8 to v17 for consistency.
  Composite score base threshold lowered from 0.38 → 0.36 (floor of 0.40 from
  scorer always exceeds this, so only genuinely weak signals are blocked).

v8 upgrades (retained from v8)
---------------------------------
  1. Async event loop  — asyncio.sleep() replacing time.sleep(300).
  2. Portfolio correlated risk  — RiskEngine.approve_correlated_trade().
  3. Walk-forward SL/TP optimisation  — _wf_optimizer_loop().
"""

import asyncio
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional

# Suppress sklearn/LightGBM "X does not have valid feature names" UserWarning.
# This is cosmetic noise: the fix in signal_engine.py's _ensemble_proba() passes
# named DataFrames to LGB/XGB, but any residual path (e.g. reloaded legacy models
# that pre-date the fix) should be silenced here rather than flooding the log.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)
# Suppress pandas/numpy FutureWarnings that pollute logs without affecting behaviour
warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress DeprecationWarnings from third-party libraries
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd

# ── Internal modules ──────────────────────────────────────────────────────────
from secure_config      import get_config
from symbol_mapper      import SymbolMapper
from symbol_discovery   import SymbolDiscovery
from broker_compat      import detect_broker, BrokerProfile
from broker_router      import BrokerRouter
from signal_engine      import SignalEngine
from market_structure   import MarketStructureAnalyzer
from trade_filters      import TradeFilters, is_premium_session
from risk_engine        import RiskEngine
from prop_guard         import PropGuard
from evaluator          import StrategyEvaluator
from visualizer         import TradingVisualizer

# v20: Real trade history learner — trains from every win + loss ever recorded
try:
    from trade_history_learner import TradeHistoryLearner
    HIST_LEARNER_AVAILABLE = True
except ImportError:
    HIST_LEARNER_AVAILABLE = False

# ── Graceful MT5 import (still used by broker_compat / legacy helpers) ────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

# ── v5 Advanced modules ───────────────────────────────────────────────────────
try:
    from mtf_confluence    import MTFConfluenceEngine, ConfluenceResult
    MTF_AVAILABLE = True
except ImportError:
    MTF_AVAILABLE = False

try:
    from regime_detector   import RegimeDetector, Regime, REGIME_CONFIGS
    REGIME_AVAILABLE = True
except ImportError:
    REGIME_AVAILABLE = False
    Regime = None          # type: ignore
    REGIME_CONFIGS = {}    # type: ignore

try:
    from trend_change_detector import TrendChangeDetector, DirectionBias
    TREND_CHANGE_AVAILABLE = True
except ImportError:
    TREND_CHANGE_AVAILABLE = False
    DirectionBias = None   # type: ignore
    logger.warning("trend_change_detector.py not found -- trend-reversal filter disabled!")

try:
    from feature_engineering import FeatureEngineer
    FE_AVAILABLE = True
except ImportError:
    FE_AVAILABLE = False

# ── Logging setup (MUST be before v6 PRO imports so logger exists on ImportError) ──
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    handlers=[
        logging.FileHandler("logs/ai_ea_v17.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("AI_EA_v17")

try:
    from ranging_scalper import RangingScalper, ScalpSignal, ScalpExitManager, RangeContext
    RANGE_SCALPER_AVAILABLE = True
except ImportError:
    RANGE_SCALPER_AVAILABLE = False
    RangeContext = None    # type: ignore
    logger.warning("ranging_scalper.py not found -- range scalping disabled!")

# ── v6 PRO modules ────────────────────────────────────────────────────────────
try:
    from news_filter        import NewsFilter
    NEWS_FILTER_AVAILABLE = True
except ImportError:
    NEWS_FILTER_AVAILABLE = False
    logger.warning("news_filter.py not found -- trading during news events!")

try:
    from execution_tracker  import ExecutionTracker
    EXEC_TRACKER_AVAILABLE = True
except ImportError:
    EXEC_TRACKER_AVAILABLE = False
    logger.warning("execution_tracker.py not found -- execution quality tracking disabled!")

try:
    from performance_monitor import PerformanceMonitor
    PERF_MON_AVAILABLE = True
except ImportError:
    PERF_MON_AVAILABLE = False
    logger.warning("performance_monitor.py not found -- performance monitoring disabled!")

# ── Load config from .env / environment ──────────────────────────────────────
cfg = get_config()

MT5_LOGIN    = cfg.mt5_login
MT5_PASSWORD = cfg.mt5_password
MT5_SERVER   = cfg.mt5_server

BROKER_TYPE          = os.getenv("BROKER_TYPE", "mt5").lower()
TIMEFRAME_STR        = "h1"
BARS                 = cfg.bars
SLEEP_SECS           = cfg.sleep_interval
PROP_MODE            = cfg.prop_mode
RISK_PER_TRADE       = cfg.risk_per_trade
MAX_DAILY_LOSS       = cfg.max_daily_loss
MAX_DRAWDOWN         = cfg.max_drawdown
MAX_TRADES_DAY       = cfg.max_trades_day
MAX_CONCURRENT       = cfg.max_concurrent
MIN_SIGNAL_PROB      = cfg.min_signal_prob
SL_ATR_MULT          = 1.5
TP_ATR_MULT          = 2.5
TRAILING_ATR_MULT    = 1.5   # PROFIT-FIX: trail distance at 1×profit — was 2.0 but never used
BREAKEVEN_ATR_MULT   = 1.2   # PROFIT-FIX: raised from 1.0 → give trades more room before locking BE
RETRAIN_EVERY_CYCLES = 20
MANAGE_LOOP_SECS     = int(os.getenv("MANAGE_LOOP_SECS", 15))  # dedicated BE/trail loop interval (seconds)

# ── v19 AUTO-ENGINE ───────────────────────────────────────────────────────────
try:
    from auto_engine import (
        AutoLiveRetrain,
        AutoKellySizer,
        AutoSymbolScorer,
        AutoStopLoss,
        AutoRetrainScheduler,
    )
    AUTO_EA_AVAILABLE = True
except ImportError:
    AUTO_EA_AVAILABLE = False
    logger.warning("auto_engine.py not found — v19 auto-directives disabled")
# Walk-forward SL/TP re-optimisation runs every 6 hours (21600 s)
WF_OPTIMIZE_INTERVAL = int(os.getenv("WF_OPTIMIZE_INTERVAL", 21600))


# ── Main EA class ─────────────────────────────────────────────────────────────
class AITradingEA:
    """
    Main orchestrator — Universal Multi-Broker AI Trading Engine.

    Broker selection flow
    ---------------------
    1. BROKER_TYPE env var selects mt5 / ibkr / ctrader
    2. BrokerRouter instantiates and connects the correct adapter
    3. All downstream calls go through broker.get_market_data(),
       broker.place_order(), broker.get_symbols() etc.
    4. Zero MT5-specific code in the main loop.

    Symbol resolution flow
    ----------------------
    1. Config supplies raw symbols from .env  e.g. ['XAUUSD..', 'BTCUSD..']
    2. SymbolMapper (backed by live broker list) converts each to its real
       broker form  e.g. 'XAUUSDm', 'BTCUSDm'
    3. If a config symbol cannot be resolved the EA falls back to
       SymbolDiscovery which returns ALL available broker symbols filtered
       by asset class so the bot always finds something to trade.
    4. Every broker call uses the resolved broker symbol throughout.
    """

    def __init__(self):
        logger.info("Initialising AI EA v17 Universal Multi-Broker Engine...")
        logger.info(f"  Broker type: {BROKER_TYPE.upper()}")

        # Risk / prop guard
        self.risk_engine = RiskEngine(
            risk_per_trade=RISK_PER_TRADE,
            max_daily_loss=MAX_DAILY_LOSS,
            max_drawdown=MAX_DRAWDOWN,
            max_trades_day=MAX_TRADES_DAY,
            max_concurrent=MAX_CONCURRENT,
            prop_mode=PROP_MODE,
        )
        self.prop_guard = PropGuard(self.risk_engine)

        # ── Universal broker connection ────────────────────────────────────
        self.broker_router = BrokerRouter(
            broker_type=BROKER_TYPE,
            risk_engine=self.risk_engine,
        )
        self.broker = self.broker_router.get_broker()

        # ── Symbol mapper and discovery ─────────────────────────────────────
        self.symbol_mapper    = SymbolMapper(broker=self.broker)
        self.symbol_discovery = SymbolDiscovery(broker=self.broker)

        # ── Broker profile (MT5 only; graceful fallback for other brokers) ──
        if BROKER_TYPE == "mt5" and MT5_AVAILABLE and mt5:
            self.broker_profile: BrokerProfile = detect_broker(mt5)
        else:
            self.broker_profile = BrokerProfile(
                name=self.broker.broker_name,
                server=BROKER_TYPE,
                suffix="",
                prefix="",
                filling_mode=0,
                contract_sizes={},
                point_values={},
            )

        # ── Resolve configured symbols → actual broker symbols ──────────────
        self.symbols: List[str] = self._resolve_symbols(cfg.symbols)

        # ── Remaining components ─────────────────────────────────────────────
        self.evaluator        = StrategyEvaluator()
        self.signal_engine    = SignalEngine()
        self.structure_engine = MarketStructureAnalyzer()
        self.filters          = TradeFilters(
            max_spread_pips=25.0,
            min_atr_pips=5.0,
            allowed_sessions=("london", "new_york", "pre_london"),  # Asian BLOCKED for forex/metals
            require_session=True,
        )
        self.visualizer = TradingVisualizer(refresh_secs=60)

        # v6 PRO: news filter, execution tracker, performance monitor
        self.news_filter    = NewsFilter() if NEWS_FILTER_AVAILABLE else None
        self.exec_tracker   = ExecutionTracker() if EXEC_TRACKER_AVAILABLE else None
        self.perf_monitor   = PerformanceMonitor() if PERF_MON_AVAILABLE else None

        # ── v5 Advanced engines ───────────────────────────────────────────────
        # MTFConfluenceEngine needs a data-fetcher-like object; wrap broker
        self.mtf_engine = MTFConfluenceEngine(self.broker) if MTF_AVAILABLE else None
        self.regime_det = RegimeDetector() if REGIME_AVAILABLE else None
        self.current_regime = None
        self._mtf_cache: dict = {}
        self._mtf_cache_ttl  = 300
        # v21: Trend-change / reversal detector — prevents buying in bear trends
        self.trend_change_det = TrendChangeDetector() if TREND_CHANGE_AVAILABLE else None
        self._trend_change_cache: dict = {}
        self._trend_change_cache_ttl   = 120   # 2-minute cache per symbol

        # v7: Ranging scalper — activated when RANGING_SCALP regime is detected
        self.range_scalper: Optional[object] = (
            RangingScalper(broker=self.broker) if RANGE_SCALPER_AVAILABLE else None
        )
        self.scalp_exit_mgr: Optional[object] = (
            ScalpExitManager() if RANGE_SCALPER_AVAILABLE else None
        )
        # Track open scalp positions: {ticket: {"bars_held": int, "range_ctx": ..., "entry": ...}}
        self._scalp_positions: dict = {}

        # Per-symbol signal engines
        self._signal_engines: dict = {}

        # v8: Walk-forward SL/TP optimizer (no external cost)
        try:
            from Optimizer import StrategyOptimizer
            self._wf_optimizer = StrategyOptimizer()
        except ImportError:
            self._wf_optimizer = None
            logger.warning("Optimizer.py not found — walk-forward SL/TP disabled")
        # Per-symbol optimized multipliers; fall back to global if not yet set
        self._wf_sl_mult: Dict[str, float] = {}
        self._wf_tp_mult: Dict[str, float] = {}

        self._cycle_count = 0
        self._is_running  = False
        # Tracks {ticket: {'symbol': str, 'profit': float}} for orphan P&L reconciliation
        self._known_tickets: dict = {}

        # ── v19 AUTO-DIRECTIVES ───────────────────────────────────────────────
        if AUTO_EA_AVAILABLE:
            self._kelly_sizer     = AutoKellySizer()
            self._symbol_scorer   = AutoSymbolScorer()
            self._auto_stop       = AutoStopLoss()
            self._live_retrain    = AutoLiveRetrain()
            self._retrain_sched   = AutoRetrainScheduler(news_filter=self.news_filter)
        else:
            self._kelly_sizer     = None
            self._symbol_scorer   = None
            self._auto_stop       = None
            self._live_retrain    = None
            self._retrain_sched   = None

        # v19 DIR-11: lot scale / max concurrent overrides (updated each cycle)
        self._auto_lot_scale:       float = 1.0
        self._auto_max_concurrent:  int   = MAX_CONCURRENT

        # v20: Trade history learner — load persisted stats immediately,
        # then kick off a full learn in background at first cycle.
        self._hist_learner: Optional[object] = None
        if HIST_LEARNER_AVAILABLE:
            self._hist_learner = TradeHistoryLearner()
            self._hist_learner.load_persisted(self.symbols)
            logger.info("[v20] TradeHistoryLearner initialised (persisted stats loaded).")

        logger.info(
            "All v20 components initialised. "
            f"broker={self.broker.broker_name} "
            f"MTF={MTF_AVAILABLE} Regime={REGIME_AVAILABLE} "
            f"AutoEngine={AUTO_EA_AVAILABLE} "
            f"HistLearner={HIST_LEARNER_AVAILABLE}"
        )

    # ── Symbol resolution ──────────────────────────────────────────────────────
    def _resolve_symbols(self, config_symbols: List[str]) -> List[str]:
        """
        Convert config symbol names (e.g. 'XAUUSD..') to the exact broker
        symbol names available on the connected broker.
        """
        try:
            all_syms = self.broker.get_symbols()
            broker_names = {s["name"] for s in all_syms} if all_syms else set()
        except Exception:
            logger.warning("Could not fetch broker symbol list — using config symbols as-is.")
            return list(config_symbols)

        resolved: List[str] = []
        for cfg_sym in config_symbols:
            cfg_sym = cfg_sym.strip()
            if not cfg_sym:
                continue

            clean = self.symbol_mapper.to_clean(cfg_sym)
            broker_sym = self.symbol_mapper.to_broker(clean)

            if broker_sym in broker_names and self._symbol_has_data(broker_sym):
                resolved.append(broker_sym)
                logger.info(f"Symbol resolved: {cfg_sym!r} -> {broker_sym!r}")
                continue

            if cfg_sym in broker_names and self._symbol_has_data(cfg_sym):
                resolved.append(cfg_sym)
                logger.info(f"Symbol resolved (direct): {cfg_sym!r} -> {cfg_sym!r}")
                continue

            fallback = self._find_broker_symbol(clean, broker_names)
            if fallback:
                resolved.append(fallback)
                logger.info(f"Symbol resolved (scan): {cfg_sym!r} -> {fallback!r}")
            else:
                logger.warning(
                    f"Could not resolve '{cfg_sym}' (clean='{clean}') to any "
                    f"broker symbol — will skip."
                )

        # Deduplicate preserving order
        seen: set = set()
        resolved = [s for s in resolved if not (s in seen or seen.add(s))]

        if not resolved:
            logger.warning(
                "No configured symbols could be resolved. "
                "Falling back to auto-discovery of all tradable symbols."
            )
            resolved = self.symbol_discovery.get_tradable(force_refresh=True)
            logger.info(
                f"Auto-discovered {len(resolved)} tradable symbols: "
                f"{resolved[:10]}{'...' if len(resolved) > 10 else ''}"
            )

        return resolved

    def _find_broker_symbol(self, clean_base: str, broker_names: set) -> Optional[str]:
        """Scan broker list for the best match to clean_base."""
        candidates = [
            b for b in broker_names
            if self.symbol_mapper.to_clean(b).upper() == clean_base.upper()
        ]
        for c in candidates:
            if self._symbol_has_data(c):
                return c
        return candidates[0] if candidates else None

    def _symbol_has_data(self, symbol: str) -> bool:
        """Return True if broker can deliver at least a few candles for symbol."""
        try:
            df = self.broker.get_market_data(symbol, "h1", 5)
            return df is not None and len(df) > 0
        except Exception:
            return False

    # ── Start / Stop ──────────────────────────────────────────────────────────
    def start(self) -> None:
        """Synchronous entry point — delegates to async event loop."""
        try:
            asyncio.run(self._async_main())
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")

    async def _async_main(self) -> None:
        """
        Async event loop — replaces the blocking time.sleep(300) cycle.

        Design:
        - _trading_loop()  : main cycle, runs every SLEEP_SECS via asyncio.sleep
          (non-blocking — other coroutines run during the wait)
        - _retrain_loop()  : periodic ML retraining, offset by half a cycle
        - _wf_optimizer_loop(): periodic walk-forward SL/TP re-optimisation
        Both loops run concurrently via asyncio.gather so neither blocks the other.
        """
        logger.info("=" * 70)
        logger.info("AI EA v17 ASYNC MULTI-BROKER ENGINE STARTED")
        logger.info(f"  Broker        : {self.broker.broker_name}")
        logger.info(f"  Broker type   : {BROKER_TYPE.upper()}")
        logger.info(f"  Symbols       : {self.symbols}")
        logger.info(f"  Prop mode     : {PROP_MODE}")
        logger.info(f"  Risk/trade    : {RISK_PER_TRADE*100:.1f}%")
        logger.info(f"  Daily loss lim: {MAX_DAILY_LOSS*100:.1f}%")
        logger.info(f"  Max drawdown  : {MAX_DRAWDOWN*100:.1f}%")
        logger.info(f"  Min signal    : {MIN_SIGNAL_PROB}")
        logger.info("=" * 70)

        equity = self.broker.get_equity()
        if equity > 0:
            self.risk_engine.set_equity_baseline(equity)
            logger.info(f"Starting equity: ${equity:.2f}")

        # FIX: Sync open-position state from live broker on startup so that a
        # restart after a crash/manual stop doesn't carry a stale daily_trades
        # counter or wrong _open_positions count.
        try:
            _live_count = 0
            for _sym in self.symbols:
                for _pos in (self.broker.get_open_positions(_sym) or []):
                    _t = _pos["ticket"]
                    self._known_tickets[_t] = {
                        "symbol":     _sym,
                        "profit":     float(_pos.get("profit", 0.0)),
                        "type":       _pos.get("type", "buy"),
                        "volume":     float(_pos.get("volume", 0.01)),
                        "open_price": float(_pos.get("open_price", 0.0)),
                        "open_time":  str(_pos.get("open_time", "")),
                    }
                    _live_count += 1
            # Reconcile: clamp the saved open_positions counter to reality
            saved_open = self.risk_engine.get_status(equity).get("open_positions", 0)
            if saved_open != _live_count:
                logger.info(
                    f"[STARTUP FIX] open_positions mismatch — "
                    f"saved={saved_open}, live={_live_count}. Correcting."
                )
                self.risk_engine._open_positions = _live_count
                self.risk_engine._save_state()
            logger.info(
                f"[STARTUP] Synced {_live_count} live open position(s) into _known_tickets."
            )
        except Exception as _se:
            logger.warning(f"[STARTUP] Could not sync live positions: {_se}")

        self.visualizer.start()
        self._is_running = True

        try:
            await asyncio.gather(
                self._trading_loop(),
                self._retrain_loop(),
                self._wf_optimizer_loop(),
                self._live_retrain_loop(),
                self._management_loop(),
            )
        finally:
            self.stop()

    async def _trading_loop(self) -> None:
        """Non-blocking main trading cycle using asyncio.sleep."""
        while self._is_running:
            self._cycle_count += 1
            loop = asyncio.get_event_loop()
            try:
                # run_in_executor keeps the event loop unblocked while broker
                # calls (which are synchronous C extensions on MT5) execute in
                # a thread pool.  For IBKR/cTrader native async adapters can
                # be dropped in without changing this wrapper.
                await loop.run_in_executor(None, self._run_cycle)
            except Exception as e:
                logger.error(
                    f"Unhandled exception in cycle {self._cycle_count}: {e}",
                    exc_info=True,
                )
            logger.info(f"Cycle {self._cycle_count} done — sleeping {SLEEP_SECS}s (async)...")
            await asyncio.sleep(SLEEP_SECS)

    async def _retrain_loop(self) -> None:
        """Periodic ML retraining — offset by half a cycle to avoid collision."""
        await asyncio.sleep(SLEEP_SECS // 2)
        retrain_interval = SLEEP_SECS * RETRAIN_EVERY_CYCLES
        while self._is_running:
            loop = asyncio.get_event_loop()
            # ── v19 DIR-12: Smart retrain timing ─────────────────────────────
            if self._retrain_sched is not None:
                allowed, reason = self._retrain_sched.can_retrain()
                if not allowed:
                    logger.info(f"[AUTO] Retrain deferred: {reason} — retry in 10 min")
                    await asyncio.sleep(600)
                    continue
            try:
                await loop.run_in_executor(None, self._retrain_signal_engine)
            except Exception as e:
                logger.error(f"Retrain loop error: {e}", exc_info=True)
            await asyncio.sleep(retrain_interval)

    async def _live_retrain_loop(self) -> None:
        """
        v19 DIR-8: Check every cycle if ≥10 new live trades warrant early retrain.
        """
        await asyncio.sleep(SLEEP_SECS * 2)
        while self._is_running:
            if AUTO_EA_AVAILABLE and self._live_retrain is not None:
                for symbol in self.symbols:
                    try:
                        eng = self._get_signal_engine(symbol)
                        if eng._live_buffer is None:
                            continue
                        count = eng._live_buffer.count()
                        if self._live_retrain.should_trigger(symbol, count):
                            loop = asyncio.get_event_loop()
                            logger.info(
                                f"[AUTO] DIR-8: triggering early retrain for {symbol} "
                                f"({count} live trades)"
                            )
                            await loop.run_in_executor(
                                None, lambda s=symbol: self._incremental_retrain_symbol(s)
                            )
                            self._live_retrain.mark_retrained(symbol, count)
                    except Exception as exc:
                        logger.debug(f"[AUTO] live retrain check {symbol}: {exc}")
            await asyncio.sleep(SLEEP_SECS)

    # ─────────────────────────────────────────────────────────────────────────
    # Dedicated BE / Trailing-stop management loop
    # Runs every MANAGE_LOOP_SECS (default 15 s) independently of signal
    # evaluation so SL moves are never delayed by a quiet market on a symbol.
    # ─────────────────────────────────────────────────────────────────────────
    async def _management_loop(self) -> None:
        """
        Dedicated breakeven + trailing-stop loop.

        Runs every MANAGE_LOOP_SECS seconds for ALL open positions across ALL
        tracked symbols, completely independent of whether a new signal fires.
        This fixes the core bug where BE/trail only triggered inside the signal
        path (i.e. never when the market was quiet or ranging).
        """
        await asyncio.sleep(5)          # slight initial delay so broker connects first
        loop = asyncio.get_event_loop()
        while self._is_running:
            try:
                await loop.run_in_executor(None, self._manage_all_positions)
            except Exception as exc:
                logger.debug(f"[MGMT] management loop error: {exc}")
            await asyncio.sleep(MANAGE_LOOP_SECS)

    def _manage_all_positions(self) -> None:
        """
        Iterate every symbol that has open positions and run
        _manage_existing_positions with fresh price + structure data.

        Fetching a lightweight DataFrame every 15 s for ATR only.
        We reuse the per-symbol signal-engine so the ATR calculation
        is consistent with the rest of the system.
        """
        try:
            all_positions = self.broker.get_open_positions()
        except Exception as exc:
            logger.debug(f"[MGMT] get_open_positions error: {exc}")
            return

        if not all_positions:
            return

        # Deduplicate by symbol so we only fetch data once per symbol
        symbols_with_positions = list({p["symbol"] for p in all_positions})

        for symbol in symbols_with_positions:
            try:
                df = self._fetch_df_for_management(symbol)
                if df is None or df.empty:
                    continue
                if "atr" not in df.columns or float(df["atr"].iloc[-1]) <= 0:
                    continue

                # Minimal structure dict — CHoCH detection needs real signal data.
                # We pass empty structure so CHoCH early-close is skipped safely;
                # full structure is still used during normal signal-path calls.
                structure: dict = {}

                self._manage_existing_positions(symbol, df, structure)

            except Exception as exc:
                logger.debug(f"[MGMT] manage {symbol} error: {exc}")

    def _fetch_df_for_management(self, symbol: str):
        """
        Fetch a lightweight DataFrame (50 bars, primary timeframe) for ATR only.
        Returns None on failure.
        """
        import pandas as pd
        try:
            eng = self._get_signal_engine(symbol)
            tf  = getattr(eng, "primary_tf", "M5")
            df  = self.broker.get_ohlcv(symbol, tf, 50)
            if df is None or len(df) < 14:
                return None
            high      = df["high"].astype(float)
            low       = df["low"].astype(float)
            close     = df["close"].astype(float)
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low  - prev_close).abs(),
            ], axis=1).max(axis=1)
            df["atr"] = tr.ewm(span=14, adjust=False).mean()
            return df
        except Exception as exc:
            logger.debug(f"[MGMT] _fetch_df_for_management({symbol}) error: {exc}")
            return None

    async def _wf_optimizer_loop(self) -> None:
        """Walk-forward SL/TP optimisation — runs every WF_OPTIMIZE_INTERVAL seconds."""
        # Offset by 3/4 of a cycle so it doesn't clash with training
        await asyncio.sleep(int(SLEEP_SECS * 0.75))
        while self._is_running:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, self._run_wf_optimization)
            except Exception as e:
                logger.error(f"WF optimizer loop error: {e}", exc_info=True)
            await asyncio.sleep(WF_OPTIMIZE_INTERVAL)

    def stop(self) -> None:
        self._is_running = False
        self.visualizer.stop()
        self.broker_router.shutdown()
        logger.info("AI EA v17 stopped.")

    # ── Main cycle ────────────────────────────────────────────────────────────
    def _run_cycle(self) -> None:
        logger.info(
            f"-- Cycle #{self._cycle_count} @ {datetime.now().strftime('%H:%M:%S')} "
            f"[{self.broker.broker_name}] --"
        )

        if not self.broker.ensure_connected():
            logger.error(f"Cannot connect to {self.broker.broker_name} — skipping cycle.")
            return

        equity = self.broker.get_equity()
        if equity <= 0:
            logger.error("Cannot read equity — skipping cycle.")
            return

        self._reconcile_closed_positions(equity)

        # FIX: Sync risk engine's _open_positions to the actual live count each
        # cycle. This prevents drift where the counter gets out of sync due to
        # externally closed positions, restarts, or broker disconnects.
        try:
            _actual_open = sum(
                len(self.broker.get_open_positions(s) or [])
                for s in self.symbols
            )
            if self.risk_engine._open_positions != _actual_open:
                logger.debug(
                    f"[RISK SYNC] open_positions: "
                    f"counter={self.risk_engine._open_positions} → live={_actual_open}"
                )
                self.risk_engine._open_positions = _actual_open
        except Exception:
            pass

        self.risk_engine.set_equity_baseline(equity)
        risk_status = self.risk_engine.get_status(equity)

        if risk_status["emergency_stop"]:
            logger.critical("EMERGENCY STOP ACTIVE — no trading.")
            return

        # ── v19 DIR-11: AutoStopLoss tiered drawdown protection ──────────────
        if AUTO_EA_AVAILABLE and self._auto_stop is not None:
            self._auto_stop.set_session_equity(equity)
            lot_scale_factor, effective_max_conc, day_stopped = self._auto_stop.evaluate(
                equity, MAX_CONCURRENT
            )
            self._auto_lot_scale = lot_scale_factor
            if day_stopped:
                logger.critical(
                    "[AUTO] DIR-11: Daily loss limit reached — closing all positions, halting today."
                )
                for sym in self.symbols:
                    try:
                        for pos in (self.broker.get_open_positions(sym) or []):
                            self.broker.close_order(pos["ticket"], symbol=sym)
                    except Exception:
                        pass
                return
            self._auto_max_concurrent = effective_max_conc
        else:
            self._auto_lot_scale = 1.0
            self._auto_max_concurrent = MAX_CONCURRENT

        # v20: TradeHistoryLearner — run on first cycle only.
        # IMPORTANT: broker.get_trade_history() MUST be called here (in the
        # executor / main cycle thread) NOT in a separate background thread,
        # because MT5's Python API is single-threaded and calling it from two
        # threads simultaneously causes a silent disconnect.
        # We fetch the raw history synchronously here, then hand the data to
        # the learner which does all the CPU-heavy stat computation in a
        # daemon thread so it doesn't block the cycle.
        if HIST_LEARNER_AVAILABLE and self._hist_learner is not None:
            if not getattr(self, "_hist_learn_done", False):
                self._hist_learn_done = True   # set immediately — prevent re-entry
                try:
                    raw_history = self.broker.get_trade_history(days=365)
                except Exception as _he:
                    logger.warning(f"[v20] Could not fetch broker trade history: {_he}")
                    raw_history = []

                import threading as _threading
                def _bg_process(hist):
                    try:
                        self._hist_learner.run_full_learn(
                            broker=None,           # broker already called above
                            symbols=self.symbols,
                            prefetched_history=hist,
                        )
                    except Exception as ex:
                        logger.warning(f"[v20] HistLearner processing error: {ex}")
                _threading.Thread(
                    target=_bg_process, args=(raw_history,),
                    daemon=True, name="hist_learner"
                ).start()

        # ── v19 DIR-10: refresh symbol active set once per cycle ──────────────
        if AUTO_EA_AVAILABLE and self._symbol_scorer is not None:
            self._symbol_scorer.refresh_active_set(self.symbols)

        self.visualizer.update_equity(equity)
        self.visualizer.update_risk_status(risk_status)

        logger.info(
            f"Account: equity=${equity:.2f} | "
            f"daily_trades={risk_status['daily_trades']}/{risk_status['max_trades_day']} | "
            f"daily_pnl=${risk_status['daily_pnl']:.2f} | "
            f"drawdown={risk_status['drawdown_pct']:.2f}%"
        )

        for symbol in self.symbols:
            try:
                self._process_symbol(symbol, equity)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    def _get_signal_engine(self, symbol: str):
        """Get or create per-symbol signal engine (v7)."""
        if symbol not in self._signal_engines:
            from signal_engine import SignalEngine
            self._signal_engines[symbol] = SignalEngine(symbol=symbol)
        return self._signal_engines[symbol]

    def _get_mtf_confluence(self, symbol: str):
        """Cache-aware MTF confluence fetch (TTL 5 min)."""
        # FIX: use top-level `import time` — no hot-path import needed
        now = time.time()
        if symbol in self._mtf_cache:
            result, ts = self._mtf_cache[symbol]
            if now - ts < self._mtf_cache_ttl:
                return result
        if self.mtf_engine:
            result = self.mtf_engine.get_confluence(symbol)
            self._mtf_cache[symbol] = (result, now)
            return result
        return None

    def _get_trend_change(self, symbol: str, df_h1, df_h4=None, df_m15=None):
        """Cache-aware TrendChangeDetector fetch (TTL 2 min). v21."""
        if not TREND_CHANGE_AVAILABLE or self.trend_change_det is None:
            return None
        now = time.time()
        if symbol in self._trend_change_cache:
            result, ts = self._trend_change_cache[symbol]
            if now - ts < self._trend_change_cache_ttl:
                return result
        result = self.trend_change_det.analyse(df_h1, df_h4=df_h4, df_m15=df_m15)
        self._trend_change_cache[symbol] = (result, now)
        return result

    def _update_regime(self, df, df_h4=None) -> None:
        """Detect regime and update scorer/risk weights. v7: passes H4 for range quality scoring."""
        if not self.regime_det:
            return
        try:
            snap = self.regime_det.detect(df, df_h4=df_h4)
            if snap.regime != self.current_regime:
                logger.info(f"[REGIME] Change: {self.current_regime} -> {snap.regime.value}"
                            + (f" | scalp_mode=ON range=[{snap.range_low:.5f},{snap.range_high:.5f}]"
                               if snap.scalp_mode else ""))
                self.current_regime = snap.regime
            # Store latest snap for use by _process_symbol
            self._latest_regime_snap = snap
            if snap.config:
                cfg = snap.config
                if not cfg.trade_allowed:
                    logger.info(f"[REGIME] {snap.regime.value} — trading suspended.")
                    return
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")

    def _process_symbol(self, symbol: str, equity: float) -> None:
        logger.info(f"  -> Processing {symbol}")

        # ── v19 DIR-10: only trade top-60% symbols ────────────────────────────
        if (AUTO_EA_AVAILABLE and self._symbol_scorer is not None
                and not self._symbol_scorer.is_active(symbol, self.symbols)):
            logger.info(f"  {symbol}: inactive (below top-60% composite score) — skipping")
            return

        # ── 7-tier H1 data: 365d macro → 90d context → 31d regime → 14d structure
        #                   → 7d session → 3d precision → 1d ultra-precision ─────
        # BARS (from .env) drives the DEEP fetch. We slice downward:
        #   df_365d — full macro window for ML feature richness (all BARS)
        #   df_90d  — 3-month context (2160 bars)
        #   df_31d  — last 31 days (744 bars) for regime detection
        #   df_14d  — last 14 days (336 bars) for structural context
        #   df_7d   — last 7 days (168 bars) for session-level signals
        #   df_3d   — last 3 days (72 bars) for precision entry
        #   df_1d   — last 24 bars (1d) for ultra-precision M10/M15 triggers
        # Regime uses df_31d; predict() uses df_7d; train() uses df_365d.
        df_365d = self.broker.get_market_data(symbol, TIMEFRAME_STR, BARS)
        if df_365d is None or len(df_365d) < 50:
            logger.warning(f"  Insufficient data for {symbol} — skipping.")
            return

        df_365d = self.evaluator.add_market_indicators(df_365d)

        # Slice all 7 tiers from the single fetch
        df_90d  = df_365d.iloc[-2160:].copy() if len(df_365d) >= 2160  else df_365d
        df_31d  = df_365d.iloc[-744:].copy()  if len(df_365d) >= 744   else df_365d
        df_14d  = df_365d.iloc[-336:].copy()  if len(df_365d) >= 336   else df_365d
        df      = df_365d.iloc[-168:].copy()  if len(df_365d) >= 168   else df_365d  # 7d (signal)
        df_3d   = df_365d.iloc[-72:].copy()   if len(df_365d) >= 72    else df_365d
        df_1d   = df_365d.iloc[-24:].copy()   if len(df_365d) >= 24    else df_365d

        logger.debug(
            f"  {symbol} data tiers: 365d={len(df_365d)}bars "
            f"90d={len(df_90d)}bars 31d={len(df_31d)}bars 14d={len(df_14d)}bars "
            f"7d={len(df)}bars 3d={len(df_3d)}bars 1d={len(df_1d)}bars"
        )

        # v5: Fetch H4 for MTF (also needed by v7 regime detector)
        df_h4 = None
        try:
            df_h4 = self.broker.get_market_data(symbol, "h4", 500)
        except Exception:
            pass

        # Regime detection — use 31d slice for more stable ADX/vol context
        # FIX: Detect regime locally per symbol — do NOT mutate self.current_regime
        # here because that shared state bleeds into other symbols processed in the
        # same cycle (e.g. XAUUSD=RANGING then BTCUSD inherits RANGING weights).
        # We update self.current_regime only as a best-effort last-seen reference.
        self._update_regime(df_31d, df_h4=df_h4)
        # Capture per-symbol regime so it doesn't get overwritten mid-cycle
        sym_regime = self.current_regime

        # ── v7: RANGING / RANGING_SCALP → scalper branch ────────────────────────
        # RANGING_SCALP: high-quality confirmed range — try scalper first.
        # Plain RANGING: try scalper; if scalper finds no entry, fall through to
        # ML trend path (price may be breaking out of the range or setting up).
        # This prevents the hard-block where ALL ranging symbols get zero trades.
        if (RANGE_SCALPER_AVAILABLE
                and self.range_scalper is not None
                and REGIME_AVAILABLE
                and sym_regime is not None):
            if sym_regime == Regime.RANGING_SCALP:
                # High-quality range: scalper only (well-confirmed range extremes)
                self._process_scalp(symbol, df, df_h4, equity)
                return   # scalp path handled; don't fall through to trend logic
            elif sym_regime == Regime.RANGING:
                # Plain ranging: try scalper first; fall through to ML if no entry.
                # We check analyse() result here to decide routing without side effects.
                # _process_scalp will re-run analyse() internally but that's acceptable.
                from ranging_scalper import ScalpSignal as _SS
                _scalp_probe = self.range_scalper.analyse(symbol, df, df_h4)
                if _scalp_probe.signal != _SS.NO_TRADE:
                    # Valid scalp entry exists — route to full scalp flow
                    self._process_scalp(symbol, df, df_h4, equity)
                    return
                # No scalp setup at range extremes — fall through to ML trend path
                logger.info(f"  [{symbol}] Ranging: no scalp at extremes, trying ML trend path")
        # ─────────────────────────────────────────────────────────────────────

        # Trade filters — get latest price from broker
        atr_val = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
        spread_pips = 0.0
        point_v     = 0.00001  # default forex point

        try:
            sym_info_dict = self.broker.get_symbol_info(symbol)
            if sym_info_dict:
                point_v = float(sym_info_dict.get("point", 0.00001))
                raw_spread = float(sym_info_dict.get("spread", 0))
                broker_type = getattr(self.broker, "broker_name", "").lower()
                if "ibkr" in broker_type or "ctrader" in broker_type or "spotware" in broker_type:
                    # spread is already in price terms (ask - bid)
                    pip_size = point_v * 10
                    spread_pips = raw_spread / pip_size if pip_size > 0 else 0.0
                else:
                    # MT5: spread field is in POINTS (integer ticks).
                    # Convert to pips: pips = points / 10  (universal MT5 convention).
                    # BUT for BTC/crypto where point can be 0.01 or 1.0 depending on
                    # broker, raw_spread in points can be eg 2090 → 209 pips ($209 at $1/pip).
                    # We cross-check against the asset-class limit in dollars instead
                    # of pips so broker-specific point sizes don't cause false blocks.
                    spread_pips = raw_spread / 10.0

            else:
                # Fallback: live price ask-bid
                latest = self.broker.get_latest_price(symbol)
                if latest:
                    price_spread = float(latest.get("spread", 0))
                    pip_size = point_v * 10
                    spread_pips = price_spread / pip_size if pip_size > 0 else 0.0
        except Exception:
            pass

        # ATR is always in price terms; convert to pips: price / (point * 10)
        pip_size_v = point_v * 10
        atr_pips = atr_val / pip_size_v if pip_size_v > 0 else 0.0

        filters_ok, filter_reason = self.filters.check_all(
            spread_pips=spread_pips,
            atr_pips=atr_pips,
            symbol=symbol,
            point=point_v,          # enables price-based ATR check in TradeFilters
        )
        if not filters_ok:
            logger.info(f"  {symbol} FILTERED: {filter_reason}")
            return

        # v5: MTF confluence
        mtf_result = self._get_mtf_confluence(symbol)
        if mtf_result:
            logger.info(
                f"  {symbol} MTF: bias={mtf_result.bias} score={mtf_result.score:.3f} "
                f"htf={mtf_result.htf_aligned} ltf={mtf_result.ltf_confirmed} "
                f"reasons={mtf_result.reasons[:3]}"
            )
            # v20-FIX: Only hard-skip if bias is neutral AND score is truly zero
            # (score=0.07 means ONLY killzone_active — no tier agreement at all).
            # score=0.000 means not even in a killzone — genuinely nothing there.
            # Do NOT skip on score=0.07 (killzone-only); that was killing XAUUSD entirely.
            no_htf = not getattr(mtf_result, "htf_aligned", False)
            if mtf_result.bias == "neutral" and mtf_result.score == 0.0 and no_htf:
                logger.info(
                    f"  {symbol} -> MTF confluence too weak "
                    f"(neutral bias, no HTF alignment, score={mtf_result.score:.3f}), skipping"
                )
                return

        # v5: Per-symbol signal engine
        # v6 PRO: News filter check
        if self.news_filter:
            blocked, news_reason = self.news_filter.is_blocked(symbol=symbol)
            if blocked:
                logger.info(f"  {symbol} -> {news_reason} (skipping)")
                return

        # v6 PRO: Performance monitor halt check
        if self.perf_monitor and self.perf_monitor.is_symbol_halted(symbol):
            logger.warning(f"  {symbol} -> HALTED by PerformanceMonitor (rolling DD exceeded)")
            return

        # df_h4 already fetched above via broker.get_market_data() — used directly below.

        sym_engine = self._get_signal_engine(symbol)

        # v8 FIX: Pass df_365d (full context) not df (7d slice).
        # The model was trained on the full window feature distribution; using a 7d
        # slice causes severe distribution shift that pushes nearly all outputs
        # to NO_TRADE (the model has never seen that input range during training).
        # predict_full() is a single-pass version — avoids double feature computation.
        signal, prob, prob_dict = sym_engine.predict_full(df_365d, df_h4=df_h4, mtf_result=mtf_result)

        # v6 PRO: Adaptive retraining check
        if sym_engine.needs_retraining():
            logger.warning(f"  [{symbol}] ML accuracy degraded -- flag for retraining")
        logger.info(
            f"  {symbol} ML signal: {signal} (prob={prob:.3f}) | "
            f"BUY={prob_dict['BUY']:.3f} SELL={prob_dict['SELL']:.3f}"
        )

        if signal == "NO_TRADE":
            logger.info(f"  {symbol} -> NO_TRADE (prob too low)")
            return

        # ── v21: Trend-change / reversal direction filter ─────────────────────
        # Prevents "buying twice in a downtrend" (see US30 chart: two failed BUYs
        # during a clear bear leg). Run BEFORE structure / score to short-circuit early.
        tc_snap = self._get_trend_change(symbol, df, df_h4=df_h4)
        if tc_snap is not None and TREND_CHANGE_AVAILABLE and DirectionBias is not None:
            if tc_snap.block_buy and signal == "BUY":
                if tc_snap.direction.value == "bear":
                    logger.info(
                        f"  {symbol} -> TrendChange BEAR confirmed — BUY blocked "
                        f"(conf={tc_snap.confidence:.2f} choch={tc_snap.choch} "
                        f"bos={tc_snap.bos} reasons={tc_snap.reasons[:3]})"
                    )
                    return
                elif tc_snap.direction.value == "transition_to_bear":
                    logger.info(
                        f"  {symbol} -> TrendChange TRANSITION_TO_BEAR — BUY requires "
                        f"very high score (conf={tc_snap.confidence:.2f})"
                    )
                    # Don't block outright — but apply heavy score penalty below
                    # (handled via _tc_penalty flag read in score section)
            if tc_snap.block_sell and signal == "SELL":
                if tc_snap.direction.value == "bull":
                    logger.info(
                        f"  {symbol} -> TrendChange BULL confirmed — SELL blocked "
                        f"(conf={tc_snap.confidence:.2f} choch={tc_snap.choch} "
                        f"bos={tc_snap.bos} reasons={tc_snap.reasons[:3]})"
                    )
                    return
                elif tc_snap.direction.value == "transition_to_bull":
                    logger.info(
                        f"  {symbol} -> TrendChange TRANSITION_TO_BULL — SELL requires "
                        f"very high score (conf={tc_snap.confidence:.2f})"
                    )
        # ─────────────────────────────────────────────────────────────────────

        # Market structure
        structure = self.structure_engine.analyse_with_mtf(df, mtf_result=mtf_result)
        logger.info(
            f"  {symbol} structure: trend={structure['trend']} | "
            f"bos={structure['bos']} | choch={structure['choch']} | "
            f"liq_sweep={structure['liquidity_sweep']} | "
            f"score={structure['structure_score']:.2f}"
        )

        aligned = self.structure_engine.is_trade_aligned_with_structure(signal, structure)

        # v5: Composite score
        score = self._score_signal_v7(signal, prob, structure, mtf_result)

        if not aligned:
            # FIX: penalty was logged but never applied — actually subtract it now.
            # 0.15 is meaningful on the 0–1 composite scale and will filter
            # counter-trend signals with neutral MTF / ranging structure (the main
            # source of bad trades in the log: prob=0.43, MTF=neutral, ranging).
            score = round(max(0.0, score - 0.15), 4)
            logger.info(
                f"  {symbol} -> signal {signal} vs structure ({structure['trend']}) "
                f"— counter-trend, score penalised by 0.15 → {score:.3f}"
            )

        # v21: Additional score penalty for TRANSITION regime (against-trend trades)
        if (tc_snap is not None and TREND_CHANGE_AVAILABLE and DirectionBias is not None):
            _is_transition_against = (
                (tc_snap.direction.value == "transition_to_bear" and signal == "BUY") or
                (tc_snap.direction.value == "transition_to_bull" and signal == "SELL")
            )
            if _is_transition_against:
                _tc_penalty = round(tc_snap.confidence * 0.20, 4)  # up to -0.20
                score = round(max(0.0, score - _tc_penalty), 4)
                logger.info(
                    f"  {symbol} -> TrendChange TRANSITION penalty -{_tc_penalty:.3f} "
                    f"(conf={tc_snap.confidence:.2f}) → score={score:.3f}"
                )

        logger.info(f"  {symbol} composite score={score:.3f}")

        # v16 FIX (Bug 1): Decouple REGIME_CONFIGS.min_signal_prob from the composite
        # score gate.  rcfg.min_signal_prob is calibrated for raw 3-class ML probability
        # (range 0.65–0.90).  It was being capped at 0.55 and used as the composite
        # score threshold, but composite scores live on a completely different scale
        # (ML@prob=0.40 → 16pts + MTF_neutral=10pts + trend=7 + struct=3 + sess=4 = 40/100
        # = 0.40), so a 0.55 gate blocks virtually every valid signal.
        #
        # Correct behaviour:
        #   1. ML probability gate  — applied HERE using rcfg.min_signal_prob (raw prob scale).
        #   2. Composite score gate — fixed at 0.38 base, nudged ±0.03 by regime.
        #      Never driven by min_signal_prob.

        # Step 1: ML prob gate (raw probability, regime-aware)
        ml_prob_gate = MIN_SIGNAL_PROB   # module-level default (0.36)
        if self.current_regime is not None and REGIME_AVAILABLE:
            rcfg = REGIME_CONFIGS.get(self.current_regime) if REGIME_CONFIGS else None
            if rcfg:
                # rcfg.min_signal_prob is the raw-probability gate — use it directly.
                ml_prob_gate = float(rcfg.min_signal_prob)
        if prob < ml_prob_gate:
            logger.info(
                f"  {symbol} -> ML prob too low ({prob:.3f} < {ml_prob_gate:.3f}), skipping"
            )
            return

        # Step 2: Composite score gate (independent of ML prob scale)
        # v20-FIX: Base lowered from 0.43 to 0.38. The previous 0.43 base + off-hours
        # penalty of 0.04 = 0.47 minimum, which blocked the vast majority of valid signals.
        # The composite score uses a 0.0–1.0 scale where 0.38 represents real confluence.
        min_score_threshold = 0.38
        if self.current_regime is not None and REGIME_AVAILABLE:
            rcfg = REGIME_CONFIGS.get(self.current_regime) if REGIME_CONFIGS else None
            if rcfg:
                regime_name = getattr(self.current_regime, "value",
                                      str(self.current_regime)).lower()
                if any(k in regime_name for k in ("volatile", "trending_volatile")):
                    min_score_threshold = 0.42
                elif any(k in regime_name for k in ("calm", "low_vol", "ranging")):
                    min_score_threshold = 0.36

        # Off-hours penalty: only +0.02 (was +0.04 which was too aggressive)
        if not is_premium_session():
            min_score_threshold += 0.02
            logger.debug(f"  {symbol} off-hours score penalty applied → threshold={min_score_threshold:.2f}")

        if score < min_score_threshold:
            logger.info(
                f"  {symbol} -> composite score too low ({score:.3f} < {min_score_threshold:.3f}), skipping"
            )
            return

        # Prop-guard check
        open_cnt = self.broker.count_open_positions()
        guard_ok, guard_reason = self.prop_guard.check(
            equity=equity,
            open_positions=open_cnt,
            signal_prob=prob,
            symbol=symbol,
        )
        if not guard_ok:
            logger.info(f"  {symbol} PROP GUARD blocked: {guard_reason}")
            return

        # v20: History-learner filter — block bad hours/days based on real trade history
        if HIST_LEARNER_AVAILABLE and self._hist_learner is not None:
            from datetime import datetime as _dt
            _now = _dt.now()
            hist_ok, hist_reason = self._hist_learner.suggest_filter(
                symbol,
                hour=_now.hour,
                weekday=_now.weekday(),
            )
            if not hist_ok:
                logger.info(f"  {symbol} HIST LEARNER blocked: {hist_reason}")
                return
            # Bias score: if strongly negative history, require higher signal prob
            _bias = self._hist_learner.bias_score(symbol)
            if _bias < -0.3 and prob < (MIN_SIGNAL_PROB + 0.05):
                logger.info(
                    f"  {symbol} HIST LEARNER: bias={_bias:+.2f} → "
                    f"requires higher confidence (prob={prob:.3f})"
                )
                return

        # FIX: risk_engine.approve_trade() is the authoritative gate —
        # it enforces daily loss, drawdown, cooldown, concurrent limits.
        # It was present in RiskEngine but never called in the main loop.
        risk_ok, risk_reason = self.risk_engine.approve_trade(
            equity=equity,
            open_positions=open_cnt,
            symbol=symbol,
            signal_prob=prob,
        )
        if not risk_ok:
            logger.info(f"  {symbol} RISK ENGINE blocked: {risk_reason}")
            return

        # Manage existing positions first
        self._manage_existing_positions(symbol, df, structure)

        # Lot sizing — use sym_info from broker (MUST be before corr check)
        sym_info_dict = self.broker.get_symbol_info(symbol) or {}
        contract_size = float(sym_info_dict.get("contract_size", 100_000))
        symbol_point  = float(sym_info_dict.get("point", 0.00001))

        # v6 PRO: composite size multipliers from execution quality and performance
        exec_mult = self.exec_tracker.get_size_adjustment(symbol) if self.exec_tracker else 1.0
        perf_mult = self.perf_monitor.get_size_multiplier(symbol) if self.perf_monitor else 1.0
        if exec_mult < 1.0:
            logger.info(f"  {symbol} size mult exec={exec_mult:.2f} perf={perf_mult:.2f}")

        lot = self.risk_engine.calculate_lot_size(
            equity=equity,
            atr=atr_val,
            min_lot=float(sym_info_dict.get("min_lot", 0.01)),
            max_lot=float(sym_info_dict.get("max_lot", 1.0)),
            symbol_point=symbol_point,
            contract_size=contract_size,
        )

        # ── v19 DIR-9: Kelly criterion sizing overrides fixed ATR formula ─────
        if AUTO_EA_AVAILABLE and self._kelly_sizer is not None:
            sym_eng_k = self._get_signal_engine(symbol)
            pnl_hist  = list(sym_eng_k._live_pnl_history)
            risk_st   = self.risk_engine.get_status(equity)
            dd_pct    = float(risk_st.get("drawdown_pct", 0.0)) / 100.0
            kelly_lot = self._kelly_sizer.calculate(
                equity=equity,
                atr=atr_val,
                pnl_history=pnl_hist,
                drawdown_pct=dd_pct,
                symbol=symbol,
                min_lot=float(sym_info_dict.get("min_lot", 0.01)),
                max_lot=float(sym_info_dict.get("max_lot", 1.0)),
                contract_size=contract_size,
            )
            lot = kelly_lot

        # ── v19 DIR-11: apply lot scale factor from AutoStopLoss ─────────────
        lot = round(lot * getattr(self, "_auto_lot_scale", 1.0), 2)

        lot = round(lot * exec_mult * perf_mult, 2)
        if lot <= 0:
            logger.warning(f"  {symbol} -> lot reduced to zero by quality/perf multipliers, skipping")
            return

        # v8: Portfolio correlated-group risk gate — AFTER lot is calculated
        corr_ok, corr_reason = self.risk_engine.approve_correlated_trade(
            symbol=symbol,
            proposed_lot=lot,
            equity=equity,
            atr=atr_val,
            symbol_point=symbol_point,
            contract_size=contract_size,
        )
        if not corr_ok:
            logger.info(f"  {symbol} CORR RISK blocked: {corr_reason}")
            return

        # Duplicate trade prevention
        existing = self.broker.get_open_positions(symbol)
        if any(p["type"] == signal.lower() for p in existing):
            logger.info(f"  {symbol} -> duplicate {signal} position already open, skipping")
            return

        # Regime-aware SL/TP multipliers
        sl_mult_use = SL_ATR_MULT
        tp_mult_use = TP_ATR_MULT
        if self.current_regime is not None and REGIME_AVAILABLE:
            rcfg = REGIME_CONFIGS.get(self.current_regime) if REGIME_CONFIGS else None
            if rcfg:
                sl_mult_use = rcfg.sl_atr_mult
                tp_mult_use = rcfg.tp_atr_mult
                logger.info(f"  [{self.current_regime.value}] regime: SL_mult={sl_mult_use} TP_mult={tp_mult_use}")

        # v8: Walk-forward optimized multipliers override regime defaults
        if symbol in self._wf_sl_mult:
            sl_mult_use = self._wf_sl_mult[symbol]
            tp_mult_use = self._wf_tp_mult.get(symbol, tp_mult_use)
            logger.debug(f"  [{symbol}] WF-optimised SL={sl_mult_use} TP={tp_mult_use}")

        if mtf_result and mtf_result.invalidation > 0:
            logger.info(f"  MTF invalidation={mtf_result.invalidation:.5f} target1R={mtf_result.target_1r:.5f}")

        # Place order via universal broker interface
        # NOTE: mt5_adapter.place_order() builds its own short pure-alphanumeric
        # safe_comment from signal_prob + direction.  The 'comment' passed here is
        # used only for the internal trade log record — it does NOT reach MT5 directly.
        # We still sanitise it so the JSON log stays clean.
        import re as _re
        _raw_comment = (
            f"AI_EA_v17_{symbol[:6]}_sc{score:.2f}_mtf{mtf_result.score:.2f}"
            if mtf_result else f"AI_EA_v17_score{score:.2f}"
        )
        comment = _re.sub(r"[^a-zA-Z0-9_ \-]", "", _raw_comment)[:27]

        # v6 PRO FIX 3: record expected price BEFORE order so record_fill() can compute slippage
        if self.exec_tracker:
            try:
                latest_pre = self.broker.get_latest_price(symbol)
                if latest_pre:
                    expected_price = float(
                        latest_pre.get("ask", 0.0) if signal == "BUY"
                        else latest_pre.get("bid", 0.0)
                    )
                    self.exec_tracker.record_signal(
                        symbol=symbol,
                        direction=signal,
                        expected_price=expected_price,
                    )
            except Exception as _ex:
                logger.debug(f"  exec_tracker.record_signal failed: {_ex}")

        result = self.broker.place_order(
            symbol=symbol,
            order_type=signal.lower(),
            volume=lot,
            atr=atr_val,
            sl_atr_mult=sl_mult_use,
            tp_atr_mult=tp_mult_use,
            signal_prob=prob,
            comment=comment,
        )

        if result:
            logger.info(
                f"  [OK] {symbol} {signal} order placed: lot={lot} | score={score:.3f}"
            )
            # FIX: record open trade in risk engine so daily count + concurrent
            # limits are enforced correctly on subsequent symbols this cycle.
            self.risk_engine.record_trade_open()
            self.risk_engine.record_open_lot(symbol, lot)  # v8: corr-group tracking
            self.visualizer.add_trade(result)
            self.visualizer.update_price_data(symbol, df)
            # v6 PRO: record signal for execution quality tracking
            if self.exec_tracker and result:
                fill_price = result.get("price", 0.0)
                sym_point  = float((self.broker.get_symbol_info(symbol) or {}).get("point", 0.00001))
                self.exec_tracker.record_fill(symbol, actual_fill=fill_price, point=sym_point)
            # ── Live-trade learning: capture entry feature snapshot ────────────
            # We store the feature vector now (at open) so that when this
            # position closes we can write (features → outcome) to disk.
            ticket = str(result.get("ticket", result.get("order", "")))
            if ticket:
                try:
                    sym_eng_snap = self._get_signal_engine(symbol)
                    sym_eng_snap.capture_entry_features(
                        ticket=ticket,
                        df=df_365d,
                        direction=signal,
                        prob=prob,
                        score=score,
                        df_h4=df_h4,
                        mtf_result=mtf_result,
                    )
                    # Also register ticket in _known_tickets with entry metadata
                    self._known_tickets[ticket] = {
                        "symbol":  symbol,
                        "profit":  0.0,
                        "type":    signal.lower(),
                        "volume":  lot,
                        "ticket":  ticket,
                    }
                except Exception as _snap_err:
                    logger.debug(f"  entry snapshot error: {_snap_err}")
        else:
            logger.warning(
                f"  [X] {symbol} order NOT placed — broker.place_order() returned None. "
                f"Check [MT5] ORDER FAILED log above for retcode and comment."
            )

    # ── v7: Range Scalp processing ────────────────────────────────────────────
    def _process_scalp(
        self, symbol: str, df_h1: pd.DataFrame,
        df_h4: Optional[pd.DataFrame], equity: float
    ) -> None:
        """
        Full ranging-market scalp flow:
          1. Manage any existing scalp positions first (quick-exit logic)
          2. Run RangingScalper to find LTF entry at range extreme
          3. Apply prop/risk guards with scalp-appropriate sizing
          4. Place scalp order with partial-close TP structure
        """
        logger.info(f"  [{symbol}] RANGING_SCALP mode — drilling into LTFs...")

        # ── 1. Manage existing scalp positions ──────────────────────────────
        self._manage_scalp_positions(symbol, df_h1, df_h4)

        # ── Filters still apply ──────────────────────────────────────────────
        if self.news_filter:
            blocked, news_reason = self.news_filter.is_blocked(symbol=symbol)
            if blocked:
                logger.info(f"  [{symbol}] News filter blocked scalp: {news_reason}")
                return

        if self.perf_monitor and self.perf_monitor.is_symbol_halted(symbol):
            logger.warning(f"  [{symbol}] Halted by PerformanceMonitor — no scalp")
            return

        # Duplicate check — don't stack scalps
        existing = self.broker.get_open_positions(symbol)
        if len(existing) >= 2:
            logger.info(f"  [{symbol}] Already {len(existing)} open positions — skip scalp")
            return

        # ── 2. RangingScalper analysis ───────────────────────────────────────
        entry = self.range_scalper.analyse(symbol, df_h1, df_h4)

        if entry.signal == ScalpSignal.NO_TRADE:
            # Log reasons from the range context for diagnostic visibility
            try:
                _rctx = self.range_scalper._build_range_context(df_h1, df_h4)
                _reason = ", ".join(_rctx.notes) if _rctx.notes else "unknown"
                logger.info(f"  [{symbol}] RangingScalper: no valid LTF entry — {_reason}")
            except Exception:
                logger.info(f"  [{symbol}] RangingScalper: no valid LTF entry found")
            return

        logger.info(
            f"  [{symbol}] Scalp entry: {entry.signal.value} | tf={entry.entry_tf} | "
            f"conf={entry.confidence:.2f} | zone={entry.zone.value} | "
            f"entry={entry.entry_price:.5f} SL={entry.sl_price:.5f} "
            f"TP1={entry.tp1_price:.5f} TP2={entry.tp2_price:.5f} | "
            f"swept={entry.liquidity_swept} ob={entry.ob_mitigated} "
            f"fvg={entry.fvg_present} disp={entry.displacement}"
        )

        # ── 3. Risk / prop guards with scalp sizing ──────────────────────────
        open_cnt  = self.broker.count_open_positions()
        guard_ok, guard_reason = self.prop_guard.check(
            equity=equity,
            open_positions=open_cnt,
            signal_prob=entry.confidence,
            symbol=symbol,
        )
        if not guard_ok:
            logger.info(f"  [{symbol}] PROP GUARD blocked scalp: {guard_reason}")
            return

        # FIX: wire risk_engine gate for scalp path same as trend path
        risk_ok, risk_reason = self.risk_engine.approve_trade(
            equity=equity,
            open_positions=open_cnt,
            symbol=symbol,
            signal_prob=entry.confidence,
        )
        if not risk_ok:
            logger.info(f"  [{symbol}] RISK ENGINE blocked scalp: {risk_reason}")
            return

        sym_info_dict  = self.broker.get_symbol_info(symbol) or {}
        contract_size  = float(sym_info_dict.get("contract_size", 100_000))
        symbol_point   = float(sym_info_dict.get("point", 0.00001))
        atr_val        = entry.ltf_atr   # use LTF ATR for scalp sizing

        exec_mult  = self.exec_tracker.get_size_adjustment(symbol) if self.exec_tracker else 1.0
        perf_mult  = self.perf_monitor.get_size_multiplier(symbol) if self.perf_monitor else 1.0

        lot = self.risk_engine.calculate_lot_size(
            equity=equity,
            atr=atr_val,
            min_lot=float(sym_info_dict.get("min_lot", 0.01)),
            max_lot=float(sym_info_dict.get("max_lot", 0.5)),   # cap scalp size
            symbol_point=symbol_point,
            contract_size=contract_size,
        )
        lot = round(lot * exec_mult * perf_mult * 0.75 * getattr(self, "_auto_lot_scale", 1.0), 2)  # extra 25% size reduction for scalps
        if lot <= 0:
            logger.warning(f"  [{symbol}] Scalp lot reduced to zero — skip")
            return

        # v8: correlated-group cap for scalp path
        corr_ok, corr_reason = self.risk_engine.approve_correlated_trade(
            symbol=symbol, proposed_lot=lot, equity=equity,
            atr=atr_val, symbol_point=symbol_point, contract_size=contract_size,
        )
        if not corr_ok:
            logger.info(f"  [{symbol}] CORR RISK blocked scalp: {corr_reason}")
            return

        # ── 4. Place scalp order ─────────────────────────────────────────────
        order_type = "buy" if entry.signal == ScalpSignal.BUY else "sell"
        sl_dist    = abs(entry.entry_price - entry.sl_price)
        tp_dist    = abs(entry.tp1_price   - entry.entry_price)  # use TP1 as primary
        sl_atr_m   = sl_dist / atr_val if atr_val > 0 else 0.6
        tp_atr_m   = tp_dist / atr_val if atr_val > 0 else 1.2

        import re as _re
        _raw_scalp_comment = (
            f"SCALP_{symbol[:6]}_{entry.entry_tf}_c{entry.confidence:.2f}"
            f"_zone{entry.zone.value[:3].upper()}"
        )
        comment = _re.sub(r"[^a-zA-Z0-9_ \-]", "", _raw_scalp_comment)[:27]

        result = self.broker.place_order(
            symbol=symbol,
            order_type=order_type,
            volume=lot,
            atr=atr_val,
            sl_atr_mult=sl_atr_m,
            tp_atr_mult=tp_atr_m,
            signal_prob=entry.confidence,
            comment=comment,
        )

        if result:
            ticket = result.get("ticket", 0)
            logger.info(
                f"  [SCALP OK] {symbol} {order_type.upper()} lot={lot} "
                f"ticket={ticket} | TP1={entry.tp1_price:.5f} TP2={entry.tp2_price:.5f}"
            )
            # FIX: record open trade so risk engine daily/concurrent limits are live
            self.risk_engine.record_trade_open()
            self.risk_engine.record_open_lot(symbol, lot)  # v8: corr-group tracking
            self.visualizer.add_trade(result)
            # Register in scalp tracker for exit management
            self._scalp_positions[ticket] = {
                "symbol": symbol,
                "bars_held": 0,
                "entry": entry,
                "range_high": getattr(self, "_latest_regime_snap", None) and
                              getattr(self._latest_regime_snap, "range_high", 0.0),
                "range_low":  getattr(self, "_latest_regime_snap", None) and
                              getattr(self._latest_regime_snap, "range_low", 0.0),
            }
        else:
            logger.warning(
                f"  [{symbol}] Scalp order NOT placed — broker.place_order() returned None. "
                f"Check [MT5] ORDER FAILED log above for retcode and comment."
            )

    def _manage_scalp_positions(
        self, symbol: str, df_h1: pd.DataFrame,
        df_h4: Optional[pd.DataFrame]
    ) -> None:
        """
        Quick-exit manager for open scalp positions using ScalpExitManager.
        Also handles partial close at TP1 (mid-range).
        """
        if not self.scalp_exit_mgr:
            return

        positions = self.broker.get_open_positions(symbol)
        if not positions:
            # Clean up stale tracker entries
            stale = [t for t, v in self._scalp_positions.items() if v["symbol"] == symbol]
            for t in stale:
                del self._scalp_positions[t]
            return

        # Fetch M15 for exit analysis
        df_m15 = None
        try:
            df_m15 = self.broker.get_market_data(symbol, "m15", 100)
        except Exception:
            pass

        if df_m15 is None or len(df_m15) < 10:
            return

        latest_price_info = self.broker.get_latest_price(symbol)
        if not latest_price_info:
            return

        # Get range context from latest snap
        snap = getattr(self, "_latest_regime_snap", None)

        for pos in positions:
            ticket = pos["ticket"]
            if ticket not in self._scalp_positions:
                continue  # not a scalp we placed

            self._scalp_positions[ticket]["bars_held"] += 1
            bars_held   = self._scalp_positions[ticket]["bars_held"]
            entry_data  = self._scalp_positions[ticket]["entry"]
            pos_type    = pos["type"]   # "buy" | "sell"
            open_price  = float(pos["open_price"])
            current_p   = float(
                latest_price_info.get("ask", 0) if pos_type == "buy"
                else latest_price_info.get("bid", 0)
            )

            # Build minimal RangeContext for exit manager
            # FIX: RangeContext imported at module level — no hot-path import needed
            rctx = RangeContext()
            if snap:
                rctx.range_high = getattr(snap, "range_high", 0.0)
                rctx.range_low  = getattr(snap, "range_low", 0.0)
                rctx.range_mid  = (rctx.range_high + rctx.range_low) / 2
                rctx.range_atr  = entry_data.ltf_atr
                rctx.is_valid   = rctx.range_high > rctx.range_low

            # Partial close at TP1 if not done yet
            tp1 = entry_data.tp1_price
            direction = 1 if pos_type == "buy" else -1
            profit_pts = (current_p - open_price) * direction
            if tp1 > 0 and profit_pts > 0:
                hit_tp1 = (pos_type == "buy" and current_p >= tp1) or \
                          (pos_type == "sell" and current_p <= tp1)
                if hit_tp1 and pos.get("volume", 0) > 0:
                    # Close 50% at TP1, let runner go to TP2
                    half_vol = round(float(pos.get("volume", 0)) / 2, 2)
                    if half_vol >= 0.01:
                        logger.info(
                            f"  [SCALP PARTIAL] {symbol} ticket={ticket} "
                            f"closing {half_vol} lot at TP1={tp1:.5f}"
                        )
                        self.broker.close_order(ticket, symbol=symbol, volume=half_vol)
                        continue  # don't also do early exit check same bar

            # Quick-exit logic
            if rctx.is_valid:
                should_exit, reason = self.scalp_exit_mgr.should_exit_early(
                    pos_type=pos_type,
                    entry_price=open_price,
                    current_price=current_p,
                    range_ctx=rctx,
                    df_ltf=df_m15,
                    bars_held=bars_held,
                )
                if should_exit:
                    logger.info(
                        f"  [SCALP EXIT] {symbol} ticket={ticket} | reason={reason} | "
                        f"profit_pts={profit_pts:.5f}"
                    )
                    self.broker.close_order(ticket, symbol=symbol)
                    del self._scalp_positions[ticket]

    # ── Position management ────────────────────────────────────────────────────

    def _reconcile_closed_positions(self, equity: float) -> None:
        """Detect positions closed externally (SL/TP hit) and feed P&L into risk engine."""
        # Build current ticket set across all symbols
        current_tickets: dict = {}
        try:
            for symbol in self.symbols:
                for pos in (self.broker.get_open_positions(symbol) or []):
                    t = pos["ticket"]
                    current_tickets[t] = {
                        "symbol": symbol,
                        "profit": float(pos.get("profit", 0.0)),
                    }
        except Exception as e:
            logger.debug(f"_reconcile_closed_positions fetch error: {e}")
            return

        # FIX: Update _known_tickets profit with the LATEST unrealised P&L from the
        # broker every cycle. This ensures that if the history lookup fails, the
        # last-seen value is at least the most recent floating P&L, not the open-time 0.0.
        for t, data in current_tickets.items():
            if t in self._known_tickets:
                self._known_tickets[t]["profit"] = data["profit"]

        # Any ticket we were tracking that is now gone has closed externally
        closed_tickets = set(self._known_tickets) - set(current_tickets)

        # FIX: Build a lookup of recently closed deals from MT5 history so we get
        # the REAL realised PnL instead of the last-seen unrealised float (which is
        # often 0.00 or stale). We look back 2 days to catch any position closed
        # since the last cycle. This call is lightweight (2-day window).
        _recent_history: dict = {}   # ticket → profit
        try:
            _hist = self.broker.get_trade_history(days=2)
            for _h in (_hist or []):
                _t = int(_h.get("ticket", 0))
                if _t:
                    _recent_history[_t] = float(_h.get("profit", 0.0))
        except Exception as _he:
            logger.debug(f"_reconcile: history fetch error: {_he}")

        for ticket in closed_tickets:
            info = self._known_tickets[ticket]
            # Prefer real closed PnL from broker history; fall back to last-seen value
            if ticket in _recent_history:
                pnl = _recent_history[ticket]
            else:
                pnl = info.get("profit", 0.0)
            symbol = info.get("symbol", "UNKNOWN")
            logger.info(
                f"  [{symbol}] Externally-closed position ticket={ticket} "
                f"pnl=${pnl:.2f} — recording to risk engine."
            )
            self.risk_engine.record_trade_close(pnl=pnl, equity=equity)
            closed_lot = info.get("volume", 0.01)
            try:
                self.risk_engine.record_close_lot(symbol, closed_lot)
            except Exception:
                pass
            if self.perf_monitor:
                try:
                    self.perf_monitor.record_trade(symbol=symbol, pnl=pnl, equity=equity)
                except Exception:
                    pass
            sym_eng = self._signal_engines.get(symbol)
            if sym_eng:
                direction = info.get("type", "buy")
                predicted = "BUY" if direction == "buy" else "SELL"
                try:
                    # record_live_trade_close combines outcome tracking + buffer write
                    sym_eng.record_live_trade_close(
                        ticket=ticket,
                        pnl=pnl,
                        predicted_signal=predicted,
                    )
                except Exception:
                    pass

            # ── v19 DIR-9 + DIR-10: feed PnL to Kelly and symbol scorer ──────
            if AUTO_EA_AVAILABLE:
                if self._kelly_sizer is not None:
                    try:
                        self._kelly_sizer.record_trade(symbol, pnl)
                    except Exception:
                        pass
                if self._symbol_scorer is not None:
                    try:
                        self._symbol_scorer.record(symbol, pnl)
                    except Exception:
                        pass

            # v20: Feed closed trade into TradeHistoryLearner for ongoing learning
            if HIST_LEARNER_AVAILABLE and self._hist_learner is not None:
                try:
                    import datetime as _datetime_mod
                    self._hist_learner.record_close({
                        "ticket":     ticket,
                        "symbol":     symbol,
                        "type":       info.get("type", "buy"),
                        "volume":     info.get("volume", 0.01),
                        "open_price": info.get("open_price", 0.0),
                        "profit":     pnl,
                        "open_time":  info.get("open_time", ""),
                        "close_time": _datetime_mod.datetime.utcnow().isoformat(),
                        "strategy":   "AI_EA",
                    })
                except Exception as _hl_e:
                    logger.debug(f"HistLearner record_close error: {_hl_e}")

        # Refresh known tickets — store latest broker profit so last-known value is accurate
        self._known_tickets = {}
        try:
            for symbol in self.symbols:
                for pos in (self.broker.get_open_positions(symbol) or []):
                    t = pos["ticket"]
                    self._known_tickets[t] = {
                        "symbol":     symbol,
                        "profit":     float(pos.get("profit", 0.0)),
                        "type":       pos.get("type", "buy"),
                        "volume":     float(pos.get("volume", 0.01)),
                        "open_price": float(pos.get("open_price", 0.0)),   # v20
                        "open_time":  str(pos.get("open_time", "")),        # v20
                    }
        except Exception as e:
            logger.debug(f"_reconcile_closed_positions refresh error: {e}")

    def _manage_existing_positions(
        self, symbol: str, df: pd.DataFrame, structure: Dict
    ) -> None:
        positions = self.broker.get_open_positions(symbol)
        if not positions:
            return

        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
        latest = self.broker.get_latest_price(symbol)
        if latest is None:
            return

        for pos in positions:
            pos_type   = pos["type"]           # "buy" | "sell"
            open_price = float(pos["open_price"])
            current_p  = float(latest["ask"] if pos_type == "buy" else latest["bid"])
            direction  = 1 if pos_type == "buy" else -1
            profit_pts = (current_p - open_price) * direction
            ticket     = pos["ticket"]
            current_sl = float(pos.get("sl") or 0)

            if atr <= 0:
                continue

            # ── Breakeven: move SL to entry once price moves 1R in our favour ──
            be_trigger = BREAKEVEN_ATR_MULT * atr
            if profit_pts >= be_trigger:
                be_sl = open_price + (0.1 * atr * direction)   # tiny buffer past entry
                needs_be = (
                    (pos_type == "buy"  and (current_sl == 0 or current_sl < be_sl)) or
                    (pos_type == "sell" and (current_sl == 0 or current_sl > be_sl))
                )
                if needs_be:
                    self.broker.modify_order(ticket, sl=round(be_sl, 5))
                    logger.info(
                        f"  [BE] {symbol} ticket={ticket} SL moved to breakeven "
                        f"{be_sl:.5f} (profit={profit_pts:.5f} >= {be_trigger:.5f})"
                    )
                    current_sl = be_sl

            # ── Trailing stop ────────────────────────────────────────────────
            # Kicks in once price has moved >= BREAKEVEN_ATR_MULT×ATR in our favour.
            # Trail distance tightens in three stages as profit grows:
            #   1×ATR profit  → trail at TRAILING_ATR_MULT×ATR (configurable, default 1.5)
            #   2×ATR profit  → trail at 0.75×ATR
            #   3×ATR profit  → trail at 0.5×ATR   (tight — protect big wins)
            if profit_pts >= BREAKEVEN_ATR_MULT * atr:
                if profit_pts >= 3.0 * atr:
                    trail_dist = 0.5 * atr
                elif profit_pts >= 2.0 * atr:
                    trail_dist = 0.75 * atr
                else:
                    trail_dist = TRAILING_ATR_MULT * atr   # use configurable constant

                # New SL trails current price
                new_sl = current_p - (trail_dist * direction)

                # Only move SL in profit direction, never backwards
                improve = (
                    (pos_type == "buy"  and new_sl > current_sl) or
                    (pos_type == "sell" and (current_sl == 0 or new_sl < current_sl))
                )
                if improve:
                    self.broker.modify_order(ticket, sl=round(new_sl, 5))
                    logger.info(
                        f"  [TRAIL] {symbol} ticket={ticket} SL -> {new_sl:.5f} "
                        f"(trail_dist={trail_dist:.5f} profit_pts={profit_pts:.5f})"
                    )

            # Close on adverse CHoCH while in profit
            choch     = structure.get("choch", False)
            choch_dir = structure.get("choch_direction")
            if choch:
                adverse = (
                    (pos_type == "buy"  and choch_dir == "bearish") or
                    (pos_type == "sell" and choch_dir == "bullish")
                )
                if adverse and profit_pts > 0:
                    logger.info(
                        f"  Closing {symbol} ticket={ticket} before adverse CHoCH"
                    )
                    self.broker.close_order(ticket, symbol=symbol)

            # v6 PRO FIX 4+5: feed closed-trade P&L into PerformanceMonitor and
            # SignalEngine so degradation detection and adaptive retraining work.
            # We record if the position is no longer open after management actions.
            try:
                still_open = any(
                    p["ticket"] == ticket
                    for p in (self.broker.get_open_positions(symbol) or [])
                )
                if not still_open:
                    # Convert price-unit P&L to pips for consistent metrics
                    sym_info_close = self.broker.get_symbol_info(symbol) or {}
                    pt = float(sym_info_close.get("point", 0.00001))
                    pip_size_c = pt * 10
                    pnl_pips = (profit_pts / pip_size_c) if pip_size_c > 0 else profit_pts
                    pnl_dollars = float(pos.get("profit", profit_pts))

                    if self.perf_monitor:
                        self.perf_monitor.record_trade(
                            symbol=symbol,
                            pnl=pnl_dollars,
                            equity=self.risk_engine.get_status(0).get("equity_baseline", 0.0),
                        )

                    # FIX: wire record_trade_close so risk_engine daily P&L,
                    # consecutive-loss streak and cooldown timer are all updated.
                    self.risk_engine.record_trade_close(pnl=pnl_dollars, equity=equity)
                    # v8: remove from correlated-group lot tracker
                    closed_lot = float(pos.get("volume", 0.01))
                    self.risk_engine.record_close_lot(symbol, closed_lot)

                    sym_eng = self._signal_engines.get(symbol)
                    if sym_eng:
                        predicted = "BUY" if pos_type == "buy" else "SELL"
                        sym_eng.record_live_trade_close(
                            ticket=str(ticket),
                            pnl=pnl_dollars,
                            predicted_signal=predicted,
                        )

                    logger.info(
                        f"  [{symbol}] Closed position ticket={ticket} "
                        f"pnl=${pnl_dollars:.2f} ({pnl_pips:.1f} pips) recorded."
                    )
            except Exception as _rec_err:
                logger.debug(f"  P&L recording failed for {symbol}/{ticket}: {_rec_err}")

    # ── Composite score ────────────────────────────────────────────────────────
    def _score_signal_v7(self, signal: str, prob: float, structure: dict, mtf_result=None) -> float:
        """
        v17 composite scorer (100 pts max → /100 → [0,1]):
          ML prob    40 pts  — calibrated probability (primary edge)
          MTF bias   25 pts  — multi-timeframe confluence
          Trend      15 pts  — structure trend alignment
          Structure  10 pts  — BOS/CHoCH/liquidity quality
          Session     7 pts  — London/NY kill zones
          Alignment   3 pts  — structure direction bonus

        v17 NOTE: ML gate is now 0.38–0.48 (down from 0.65–0.90) matching
        T=1.5 temperature-scaled 3-class model output distribution.
        Floor of 0.40 ensures passing-ML signals always clear the 0.36 composite gate.
        """
        try:
            # ML: calibrated prob [0,1] * 40
            ml_pts = float(max(0.0, prob)) * 40.0

            # MTF: default 10 pts (neutral) up to 25
            mtf_pts = 10.0
            if mtf_result:
                bias_match = (
                    (signal == "BUY"  and mtf_result.bias == "bullish") or
                    (signal == "SELL" and mtf_result.bias == "bearish")
                )
                bias_mul = 1.3 if bias_match else 0.7
                mtf_pts  = min(mtf_result.score * 25 * bias_mul, 25)
                if getattr(mtf_result, "macro_aligned",  False): mtf_pts = min(mtf_pts + 2, 25)
                if getattr(mtf_result, "htf_aligned",    False): mtf_pts = min(mtf_pts + 4, 25)
                if getattr(mtf_result, "mtf_aligned",    False): mtf_pts = min(mtf_pts + 2, 25)
                if getattr(mtf_result, "ltf_confirmed",  False): mtf_pts = min(mtf_pts + 3, 25)
                tier_bonus = getattr(mtf_result, "tier_score", 0) * 0.3
                mtf_pts = min(mtf_pts + tier_bonus, 25)

            # Trend
            trend = structure.get("trend", "ranging")
            if (signal == "BUY" and trend == "bullish") or (signal == "SELL" and trend == "bearish"):
                trend_pts = 15
            elif trend == "ranging":
                trend_pts = 7   # valid for mean-reversion scalps
            else:
                trend_pts = 3   # counter-trend: penalty not zero

            # Structure quality
            struct_score = structure.get("structure_score", 0.0)
            struct_pts   = min(struct_score * 10, 8)
            if structure.get("bos"):             struct_pts = min(struct_pts + 2, 10)
            if structure.get("choch"):           struct_pts = min(struct_pts + 2, 10)
            if structure.get("liquidity_sweep"): struct_pts = min(struct_pts + 1, 10)

            # Session — Bug 3 FIX: 03:00–07:00 UTC was off-peak (3pts).
            # Asian pre-London / metals session should get 4 pts, not 3.
            # New tiers match signal_scorer._score_session().
            import datetime as _dt
            now_h = _dt.datetime.now(_dt.timezone.utc).hour
            if now_h in range(13, 17):            # London/NY overlap → peak
                sess_pts = 7
            elif now_h in range(7, 13):           # London pre-overlap
                sess_pts = 5
            elif now_h in range(17, 22):          # NY only
                sess_pts = 4
            elif now_h in range(0, 7):            # Asian + pre-London (metals/BTC liquid)
                sess_pts = 4
            else:                                  # 22–24 UTC: true dead zone
                sess_pts = 1

            # Alignment bonus
            try:
                aligned   = self.structure_engine.is_trade_aligned_with_structure(signal, structure)
                align_pts = 3 if aligned else 0
            except Exception:
                align_pts = 0

            total = ml_pts + mtf_pts + trend_pts + struct_pts + sess_pts + align_pts
            score = round(min(total / 100.0, 1.0), 4)

            # NOTE: Score floor removed. The floor was bumping every ML-passing signal
            # to 0.40, guaranteeing it cleared the composite gate (0.36-0.39) regardless
            # of how weak the MTF/structure/trend context was. That produced the neutral-
            # MTF, ranging, counter-trend trades seen in the log. The ML prob gate, the
            # MTF neutral-bias skip, and the counter-trend penalty now carry that weight
            # correctly without a blanket floor.

            return score
        except Exception as e:
            logger.warning(f"_score_signal_v7 error: {e}")
            return self._score_signal(prob, structure)

    def _score_signal(self, ml_prob: float, structure: Dict) -> float:
        struct_score = structure.get("structure_score", 0.0)
        bonus = 0.0
        if structure.get("bos"):   bonus += 0.10
        if structure.get("choch"): bonus += 0.10
        return min(ml_prob * 0.5 + struct_score * 0.3 + bonus, 1.0)

    # ── Walk-forward SL/TP optimisation ───────────────────────────────────────
    def _run_wf_optimization(self) -> None:
        """
        Walk-forward SL/TP re-optimisation.

        v11 FIX: Previous version used only 744 bars (31d) which is far too few
        for a breakout strategy to accumulate statistical edge.  Raised to 2160
        bars (~90d at H1) which matches the ML training window.

        v11 FIX: direction="both" was not handled by Backtester._generate_signals()
        or test_strategy() — the direction multiplier was always 1 (buy).  We now
        run BUY and SELL strategies separately and pick the best profitable combo
        across both directions.

        v11 FIX: wr logging bug — win_rate from _metrics() is already on a 0-100
        scale, so f"{wr:.2%}" displayed e.g. "5455.00%".  Fixed to f"{wr:.1f}%".

        v11 FIX: Expanded SL grid — previous lower bound 0.8 ATR is too tight for
        volatile instruments (BTC, XAG).  Added 3.0 and 4.0 ATR to the grid so
        the optimizer can find profitable configs on high-ATR instruments.
        """
        if self._wf_optimizer is None:
            return
        logger.info("[WF-OPT] Starting walk-forward SL/TP optimisation...")
        for symbol in self.symbols:
            try:
                # Use 2160 bars (~90d H1) — same window as ML training for consistency
                n_bars = max(int(os.environ.get("BARS", 2160)), 2160)
                df = self.broker.get_market_data(symbol, TIMEFRAME_STR, n_bars)
                if df is None or len(df) < 200:
                    continue
                df = self.evaluator.add_market_indicators(df)

                # Expanded SL/TP grid — wider range covers volatile instruments
                sl_values = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]
                tp_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

                best_result = None
                best_profit = float("-inf")

                # Run BUY and SELL directions separately — backtester handles
                # "buy" and "sell" correctly; "both" was silently treated as buy.
                for direction, high_rule, low_rule in [
                    ("buy",  "crosses_above", "high_20"),
                    ("sell", "crosses_below", "low_20"),
                ]:
                    stub_strategy = {
                        "name": f"wf_{symbol}_{direction}",
                        "direction": direction,
                        "rules": [
                            {"indicator": "close", "condition": high_rule,
                             "reference": low_rule},
                        ],
                    }
                    result = self._wf_optimizer.optimize_parameters(
                        strategy=stub_strategy,
                        market_data=df,
                        sl_values=sl_values,
                        tp_values=tp_values,
                    )
                    p = result.get("profit", float("-inf")) if result else float("-inf")
                    if p > best_profit:
                        best_profit = p
                        best_result = result

                if best_result and best_profit > 0:
                    self._wf_sl_mult[symbol] = float(best_result["sl"])
                    self._wf_tp_mult[symbol] = float(best_result["tp"])
                    wr = best_result.get("win_rate", 0)
                    # win_rate is already 0-100 scale from _metrics()
                    logger.info(
                        f"[WF-OPT] {symbol}: SL={best_result['sl']} TP={best_result['tp']} "
                        f"wr={wr:.1f}% profit={best_profit:.2f}"
                    )
                else:
                    # Genuinely no profitable combo on any direction — keep current multipliers
                    # This is diagnostic, not a bug: it means the breakout strategy has no
                    # edge on this instrument in this window.  ML signal thresholds still apply.
                    logger.info(
                        f"[WF-OPT] {symbol}: no profitable combo found "
                        f"(best={best_profit:.2f}) — "
                        f"keeping SL={self._wf_sl_mult.get(symbol, 'global')} "
                        f"TP={self._wf_tp_mult.get(symbol, 'global')}"
                    )
            except Exception as e:
                logger.warning(f"[WF-OPT] {symbol} failed: {e}")
        logger.info("[WF-OPT] Walk-forward optimisation complete.")

    # ── ML retraining ──────────────────────────────────────────────────────────
    def _incremental_retrain_symbol(self, symbol: str) -> None:
        """
        v19 DIR-8: Incremental single-symbol retrain triggered by live trade buffer.
        Hot-swaps the model in memory if the new one beats the saved one.
        """
        logger.info(f"[AUTO] DIR-8: incremental retrain for {symbol}")
        try:
            sym_engine = self._get_signal_engine(symbol)
            saved_acc  = sym_engine._wf_mean_acc
            gate_ok    = sym_engine._wf_gate_passed

            df_full = self.broker.get_market_data(symbol, TIMEFRAME_STR, BARS)
            if df_full is None or len(df_full) < 300:
                logger.warning(f"[AUTO] DIR-8: {symbol}: insufficient data for incremental retrain")
                return
            df_full = self.evaluator.add_market_indicators(df_full)

            df_h4 = None
            try:
                df_h4 = self.broker.get_market_data(symbol, "h4", max(600, BARS // 4))
            except Exception:
                pass

            from signal_engine import SignalEngine as _SE
            candidate = _SE(symbol=symbol)
            if sym_engine._live_buffer is not None:
                candidate._live_buffer = sym_engine._live_buffer

            ok = candidate.train(df_full, df_h4=df_h4, forward_bars=5, save_if_best=False)
            if not ok:
                logger.warning(f"[AUTO] DIR-8: {symbol}: incremental retrain failed")
                return

            new_acc   = candidate._wf_mean_acc
            new_gated = candidate._wf_gate_passed
            if (new_gated and new_acc > saved_acc) or (not gate_ok and new_acc > saved_acc):
                candidate._save_model()
                sym_engine.rf_model        = candidate.rf_model
                sym_engine.gbm_model       = candidate.gbm_model
                sym_engine.xgb_model       = candidate.xgb_model
                sym_engine.lgb_model       = candidate.lgb_model
                sym_engine.meta_model      = candidate.meta_model
                sym_engine.scaler          = candidate.scaler
                sym_engine.feature_names   = candidate.feature_names
                sym_engine._n_features     = candidate._n_features
                sym_engine._wf_mean_acc    = new_acc
                sym_engine._wf_gate_passed = new_gated
                sym_engine.is_trained      = True
                logger.info(
                    f"[AUTO] DIR-8: {symbol}: hot-swapped model "
                    f"acc={new_acc:.3f} > saved={saved_acc:.3f}"
                )
                if AUTO_EA_AVAILABLE and self._symbol_scorer is not None:
                    self._symbol_scorer.update_wf_acc(symbol, new_acc)
            else:
                logger.info(
                    f"[AUTO] DIR-8: {symbol}: incremental candidate acc={new_acc:.3f} "
                    f"did not beat saved={saved_acc:.3f} — keeping saved"
                )
        except Exception as exc:
            logger.error(f"[AUTO] DIR-8: {symbol} incremental retrain error: {exc}", exc_info=True)

    def _retrain_signal_engine(self) -> None:
        """
        v17 SAFE RETRAIN — Protect models produced by trainer.py.

        Core rule: the live EA must NEVER overwrite a saved model unless the new
        candidate is strictly better (higher wf_mean_acc AND gate passed).

        Problems fixed vs v13:
          BUG 1 — train() defaulted save_if_best=True, so every window immediately
                   overwrote the pkl on disk.  The 7d window (acc≈0.48) silently
                   replaced the trainer.py 365d model (acc≈0.58+).
          BUG 2 — No comparison against the existing saved accuracy.  Even when
                   best_acc tracking worked, the model that was ultimately in memory
                   was the LAST window trained, not the best.
          BUG 3 — No gate check: models below WF_GATE_MEAN=0.44 or with high fold
                   spread were saved whenever ok=True was returned.
          BUG 4 — Shared fallback trained unconditionally every cycle and overwrote
                   per-symbol pkls that loaded from the shared path as fallback.

        Algorithm:
          1. Load current saved wf_mean_acc as the "baseline to beat".
          2. Train all 7 windows with save_if_best=False (nothing writes to disk).
          3. Track the best candidate: must pass gate AND beat baseline.
          4. Only if a better candidate exists: call _save_model() once.
          5. Skip retrain entirely if the saved model already passes the gate and
             has not degraded (needs_retraining() returns False).
        """
        logger.info("[TRAINER v17] Per-symbol safe retrain (protect trainer.py models)...")
        trained_any = False

        # 7-tier window definitions — bars/forward_bars mirror trainer.py
        windows = {
            "365d": {"bars": BARS,             "forward_bars": 10},
            "90d":  {"bars": min(2160, BARS),  "forward_bars": 8},
            "31d":  {"bars": min(744,  BARS),  "forward_bars": 5},
            "14d":  {"bars": min(336,  BARS),  "forward_bars": 4},
            "7d":   {"bars": min(168,  BARS),  "forward_bars": 3},
            "3d":   {"bars": min(72,   BARS),  "forward_bars": 2},
            "1d":   {"bars": min(24,   BARS),  "forward_bars": 1},
        }

        WF_GATE_MEAN   = 0.44   # must match signal_engine.py constant
        WF_GATE_SPREAD = 0.35   # fold spread ceiling for consistency

        for symbol in self.symbols:
            try:
                sym_engine = self._get_signal_engine(symbol)

                # ── Step 1: read the baseline accuracy of the saved model ──────
                saved_acc = sym_engine._wf_mean_acc  # loaded from pkl by __init__
                gate_ok   = sym_engine._wf_gate_passed

                # Skip retrain if model is healthy and live accuracy has not degraded
                if gate_ok and saved_acc >= WF_GATE_MEAN and not sym_engine.needs_retraining():
                    logger.info(
                        f"[TRAINER v17] {symbol}: saved model healthy "
                        f"(wf_acc={saved_acc:.3f}, gate=passed) — skipping retrain"
                    )
                    continue

                logger.info(
                    f"[TRAINER v17] {symbol}: retraining "
                    f"(saved_acc={saved_acc:.3f}, gate={'pass' if gate_ok else 'FAIL'}, "
                    f"degraded={sym_engine.needs_retraining()})"
                )

                # ── Step 2: fetch data ────────────────────────────────────────
                df_full = self.broker.get_market_data(symbol, TIMEFRAME_STR, BARS)
                if df_full is None or len(df_full) < 300:
                    logger.warning(
                        f"[TRAINER v17] {symbol}: insufficient data "
                        f"({len(df_full) if df_full is not None else 0} bars) — skipping"
                    )
                    continue
                df_full = self.evaluator.add_market_indicators(df_full)

                df_h4_train = None
                try:
                    df_h4_train = self.broker.get_market_data(
                        symbol, "h4", max(600, BARS // 4)
                    )
                except Exception:
                    pass

                # ── Step 3: train all windows with save_if_best=False ─────────
                # Nothing writes to disk during this loop.
                best_acc     = -1.0
                best_engine  = None   # the in-memory engine with the best result
                fallback_acc = -1.0
                fallback_eng = None   # best result even if gate not cleared

                for wkey, wcfg in windows.items():
                    n_bars = wcfg["bars"]
                    fwd    = wcfg["forward_bars"]
                    df_sl  = (df_full.iloc[-n_bars:].copy()
                              if len(df_full) >= n_bars else df_full.copy())

                    # Use a fresh engine per window so models don't bleed
                    from signal_engine import SignalEngine as _SE
                    candidate = _SE(symbol=symbol)
                    # ── Share live buffer so blend_live_samples() picks it up ──
                    # The candidate gets the SAME buffer object as the live engine
                    # so real trades are injected into every candidate window fit.
                    if sym_engine._live_buffer is not None:
                        candidate._live_buffer = sym_engine._live_buffer

                    ok = candidate.train(
                        df_sl,
                        df_h4=df_h4_train,
                        forward_bars=fwd,
                        save_if_best=False,   # ← KEY: never write to disk here
                    )
                    if not ok:
                        logger.warning(
                            f"[TRAINER v17] {symbol}/{wkey}: "
                            f"training failed (low labels/variance)"
                        )
                        continue

                    acc    = candidate._wf_mean_acc
                    gated  = candidate._wf_gate_passed
                    logger.info(
                        f"[TRAINER v17] {symbol}/{wkey}: "
                        f"wf_acc={acc:.3f} gate={'pass' if gated else 'FAIL'} "
                        f"bars={len(df_sl)}"
                    )

                    # Gated best: gate passed AND strictly better than saved AND best so far
                    if gated and acc > saved_acc and acc > best_acc:
                        best_acc    = acc
                        best_engine = candidate

                    # Fallback: best ungated (only used if nothing gated beats saved)
                    if acc > fallback_acc:
                        fallback_acc = acc
                        fallback_eng = candidate

                # ── Step 4: save only if we have something strictly better ─────
                if best_engine is not None:
                    # Gate passed AND beats saved model — safe to overwrite
                    best_engine._save_model()
                    # Sync the live engine to the newly saved state
                    sym_engine.rf_model       = best_engine.rf_model
                    sym_engine.gbm_model      = best_engine.gbm_model
                    sym_engine.xgb_model      = best_engine.xgb_model
                    sym_engine.lgb_model      = best_engine.lgb_model
                    sym_engine.meta_model     = best_engine.meta_model
                    sym_engine.scaler         = best_engine.scaler
                    sym_engine.feature_names  = best_engine.feature_names
                    sym_engine._n_features    = best_engine._n_features
                    sym_engine._wf_mean_acc   = best_engine._wf_mean_acc
                    sym_engine._wf_gate_passed = best_engine._wf_gate_passed
                    sym_engine.is_trained     = True
                    trained_any = True
                    logger.info(
                        f"[TRAINER v17] {symbol}: IMPROVED — "
                        f"new wf_acc={best_acc:.3f} > saved={saved_acc:.3f} | model saved"
                    )
                    # v19 DIR-10 + DIR-12: update symbol score, clear worse-model delay
                    if AUTO_EA_AVAILABLE:
                        if self._symbol_scorer is not None:
                            self._symbol_scorer.update_wf_acc(symbol, best_acc)
                        if self._retrain_sched is not None:
                            self._retrain_sched.notify_improved()
                elif fallback_eng is not None and fallback_acc > saved_acc:
                    # No gated winner but fallback beats saved (saved was also ungated)
                    # Only upgrade if saved model itself was below gate
                    if not gate_ok:
                        fallback_eng._save_model()
                        sym_engine.rf_model       = fallback_eng.rf_model
                        sym_engine.gbm_model      = fallback_eng.gbm_model
                        sym_engine.xgb_model      = fallback_eng.xgb_model
                        sym_engine.lgb_model      = fallback_eng.lgb_model
                        sym_engine.meta_model     = fallback_eng.meta_model
                        sym_engine.scaler         = fallback_eng.scaler
                        sym_engine.feature_names  = fallback_eng.feature_names
                        sym_engine._n_features    = fallback_eng._n_features
                        sym_engine._wf_mean_acc   = fallback_eng._wf_mean_acc
                        sym_engine._wf_gate_passed = fallback_eng._wf_gate_passed
                        sym_engine.is_trained     = True
                        trained_any = True
                        logger.info(
                            f"[TRAINER v17] {symbol}: FALLBACK upgrade — "
                            f"acc={fallback_acc:.3f} > saved={saved_acc:.3f} "
                            f"(both below gate, best available saved)"
                        )
                        if AUTO_EA_AVAILABLE and self._symbol_scorer is not None:
                            self._symbol_scorer.update_wf_acc(symbol, fallback_acc)
                    else:
                        logger.info(
                            f"[TRAINER v17] {symbol}: no improvement found — "
                            f"keeping saved model (acc={saved_acc:.3f}, gate=passed)"
                        )
                        # v19 DIR-12: schedule retry after worse model
                        if AUTO_EA_AVAILABLE and self._retrain_sched is not None:
                            self._retrain_sched.notify_worse_model()
                else:
                    logger.info(
                        f"[TRAINER v17] {symbol}: no improvement found — "
                        f"keeping saved model (acc={saved_acc:.3f})"
                    )
                    # v19 DIR-12: schedule retry after worse model
                    if AUTO_EA_AVAILABLE and self._retrain_sched is not None:
                        self._retrain_sched.notify_worse_model()

            except Exception as e:
                logger.error(f"[TRAINER v17] {symbol} error: {e}", exc_info=True)

        # ── Shared fallback engine ────────────────────────────────────────────
        # Only retrain the shared model if it is degraded; same gate logic applies.
        try:
            shared_saved_acc  = self.signal_engine._wf_mean_acc
            shared_gate_ok    = self.signal_engine._wf_gate_passed
            shared_degraded   = self.signal_engine.needs_retraining()

            if shared_gate_ok and shared_saved_acc >= WF_GATE_MEAN and not shared_degraded:
                logger.info(
                    f"[TRAINER v17] Shared engine healthy "
                    f"(wf_acc={shared_saved_acc:.3f}) — skipping"
                )
            else:
                combined_dfs = []
                for symbol in self.symbols[:5]:
                    try:
                        df = self.broker.get_market_data(symbol, TIMEFRAME_STR, BARS)
                        if df is not None and len(df) > 200:
                            df = self.evaluator.add_market_indicators(df)
                            combined_dfs.append(df)
                    except Exception as e:
                        logger.warning(f"[TRAINER v17] Shared fetch failed for {symbol}: {e}")

                if combined_dfs:
                    combined = pd.concat(combined_dfs, ignore_index=True).dropna()
                    from signal_engine import SignalEngine as _SE
                    shared_candidate = _SE(symbol="default")
                    # Share live buffer from live shared engine
                    if self.signal_engine._live_buffer is not None:
                        shared_candidate._live_buffer = self.signal_engine._live_buffer
                    ok = shared_candidate.train(combined, forward_bars=8, save_if_best=False)
                    if ok:
                        new_acc   = shared_candidate._wf_mean_acc
                        new_gated = shared_candidate._wf_gate_passed
                        if (new_gated and new_acc > shared_saved_acc) or \
                           (not shared_gate_ok and new_acc > shared_saved_acc):
                            shared_candidate._save_model()
                            self.signal_engine.rf_model      = shared_candidate.rf_model
                            self.signal_engine.gbm_model     = shared_candidate.gbm_model
                            self.signal_engine.xgb_model     = shared_candidate.xgb_model
                            self.signal_engine.lgb_model     = shared_candidate.lgb_model
                            self.signal_engine.meta_model    = shared_candidate.meta_model
                            self.signal_engine.scaler        = shared_candidate.scaler
                            self.signal_engine.feature_names = shared_candidate.feature_names
                            self.signal_engine._n_features   = shared_candidate._n_features
                            self.signal_engine._wf_mean_acc  = new_acc
                            self.signal_engine.is_trained    = True
                            trained_any = True
                            logger.info(
                                f"[TRAINER v17] Shared engine improved: "
                                f"acc={new_acc:.3f} > saved={shared_saved_acc:.3f} | saved"
                            )
                        else:
                            logger.info(
                                f"[TRAINER v17] Shared engine: candidate "
                                f"acc={new_acc:.3f} did not beat saved={shared_saved_acc:.3f} "
                                f"— keeping saved"
                            )
        except Exception as e:
            logger.error(f"[TRAINER v17] Shared engine error: {e}", exc_info=True)

        if not trained_any:
            logger.info(
                "[TRAINER v17] No model upgrades — all saved models held. "
                "Run trainer.py to force a full retrain."
            )


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    ea = AITradingEA()
    try:
        ea.start()   # delegates to asyncio.run(_async_main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        ea.stop()


if __name__ == "__main__":
    main()
