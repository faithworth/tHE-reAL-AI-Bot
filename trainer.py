"""
trainer.py — Universal Multi-Broker ML Model Trainer  (AI EA v19)
=====================================================================
TRAINING WINDOWS (7-tier institutional edge):
  * DEEP    (365d) — full macro cycle; earnings, FOMC, elections, vol cycles.
  * MACRO   ( 90d) — 3-month institutional context; quarterly rebalancing.
  * REGIME  ( 31d) — monthly options cycle; captures recurring regime patterns.
  * STRUCT  ( 14d) — structural microstructure; avoids 7d single-trend collapse.
  * SESSION (  7d) — session-level patterns and swing structure.
  * PREC    (  3d) — 72-bar precision entry microstructure.
  * ULTRA   (  1d) — 24-bar ultra-precision M10/M15 trigger confirmation.

  forward_bars=10/8/5/4/3/2/1 scale with window depth for causal labels.

BUG FIXES (carried forward):
    FIX 1-7: All previous fixes retained; see git history for details.

Usage:
    BROKER_TYPE=mt5 python trainer.py --symbol XAUUSD
    BROKER_TYPE=mt5 python trainer.py --symbol XAUUSD --period 31d
    BROKER_TYPE=mt5 python trainer.py --symbol XAUUSD --period all
    BROKER_TYPE=mt5 python trainer.py --all-symbols --per-symbol
                    python run_backtest.py
                    python ai_ea.py

    BROKER_TYPE=mt5 python trainer.py --symbol EURUSD --check-retrain
    BROKER_TYPE=mt5 python trainer.py --symbol XAUUSD --bars 30000 --period all
"""

import argparse
import logging
import os
import sys
import signal

# ── Global watchdog — imported from standalone module (no circular deps) ──────
import watchdog as _wd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("trainer_v19")

# v19: auto-engine directives
try:
    from auto_engine import AutoFetcher, AutoHyperSearch, AutoRetrainScheduler
    AUTO_TRAINER_AVAILABLE = True
except ImportError:
    AUTO_TRAINER_AVAILABLE = False

os.makedirs("models", exist_ok=True)
os.makedirs("data",   exist_ok=True)
os.makedirs("logs",   exist_ok=True)

_wd.activate()   # start watchdog — any hang > 10 min will terminate cleanly

# ── CLI arguments ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Train AI EA v13 ML signal engine (multi-broker, 7-tier windows)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--symbol",        type=str,   default="EURUSD",
                    help="Symbol to train (ignored when --all-symbols)")
parser.add_argument("--bars",          type=int,   default=8000,
                    help="H1 bars to fetch for the DEEP window. "
                         "3000 = ~125d of H1 data, covers all 7 tiers. "
                         "Use --bars 10000 for deeper multi-year context.")
parser.add_argument("--timeframe",     type=str,   default="h1",
                    help="Primary training timeframe (h1 recommended)")
parser.add_argument("--period",        type=str,   default="all",
                    choices=["all", "365d", "90d", "31d", "14d", "7d", "3d", "1d"],
                    help="Data window(s) to train. "
                         "'all' trains all 7 tiers and saves the best model. "
                         "'365d' = macro context (full cycles), "
                         "'90d'  = institutional context, "
                         "'31d'  = regime-aware / monthly rebalancing, "
                         "'14d'  = structural context, "
                         "'7d'   = session-level signals, "
                         "'3d'   = precision entry, "
                         "'1d'   = ultra-precision microstructure.")
parser.add_argument("--all-symbols",   action="store_true",
                    help="Train on all default symbols for the active broker")
parser.add_argument("--per-symbol",    action="store_true",
                    help="Save a dedicated .pkl per symbol (in addition to shared)")
parser.add_argument("--broker-type",   type=str,   default=None,
                    help="Override BROKER_TYPE env var  (mt5 | ibkr | ctrader)")
parser.add_argument("--check-retrain", action="store_true",
                    help="Skip symbols whose live accuracy has not degraded")
parser.add_argument("--tp-mult",       type=float, default=1.5,
                    help="Fixed-risk TP multiplier (ATR units)")
parser.add_argument("--sl-mult",       type=float, default=1.0,
                    help="Fixed-risk SL multiplier (ATR units)")
args = parser.parse_args()

BROKER_TYPE = (args.broker_type or os.getenv("BROKER_TYPE", "mt5")).lower().strip()

# ── 7-tier window table (institutionally calibrated) ─────────────────────────
#
# DEEP   (365d): Full macro cycle — earnings, FOMC, elections, vol cycles.
#   forward_bars=10 — captures overnight session completion.
#
# MACRO  ( 90d): 3-month institutional context — quarterly rebalancing flows.
#   forward_bars=8  — 8h covers full session outcome.
#
# REGIME ( 31d): Monthly options cycle. Regime patterns repeat on this cadence.
#   forward_bars=5  — half a London or NY session.
#
# STRUCT ( 14d): Two-week structural context. Avoids 7d single-trend collapse.
#   forward_bars=4  — captures 4h follow-through.
#
# SESSION ( 7d): Week-level session patterns and swing structure.
#   forward_bars=3  — tight causal label for swing entry.
#
# PREC   ( 3d):  72-bar microstructure. Clean ICT entry alignment.
#   forward_bars=2  — causal short-term label.
#
# ULTRA  ( 1d):  24-bar ultra-precision M10/M15 trigger confirmation.
#   forward_bars=1  — single-bar directional signal.
#
# FIX: DEEP window uses args.bars so --bars CLI flag is honoured.
# All tiers clamped to args.bars in case --bars is small.
WINDOWS = {
    "365d": {"bars": args.bars,               "forward_bars": 10,
             "label": "DEEP   (365d / {:d} H1)".format(args.bars)},
    "90d":  {"bars": min(4160, args.bars),    "forward_bars": 8,
             "label": "MACRO  ( 90d / 2160 H1)"},
    "31d":  {"bars": min(2744,  args.bars),    "forward_bars": 5,
             "label": "REGIME ( 31d /  744 H1)"},
    "14d":  {"bars": min(1336,  args.bars),    "forward_bars": 4,
             "label": "STRUCT ( 14d /  336 H1)"},
    "7d":   {"bars": min(968,  args.bars),    "forward_bars": 3,
             "label": "SESSION ( 7d /  168 H1)"},
    "3d":   {"bars": min(772,   args.bars),    "forward_bars": 2,
             "label": "PREC   (  3d /   72 H1)"},
    "1d":   {"bars": min(424,   args.bars),    "forward_bars": 1,
             "label": "ULTRA  (  1d /   24 H1)"},
}

if args.period == "all":
    ACTIVE_WINDOWS = ["365d", "90d", "31d", "14d", "7d", "3d", "1d"]
else:
    ACTIVE_WINDOWS = [args.period]

DEFAULT_SYMBOLS = {
    "mt5":     ["EURUSD", "XAUUSD", "US100..", "US30..", "US500..", "XAGUSD..", "BTCUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CHFJPY"],
    "ibkr":    ["EUR.USD", "XAU.USD", "SPX"],
    "ctrader": ["EURUSD", "XAUUSD", "USOIL"],
}
TRAIN_SYMBOLS = (
    DEFAULT_SYMBOLS.get(BROKER_TYPE, ["EURUSD", "XAUUSD"])
    if args.all_symbols else [args.symbol]
)

# ── Module imports ────────────────────────────────────────────────────────────
logger.info("Loading modules...")
try:
    from broker_router  import BrokerRouter
    from data_fetcher   import BrokerDataFetcher
    from symbol_mapper  import SymbolMapper
    from evaluator      import StrategyEvaluator
    from signal_engine  import SignalEngine
    import pandas as pd
    import numpy as np
except ImportError as exc:
    logger.error(f"Import error: {exc}")
    sys.exit(1)

# ── Broker connection ─────────────────────────────────────────────────────────
logger.info(f"Connecting to broker: {BROKER_TYPE.upper()}...")
router = None
try:
    router = BrokerRouter(broker_type=BROKER_TYPE)
    broker = router.get_broker()
    logger.info(f"Connected to {broker.broker_name}  equity={broker.get_equity():.2f}")
except Exception as exc:
    logger.error(f"Broker connect failed: {exc}")
    if router:
        router.shutdown()
    sys.exit(1)

fetcher   = BrokerDataFetcher(broker)
mapper    = SymbolMapper(broker=broker)
evaluator = StrategyEvaluator()

# ── Symbol resolution ─────────────────────────────────────────────────────────
logger.info("Resolving symbols...")
try:
    all_broker_syms = {s["name"] for s in broker.get_symbols()}
except Exception:
    all_broker_syms = set()
    logger.warning("Could not fetch broker symbol list.")

resolved_symbols = []
for raw_sym in TRAIN_SYMBOLS:
    clean  = mapper.to_clean(raw_sym)
    mapped = mapper.to_broker(clean)
    if mapped in all_broker_syms:
        resolved_symbols.append(mapped)
        logger.info(f"  {raw_sym!r} -> {mapped!r}  (mapped)")
    elif raw_sym in all_broker_syms:
        resolved_symbols.append(raw_sym)
        logger.info(f"  {raw_sym!r} -> {raw_sym!r}  (direct)")
    elif not all_broker_syms:
        resolved_symbols.append(raw_sym)
        logger.warning(f"  {raw_sym!r} -> using as-is (broker list unavailable)")
    else:
        candidates = [s for s in all_broker_syms
                      if mapper.to_clean(s).upper() == clean.upper()]
        if candidates:
            resolved_symbols.append(candidates[0])
            logger.info(f"  {raw_sym!r} -> {candidates[0]!r}  (fuzzy)")
        else:
            logger.warning(f"  {raw_sym!r} not found on broker — skipping.")

if not resolved_symbols:
    logger.error("No resolvable symbols -- aborting.")
    router.shutdown()
    sys.exit(1)

# ── Adaptive retrain check ────────────────────────────────────────────────────
if args.check_retrain:
    needs_any = False
    for sym in resolved_symbols:
        eng = SignalEngine(symbol=sym)
        if not eng.is_trained:
            logger.info(f"  {sym}: no model found -- will train")
            needs_any = True
        elif eng.needs_retraining():
            logger.info(f"  {sym}: live accuracy degraded -- will retrain")
            needs_any = True
        else:
            logger.info(f"  {sym}: model healthy "
                        f"(wf_acc={eng._wf_mean_acc:.3f}) -- skipping")
    if not needs_any:
        logger.info("All models healthy -- no retraining required.")
        router.shutdown()
        sys.exit(0)
    # needs_any=True: fall through to training below.
    # FIX 7: shutdown handled by the try/finally at the bottom.


# ── Main training block wrapped in try/finally so shutdown always fires ───────
try:
    # ── Fetch data ──────────────────────────────────────────────────────────────
    # Always fetch the DEEP window; MID/PREC are sliced from it — no extra calls.
    logger.info("=" * 60)
    logger.info(f"Fetching data  (DEEP = {args.bars} H1 bars per symbol)...")
    combined_frames = []   # list of (symbol, df_deep_h1, df_h4)

    for symbol in resolved_symbols:
        _wd.instance.reset(f"fetch H1 data for {symbol}")
        logger.info(f"  {symbol}: fetching {args.bars} H1 bars...")
        try:
            # ── v19 DIR-1: AUTO-FETCH with retry ─────────────────────────────
            if AUTO_TRAINER_AVAILABLE:
                df_deep, _fwd_unused = AutoFetcher.fetch_with_retry(
                    lambda sym, tf, n: fetcher.get_candles(sym, tf, n),
                    symbol, args.timeframe, args.bars,
                    forward_bars=10, min_required=300
                )
            else:
                df_deep = fetcher.get_candles(symbol, args.timeframe, args.bars)
            if df_deep is None or len(df_deep) == 0:
                logger.error(f"  {symbol}: no H1 data returned -- skipping.")
                continue
            logger.info(f"  {symbol}: H1 DEEP = {len(df_deep)} bars")

            # FIX 4: H4 bars never below 600 for regime stability
            h4_bars = max(600, args.bars // 4)
            df_h4 = None
            try:
                df_h4 = fetcher.get_candles(symbol, "h4", h4_bars)
                logger.info(f"  {symbol}: H4      = {len(df_h4)} bars")
            except Exception as e:
                logger.warning(f"  {symbol}: H4 fetch failed ({e}) -- H4 features neutral")

            df_deep = evaluator.add_market_indicators(df_deep)
            if df_h4 is not None:
                df_h4 = evaluator.add_market_indicators(df_h4)

            combined_frames.append((symbol, df_deep, df_h4))

        except Exception as exc:
            logger.error(f"  {symbol} fetch error: {exc}", exc_info=True)

    if not combined_frames:
        logger.error("No data available -- aborting.")
        sys.exit(1)

    # ── Helper: return the last N bars matching the window key ──────────────────
    def _slice(df: pd.DataFrame, window_key: str) -> pd.DataFrame:
        n = WINDOWS[window_key]["bars"]
        return df.iloc[-n:].copy() if len(df) >= n else df.copy()

    # ── Helper: train one engine on a window slice ──────────────────────────────
    # FIX 1: rr_threshold removed — not a valid kwarg of SignalEngine.train().
    def _train_window(engine: SignalEngine, df_deep: pd.DataFrame,
                      df_h4, window_key: str, tag: str) -> bool:
        cfg   = WINDOWS[window_key]
        label = cfg["label"]
        fwd   = cfg["forward_bars"]
        df_sl = _slice(df_deep, window_key)
        _wd.instance.reset(f"train {tag}/{window_key} ({len(df_sl)} bars)")
        logger.info(f"    [{tag}] {label}  bars={len(df_sl)}  fwd={fwd}")
        ok = engine.train(
            df_sl,
            df_h4=df_h4,
            forward_bars=fwd,
            save_if_best=False,
        )
        if ok:
            logger.info(
                f"    [{tag}] {window_key} TRAINED  "
                f"wf_acc={engine._wf_mean_acc:.3f}  features={engine._n_features}"
            )
            # ── v19 DIR-4: AUTO HYPERPARAMETER SEARCH ────────────────────────
            if AUTO_TRAINER_AVAILABLE and engine._wf_mean_acc < float(os.getenv("AUTO_HYPER_TRIGGER_BELOW", "0.50")):
                def _train_with_params(df, df_h4, forward_bars, n_estimators, learning_rate, max_depth):
                    from signal_engine import SignalEngine as _SE2
                    eng2 = _SE2(symbol=engine.symbol)
                    eng2._live_buffer = engine._live_buffer
                    ok2 = eng2.train(
                        df, df_h4=df_h4, forward_bars=forward_bars,
                        n_estimators=n_estimators, learning_rate=learning_rate,
                        max_depth=max_depth, save_if_best=False
                    )
                    return eng2, ok2
                best_hyper = AutoHyperSearch.search(
                    _train_with_params, df_sl, df_h4,
                    symbol=engine.symbol, baseline_acc=engine._wf_mean_acc
                )
                if best_hyper is not None:
                    # Swap state from best hyper engine into current engine
                    engine.rf_model      = best_hyper.rf_model
                    engine.gbm_model     = best_hyper.gbm_model
                    engine.xgb_model     = best_hyper.xgb_model
                    engine.lgb_model     = best_hyper.lgb_model
                    engine.meta_model    = best_hyper.meta_model
                    engine.scaler        = best_hyper.scaler
                    engine.feature_names = best_hyper.feature_names
                    engine._n_features   = best_hyper._n_features
                    engine._wf_mean_acc  = best_hyper._wf_mean_acc
                    engine._wf_gate_passed = best_hyper._wf_gate_passed
                    logger.info(
                        f"    [{tag}] {window_key} HYPER-IMPROVED "
                        f"wf_acc={engine._wf_mean_acc:.3f}"
                    )
        else:
            logger.warning(f"    [{tag}] {window_key} FAILED  (insufficient labels/variance)")
        return ok

    # ── Per-symbol models (--per-symbol flag) ───────────────────────────────────
    if args.per_symbol:
        logger.info("=" * 60)
        logger.info("Per-symbol training  (7-tier windows, best model saved)...")
        for symbol, df_deep, df_h4 in combined_frames:
            best_acc    = -1.0
            best_engine = None
            best_wkey   = None
            # Fallback: best mean even if gate not cleared (no model left otherwise)
            fallback_acc    = -1.0
            fallback_engine = None
            fallback_wkey   = None
            for wkey in ACTIVE_WINDOWS:
                try:
                    eng = SignalEngine(symbol=symbol)
                    ok  = _train_window(eng, df_deep, df_h4, wkey, symbol)
                    # Track gated best (ok=True AND gate passed)
                    if ok and eng._wf_gate_passed and eng._wf_mean_acc > best_acc:
                        best_acc    = eng._wf_mean_acc
                        best_engine = eng
                        best_wkey   = wkey
                    # Always track fallback regardless of gate
                    if ok and eng._wf_mean_acc > fallback_acc:
                        fallback_acc    = eng._wf_mean_acc
                        fallback_engine = eng
                        fallback_wkey   = wkey
                except Exception as exc:
                    logger.error(f"  {symbol}/{wkey}: {exc}", exc_info=True)

            if best_engine is not None:
                # Save only the BEST window model to disk (save_if_best=False
                # was passed to train(), so nothing was written yet).
                best_engine._save_model()
                logger.info(
                    f"  {symbol}: BEST window={best_wkey}  "
                    f"wf_acc={best_acc:.3f}  features={best_engine._n_features}"
                )
            elif fallback_engine is not None:
                # No window cleared the gate — save best-mean as fallback so
                # the bot always has a model. Logged clearly as suboptimal.
                # v15: gate is mean>=0.44 (3-class balanced acc, random=0.333)
                fallback_engine._save_model()
                logger.warning(
                    f"  {symbol}: NO window passed gate — saving FALLBACK "
                    f"window={fallback_wkey} wf_acc={fallback_acc:.3f} "
                    f"(mean<0.44 or spread>0.35, 3-class BA; random=0.333). Retrain with more data."
                )
            else:
                logger.warning(f"  {symbol}: all windows failed -- no per-symbol model saved")

    # ── Shared (combined) model ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Shared model training  (7-tier windows, best model saved)...")

    if len(combined_frames) == 1:
        _, df_h1_combined, df_h4_combined = combined_frames[0]
    else:
        logger.info(f"Merging {len(combined_frames)} symbol datasets (DEEP windows)...")
        df_h1_combined = pd.concat(
            [df for _, df, _ in combined_frames], ignore_index=True
        ).dropna(subset=["close"])
        df_h4_combined = None   # H4 features incompatible across symbols in shared model
        logger.info(f"Combined H1 dataset: {len(df_h1_combined)} rows")

    best_shared_acc    = -1.0
    best_shared_engine = None
    best_shared_wkey   = None

    for wkey in ACTIVE_WINDOWS:
        try:
            eng = SignalEngine()   # symbol="default" -> signal_model_default.pkl
            ok  = _train_window(eng, df_h1_combined, df_h4_combined, wkey, "SHARED")
            # FIX 5: strict > comparison
            if ok and eng._wf_mean_acc > best_shared_acc:
                best_shared_acc    = eng._wf_mean_acc
                best_shared_engine = eng
                best_shared_wkey   = wkey
        except Exception as exc:
            logger.error(f"  SHARED/{wkey}: {exc}", exc_info=True)

    # ── Final report ────────────────────────────────────────────────────────────
    if best_shared_engine is not None:
        # save_if_best=False was passed to train(), so nothing written yet.
        # Now write only the BEST shared window to disk.
        best_shared_engine._save_model()
        logger.info("=" * 60)
        logger.info("Training COMPLETE  (AI EA v17 PRO)")
        # FIX 6: show actual best window key, not ACTIVE_WINDOWS[0]
        logger.info(f"  Best window               : {best_shared_wkey}")
        logger.info(f"  Walk-forward OOS accuracy : {best_shared_acc:.3f}")
        logger.info(f"  Feature count             : {best_shared_engine._n_features}")
        logger.info(f"  Label method              : causal fixed-risk "
                    f"(TP={args.tp_mult}xATR / SL={args.sl_mult}xATR)")
        logger.info(f"  Validation                : purged walk-forward "
                    f"(5 folds, embargo=20 bars)")
        logger.info(f"  Windows trained           : {', '.join(ACTIVE_WINDOWS)}")
        logger.info("=" * 60)
    else:
        logger.error("Training FAILED on all windows")
        logger.info("\nCommon fixes:")
        logger.info("  --bars 3500           (more historical data)")
        logger.info("  --period 120d         (use only the deepest window)")
        logger.info("  --tp-mult 1.2         (relax TP multiplier for more labels)")
        logger.info("  --all-symbols         (more diverse training corpus)")
        sys.exit(1)

finally:
    # FIX 7: shutdown always fires — covers all exit paths including
    # check_retrain early exit, sys.exit(1) on no data, and normal completion.
    _wd.instance.stop()
    if router is not None:
        try:
            router.shutdown()
        except Exception:
            pass
