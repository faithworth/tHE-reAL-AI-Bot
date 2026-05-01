"""
run_backtest.py — Full ML Backtest Runner (AI EA v13)
======================================================
Fetches real H1 data via MT5, runs the v10 SignalEngine ML model to generate
signals, then evaluates with the Backtester (walk-forward + Monte Carlo).

Usage:
    python run_backtest.py
    python run_backtest.py --bars 8000
    python run_backtest.py --symbols XAUUSD.. BTCUSD..
    python run_backtest.py --lot 0.05

Saves results to:  backtest_results.txt  (same folder)
"""

import os, sys, logging, argparse, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_backtest")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="AI EA v13 Full ML Backtest Runner (7-tier MTF)")
parser.add_argument("--symbols", nargs="+",
                    default=["XAUUSD..","BTCUSD..","US100..","US30..","US500..","XAGUSD..", "EURUSD..","GBPUSD..","USDJPY..","AUDUSD..","USDCHF..","NZDUSD..","EURGBP..","EURJPY..","GBPJPY..","AUDJPY..","CHFJPY.."],
                    help="Symbols to backtest")
parser.add_argument("--bars",    type=int, default=8000,
                    help="H1 bars of history to fetch per symbol")
parser.add_argument("--lot",     type=float, default=0.01,
                    help="Lot size for PnL simulation")
parser.add_argument("--wf-windows", type=int, default=5,
                    help="Walk-forward windows")
parser.add_argument("--mc-runs",    type=int, default=500,
                    help="Monte Carlo simulation runs")
args = parser.parse_args()

os.makedirs("models", exist_ok=True)
os.makedirs("data",   exist_ok=True)
os.makedirs("logs",   exist_ok=True)

# ── Imports ───────────────────────────────────────────────────────────────────
logger.info("Loading modules...")
try:
    from broker_router   import BrokerRouter
    from data_fetcher    import BrokerDataFetcher
    from symbol_mapper   import SymbolMapper
    from evaluator       import StrategyEvaluator
    from signal_engine   import SignalEngine
    from Backtester      import Backtester
    import numpy as np
    import pandas as pd
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

BROKER_TYPE = os.getenv("BROKER_TYPE", "mt5").lower().strip()

# ── Broker connection ─────────────────────────────────────────────────────────
logger.info(f"Connecting to {BROKER_TYPE.upper()}...")
router = None
try:
    router  = BrokerRouter(broker_type=BROKER_TYPE)
    broker  = router.get_broker()
    equity  = broker.get_equity()
    logger.info(f"Connected — equity: {equity:.2f}")
except Exception as e:
    logger.error(f"Broker connect failed: {e}")
    if router:
        router.shutdown()
    sys.exit(1)

fetcher   = BrokerDataFetcher(broker)
mapper    = SymbolMapper(broker=broker)
evaluator = StrategyEvaluator()
bt        = Backtester()

# ── Resolve symbols ───────────────────────────────────────────────────────────
try:
    all_broker_syms = {s["name"] for s in broker.get_symbols()}
except Exception:
    all_broker_syms = set()

resolved = []
for raw in args.symbols:
    clean  = mapper.to_clean(raw)
    mapped = mapper.to_broker(clean)
    if mapped in all_broker_syms:
        resolved.append((raw, mapped))
    elif raw in all_broker_syms:
        resolved.append((raw, raw))
    elif not all_broker_syms:
        resolved.append((raw, raw))
    else:
        candidates = [s for s in all_broker_syms
                      if mapper.to_clean(s).upper() == clean.upper()]
        if candidates:
            resolved.append((raw, candidates[0]))
        else:
            logger.warning(f"  {raw!r} not found on broker — skipping")

if not resolved:
    logger.error("No resolvable symbols — aborting.")
    router.shutdown()
    sys.exit(1)

logger.info(f"Symbols to backtest: {[r[1] for r in resolved]}")

# ── Helper: ML signal strategy dict ──────────────────────────────────────────
def build_ml_strategy(engine: SignalEngine, df: pd.DataFrame,
                      df_h4=None) -> dict:
    """
    FAST vectorized signal generation — builds features once for the full
    DataFrame, then runs predict_proba in a single batch call.

    v11 FIX: Pass named DataFrame to LGB/XGB at predict time to eliminate
    UserWarning 'X does not have valid feature names'.  RF/GBM receive numpy.

    v11 FIX: Track direction per bar (BUY=+1, SELL=-1) so the backtest can
    simulate both long and short trades correctly rather than treating all
    signals as buys.

    v11 FIX: Run ATR-based SL/TP simulation (the same labeling logic the model
    was trained on) rather than the naive 1-bar open→close hold.  This makes
    the backtest self-consistent with training.
    """
    import warnings

    try:
        # Build full feature matrix once
        feat = engine._build_features(df, df_h4=df_h4)
        if feat is None or len(feat) < 50:
            raise ValueError("Feature matrix too small")

        feat = engine._align_features(feat)
        X    = feat.values.astype(np.float64)
        X_sc = engine.scaler.transform(X)

        # FIX UserWarning: build named DataFrame for LGB/XGB, numpy for RF/GBM
        X_named = None
        if engine.feature_names and X_sc.shape[1] == len(engine.feature_names):
            X_named = pd.DataFrame(X_sc, columns=engine.feature_names)

        # Batch ensemble proba — suppress any residual sklearn warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            probas = engine._ensemble_proba(X_sc)   # shape (n, 3): [NO_TRADE, BUY, SELL]

        p_buy  = probas[:, 1]
        p_sell = probas[:, 2]
        p_no   = probas[:, 0]

        from signal_engine import MIN_SIGNAL_PROBABILITY

        # Realistic filter: match what the live system actually passes.
        # Live system requires: (1) prob >= MIN_SIGNAL_PROBABILITY,
        # (2) directional class must clearly beat NO_TRADE prob.
        # Backtest was previously taking 61-65% of bars — far more than live.
        # Apply a margin filter: directional prob must beat the competing
        # directional class by at least 0.04 AND beat NO_TRADE by at least 0.05.
        MARGIN_VS_OPPOSITE = 0.03   # BUY must exceed SELL by this much (or vice versa)
        MARGIN_VS_NO_TRADE = 0.04   # directional prob must beat NO_TRADE by this much

        # Direction array: +1 = BUY, -1 = SELL, 0 = NO_TRADE
        direction_arr = np.zeros(len(p_buy), dtype=int)
        buy_mask  = (
            (p_buy  >= MIN_SIGNAL_PROBABILITY) &
            (p_buy  > p_sell + MARGIN_VS_OPPOSITE) &
            (p_buy  > p_no   + MARGIN_VS_NO_TRADE)
        )
        sell_mask = (
            (p_sell >= MIN_SIGNAL_PROBABILITY) &
            (p_sell > p_buy  + MARGIN_VS_OPPOSITE) &
            (p_sell > p_no   + MARGIN_VS_NO_TRADE)
        )
        direction_arr[buy_mask]  =  1
        direction_arr[sell_mask] = -1

        feat_index = feat.index
        dir_series = pd.Series(direction_arr, index=feat_index)
        dir_series = dir_series.reindex(df.index).fillna(0).astype(int)

        # Aggregate signal (1 = trade, any direction)
        sig_series = (dir_series != 0).astype(int)

    except Exception as e:
        logger.warning(f"  Fast signal generation failed ({e}) — using zero signals")
        sig_series = pd.Series(0, index=df.index)
        dir_series = pd.Series(0, index=df.index)

    total_signals = int(sig_series.sum())
    logger.info(f"  ML signals generated: {total_signals} "
                f"({total_signals/len(df)*100:.1f}% of bars)")

    # v12: probability distribution diagnostics
    # High signal rate (>15%) = model lacks selectivity = no edge
    # Healthy range: 3-12% of bars should generate signals
    try:
        all_max_prob = np.maximum(p_buy, p_sell)
        buy_signals  = int((dir_series == 1).sum())
        sell_signals = int((dir_series == -1).sum())
        logger.info(
            f"  Signal breakdown: BUY={buy_signals} SELL={sell_signals} "
            f"| prob p50={np.percentile(all_max_prob,50):.3f} "
            f"p90={np.percentile(all_max_prob,90):.3f} "
            f"p99={np.percentile(all_max_prob,99):.3f}"
        )
        signal_pct = total_signals / len(df) * 100
        if signal_pct > 20:
            logger.warning(
                f"  ⚠  Signal rate {signal_pct:.1f}% is HIGH (>20%) — "
                "model lacks selectivity. Retrain needed for real edge."
            )
        elif signal_pct > 10:
            logger.warning(
                f"  ⚠  Signal rate {signal_pct:.1f}% is ELEVATED (>10%) — "
                "backtest trades > what live system will take (live adds regime/MTF/session gates)."
            )
        elif signal_pct < 1:
            logger.warning(
                f"  ⚠  Signal rate {signal_pct:.1f}% is VERY LOW (<1%) — "
                "threshold may be too strict. Check MIN_SIGNAL_PROBABILITY."
            )
    except Exception:
        pass

    return {
        "name":                  "ML_SignalEngine_v11",
        "direction":             "buy",   # legacy field; actual direction in _dir_series
        "rules":                 ["sma20"],
        "_precomputed_signals":  sig_series,
        "_precomputed_direction": dir_series,
    }


def run_atr_backtest(strategy: dict, df: pd.DataFrame,
                     symbol: str = "", lot: float = 0.01,
                     tp_atr: float = 2.0, sl_atr: float = 1.0,
                     max_bars: int = 30) -> dict:
    """
    Proper ATR-based SL/TP backtest — mirrors the causal labeling used during
    ML training.  For each signal bar:
      - Entry = next bar open
      - BUY:  TP = entry + tp_atr * ATR,  SL = entry - sl_atr * ATR
      - SELL: TP = entry - tp_atr * ATR,  SL = entry + sl_atr * ATR
    Scan forward up to max_bars for first hit.  If neither hit: flat-close.

    This replaces the 1-bar open→close hold which ignores SL/TP entirely
    and produces artificially terrible backtest results.
    """
    from Backtester import _get_spec
    import numpy as np

    sig        = strategy.get("_precomputed_signals")
    directions = strategy.get("_precomputed_direction")
    if sig is None:
        return {"win_rate": 0, "profit": 0, "profit_factor": 0,
                "max_drawdown": 0, "sharpe": 0, "total_trades": 0,
                "score": 0, "equity_curve": []}

    spec     = _get_spec(symbol)
    cost_pts = spec["spread_pts"] + spec["slippage_pts"]
    comm     = spec["commission_usd"] * lot
    cs       = spec["contract_size"] * lot

    sig_arr = sig.reindex(df.index).fillna(0).values.astype(int)
    dir_arr = (directions.reindex(df.index).fillna(0).values.astype(int)
               if directions is not None else sig_arr)  # fallback: all buy

    opens  = df["open"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)

    # ATR (14-bar)
    if "atr" in df.columns and df["atr"].notna().sum() > 14:
        atr_vals = df["atr"].values.astype(float)
    else:
        from signal_engine import SignalEngine
        atr_vals = SignalEngine._calc_atr(df).values.astype(float)

    n      = len(df)
    trades = []

    # Spread-to-ATR viability: skip bars where cost > 25% of expected TP.
    # TP = tp_atr * ATR_bar; cost = spread + slippage.
    # If cost / (tp_atr * ATR) > 0.25 the trade is structurally uneconomical.
    SPREAD_ATR_MAX = 0.25

    for i in range(n - 1):
        if sig_arr[i] == 0:
            continue
        direction = int(dir_arr[i])
        if direction == 0:
            direction = 1  # default buy if direction not set

        entry   = opens[i + 1] + cost_pts * direction   # include spread on entry side
        atr_val = atr_vals[i]
        if not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(entry):
            continue

        # Skip economically unviable bars (spread eats > 25% of TP)
        if atr_val > 0 and (cost_pts / (tp_atr * atr_val)) > SPREAD_ATR_MAX:
            continue

        if direction == 1:   # BUY
            tp_price = entry + tp_atr * atr_val
            sl_price = entry - sl_atr * atr_val
        else:                # SELL
            tp_price = entry - tp_atr * atr_val
            sl_price = entry + sl_atr * atr_val

        pnl = None
        end = min(i + 2 + max_bars, n)
        for j in range(i + 2, end):
            hi = highs[j]; lo = lows[j]
            if direction == 1:
                if hi >= tp_price:
                    pnl = (tp_price - entry) * cs - comm; break
                if lo <= sl_price:
                    pnl = (sl_price - entry) * cs - comm; break
            else:
                if lo <= tp_price:
                    pnl = (entry - tp_price) * cs - comm; break
                if hi >= sl_price:
                    pnl = (entry - sl_price) * cs - comm; break

        if pnl is None:
            # Flat-close at last bar close
            exit_price = closes[end - 1]
            pnl = (exit_price - entry) * direction * cs - comm

        trades.append(pnl)

    if not trades:
        return {"win_rate": 0, "profit": 0, "profit_factor": 0,
                "max_drawdown": 0, "sharpe": 0, "total_trades": 0,
                "score": 0, "equity_curve": []}

    arr  = np.array(trades)
    wins = arr[arr > 0]; loss = arr[arr <= 0]
    wr   = len(wins) / len(arr) * 100
    gp   = float(wins.sum()) if len(wins) else 0.0
    gl   = float(abs(loss.sum())) if len(loss) else 0.0
    # FIX: cap PF at 99.99 — no-loss windows return infinity with 1e-9 denominator
    pf   = min(gp / gl, 99.99) if gl > 0 else (99.99 if gp > 0 else 0.0)
    eq   = np.cumsum(arr); pk = np.maximum.accumulate(eq)
    mdd  = float((pk - eq).max()) if len(eq) else 0.0
    sh   = float((arr.mean() / arr.std()) * np.sqrt(252)) if arr.std() > 0 else 0.0
    pr   = float(eq[-1])
    sc   = wr * 0.4 + min(pf, 5) * 0.4 * 20 + sh * 0.2 * 10
    return {
        "win_rate": round(wr, 2), "profit": round(pr, 2),
        "profit_factor": round(pf, 3), "max_drawdown": round(mdd, 2),
        "sharpe": round(sh, 3), "total_trades": len(arr),
        "score": round(sc, 2), "equity_curve": eq.tolist(),
    }


_original_generate_signals = Backtester._generate_signals

def _ml_generate_signals(self, strategy, df):
    if "_precomputed_signals" in strategy:
        sig = strategy["_precomputed_signals"]
        # Align index
        sig = sig.reindex(df.index).fillna(0).astype(int)
        return sig
    return _original_generate_signals(self, strategy, df)

Backtester._generate_signals = _ml_generate_signals

# ── Results storage ───────────────────────────────────────────────────────────
all_results = []

DIVIDER  = "=" * 72
DIVIDER2 = "-" * 72

def fmt_row(label, value, good_fn=None):
    mark = ""
    if good_fn is not None:
        try:
            mark = " ✓" if good_fn(value) else " ✗"
        except Exception:
            pass
    return f"  {label:<28} {value}{mark}"

# ── Main loop ─────────────────────────────────────────────────────────────────
try:
    for raw_sym, symbol in resolved:
        logger.info(DIVIDER)
        logger.info(f"BACKTESTING: {symbol}  ({args.bars} H1 bars)")
        logger.info(DIVIDER)

        result_block = {
            "symbol": symbol,
            "status": "failed",
        }

        try:
            # 1. Fetch data
            logger.info(f"  Fetching {args.bars} H1 bars...")
            df = fetcher.get_candles(symbol, "h1", args.bars)
            if df is None or len(df) < 300:
                logger.error(f"  Insufficient data ({len(df) if df is not None else 0} bars) — skipping")
                all_results.append(result_block)
                continue
            logger.info(f"  H1 bars received: {len(df)}")

            # H4 for multi-timeframe features
            df_h4 = None
            try:
                df_h4 = fetcher.get_candles(symbol, "h4", max(600, args.bars // 4))
                logger.info(f"  H4 bars received: {len(df_h4)}")
            except Exception:
                logger.warning("  H4 fetch failed — single timeframe mode")

            # 2. Add indicators
            df    = evaluator.add_market_indicators(df)
            if df_h4 is not None:
                df_h4 = evaluator.add_market_indicators(df_h4)

            # 3. Load ML engine
            engine = SignalEngine(symbol=symbol)
            if not engine.is_trained:
                # Try default/shared model
                engine = SignalEngine(symbol="default")
            if not engine.is_trained:
                logger.warning(f"  No trained model for {symbol} — run trainer.py first")
                result_block["status"] = "no_model"
                all_results.append(result_block)
                continue

            logger.info(f"  Model: {engine._n_features} features, "
                        f"wf_acc={engine._wf_mean_acc:.3f}")

            # 4. Generate ML signals
            logger.info("  Generating ML signals (this takes ~30s)...")
            strategy = build_ml_strategy(engine, df, df_h4)

            # 5. Proper ATR-based SL/TP backtest (self-consistent with ML training labels)
            logger.info("  Running backtest...")

            # v12: per-instrument viability warning
            from Backtester import _get_spec
            _spec = _get_spec(symbol)
            _cost = _spec["spread_pts"] + _spec["slippage_pts"]
            _min_atr_est = float(df["high"].sub(df["low"]).rolling(14).mean().dropna().median())
            if _min_atr_est > 0:
                _spread_ratio = _cost / (2.0 * _min_atr_est)
                if _spread_ratio > 0.25:
                    logger.warning(
                        f"  ⚠  {symbol}: spread/slippage ({_cost:.5f}) is "
                        f"{_spread_ratio:.0%} of median ATR ({_min_atr_est:.5f}). "
                        f"Instrument is structurally uneconomical at {args.lot} lot. "
                        "Consider larger lot or excluding this symbol."
                    )

            bt_result = run_atr_backtest(strategy, df, symbol=symbol, lot=args.lot)

            # 6. Walk-forward (ATR-based, OOS windows)
            logger.info(f"  Running walk-forward ({args.wf_windows} windows)...")
            n_wf = args.wf_windows
            total = len(df); wsize = total // n_wf; oos_results = []
            for wi in range(n_wf):
                s = wi * wsize; e = s + wsize
                sp = int(s + (e - s) * 0.7)
                oos_df = df.iloc[sp:e]
                if len(oos_df) >= 50:
                    oos_r = run_atr_backtest(strategy, oos_df, symbol=symbol, lot=args.lot)
                    oos_results.append(oos_r)
            if oos_results:
                # Cap individual window PF before averaging — a no-loss OOS window
                # (legitimate but rare) should not skew the mean to infinity.
                _pf_vals = [min(r["profit_factor"], 99.99) for r in oos_results]
                wf_result = {
                    "type": "walk_forward", "n_windows": len(oos_results),
                    "win_rate":      round(float(np.mean([r["win_rate"]      for r in oos_results])), 2),
                    "profit_factor": round(float(np.mean(_pf_vals)), 3),
                    "max_drawdown":  round(float(np.max( [r["max_drawdown"]  for r in oos_results])), 2),
                    "sharpe":        round(float(np.mean([r["sharpe"]        for r in oos_results])), 3),
                    "score":         round(float(np.mean([r["score"]         for r in oos_results])), 2),
                    "profit":        round(float(sum(     r["profit"]        for r in oos_results)), 2),
                }
            else:
                wf_result = {"win_rate": 0, "profit_factor": 0, "max_drawdown": 0,
                             "sharpe": 0, "profit": 0, "score": 0, "n_windows": 0}

            # 7. Monte Carlo (resample trades from full ATR backtest)
            logger.info(f"  Running Monte Carlo ({args.mc_runs} runs)...")
            eq_curve = bt_result.get("equity_curve", [])
            if len(eq_curve) > 1:
                trade_pnls = np.diff([0.0] + list(eq_curve))
                mdd_mc = []
                rng = np.random.default_rng(42)
                for _ in range(args.mc_runs):
                    eq_sim = np.cumsum(rng.choice(trade_pnls, size=len(trade_pnls), replace=True))
                    pk_sim = np.maximum.accumulate(eq_sim)
                    mdd_mc.append(float((pk_sim - eq_sim).max()))
                mdd_mc = np.array(mdd_mc)
                mc_result = {
                    "type": "monte_carlo", "n_runs": args.mc_runs,
                    "base_profit":       round(float(trade_pnls.sum()), 2),
                    "win_rate":          round(float((trade_pnls > 0).mean() * 100), 2),
                    "profit_factor":     round(float(
                        min(trade_pnls[trade_pnls > 0].sum() /
                            max(abs(trade_pnls[trade_pnls <= 0].sum()), 1e-6), 99.99)
                    ), 3),
                    "max_drawdown_p50":  round(float(np.percentile(mdd_mc, 50)), 2),
                    "max_drawdown_p95":  round(float(np.percentile(mdd_mc, 95)), 2),
                    "max_drawdown_p99":  round(float(np.percentile(mdd_mc, 99)), 2),
                }
            else:
                mc_result = {"error": "no_trades"}

            result_block.update({
                "status":    "ok",
                "bt":        bt_result,
                "wf":        wf_result,
                "mc":        mc_result,
                "n_features": engine._n_features,
                "wf_acc":    engine._wf_mean_acc,
                "bars":      len(df),
            })

        except Exception as e:
            logger.error(f"  Error backtesting {symbol}: {e}", exc_info=True)
            result_block["error"] = str(e)

        all_results.append(result_block)

    # ── Print summary ─────────────────────────────────────────────────────────
    lines = []
    lines.append("")
    lines.append(DIVIDER)
    lines.append("  AI EA v13 — BACKTEST RESULTS SUMMARY (7-tier MTF)")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Bars/symbol: {args.bars}   Lot: {args.lot}   "
                 f"WF windows: {args.wf_windows}   MC runs: {args.mc_runs}")
    lines.append(DIVIDER)

    for r in all_results:
        sym = r["symbol"]
        lines.append("")
        lines.append(f"  ▶ {sym}")
        lines.append(DIVIDER2)

        if r["status"] == "no_model":
            lines.append("  ⚠  No trained model found. Run: python trainer.py --symbol " + sym)
            continue
        if r["status"] == "failed" or "bt" not in r:
            lines.append("  ✗  Failed: " + r.get("error", "unknown error"))
            continue

        bt  = r["bt"]
        wf  = r["wf"]
        mc  = r["mc"]

        lines.append(fmt_row("Bars of history:",     r["bars"]))
        lines.append(fmt_row("ML features:",         r["n_features"]))
        lines.append(fmt_row("ML WF accuracy:",      f"{r['wf_acc']:.3f}",
                              lambda v: v > 0.33))
        lines.append("")
        lines.append("  BACKTEST (full period):")
        lines.append(fmt_row("  Total trades:",       bt["total_trades"],
                              lambda v: v >= 30))
        lines.append(fmt_row("  Win rate:",           f"{bt['win_rate']:.1f}%",
                              lambda v: v >= 50))
        lines.append(fmt_row("  Profit factor:",      f"{bt['profit_factor']:.3f}",
                              lambda v: v >= 1.2))
        lines.append(fmt_row("  Sharpe ratio:",       f"{bt['sharpe']:.3f}",
                              lambda v: v >= 0.8))
        lines.append(fmt_row("  Max drawdown ($):",   f"${bt['max_drawdown']:.2f}"))
        lines.append(fmt_row("  Net profit ($):",     f"${bt['profit']:.2f}",
                              lambda v: v > 0))
        lines.append(fmt_row("  Score:",              f"{bt['score']:.1f}"))
        lines.append("")
        lines.append("  WALK-FORWARD (OOS average):")
        lines.append(fmt_row("  WF windows:",         wf.get("n_windows", "-")))
        lines.append(fmt_row("  WF win rate:",        f"{wf.get('win_rate', 0):.1f}%",
                              lambda v: v >= 50))
        lines.append(fmt_row("  WF profit factor:",   f"{wf.get('profit_factor', 0):.3f}",
                              lambda v: v >= 1.2))
        lines.append(fmt_row("  WF Sharpe:",          f"{wf.get('sharpe', 0):.3f}",
                              lambda v: v >= 0.8))
        lines.append(fmt_row("  WF net profit ($):",  f"${wf.get('profit', 0):.2f}",
                              lambda v: v > 0))
        lines.append("")
        if "error" not in mc:
            lines.append("  MONTE CARLO (drawdown risk):")
            lines.append(fmt_row("  Base profit ($):",    f"${mc.get('base_profit', 0):.2f}"))
            lines.append(fmt_row("  Win rate:",           f"{mc.get('win_rate', 0):.1f}%"))
            lines.append(fmt_row("  Profit factor:",      f"{mc.get('profit_factor', 0):.3f}"))
            lines.append(fmt_row("  Max DD p50 ($):",     f"${mc.get('max_drawdown_p50', 0):.2f}"))
            lines.append(fmt_row("  Max DD p95 ($):",     f"${mc.get('max_drawdown_p95', 0):.2f}"))
            lines.append(fmt_row("  Max DD p99 ($):",     f"${mc.get('max_drawdown_p99', 0):.2f}"))

        # Verdict
        lines.append("")
        wf_pf = wf.get("profit_factor", 0)
        wf_sh = wf.get("sharpe", 0)
        wf_wr = wf.get("win_rate", 0)
        wf_pr = wf.get("profit", 0)
        if   wf_pf >= 1.5 and wf_sh >= 1.0 and wf_pr > 0:
            verdict = "🟢 STRONG EDGE — looks profitable, consider live testing"
        elif wf_pf >= 1.2 and wf_pr > 0:
            verdict = "🟡 MODERATE EDGE — promising but needs more data/tuning"
        elif wf_pr > 0:
            verdict = "🟠 WEAK EDGE — marginally positive, high risk"
        else:
            verdict = "🔴 NO EDGE — not profitable on OOS data, retrain needed"
        lines.append(f"  VERDICT: {verdict}")
        lines.append(DIVIDER2)

    lines.append("")
    lines.append(DIVIDER)
    lines.append("  ✓ = meets target threshold   ✗ = below threshold")
    lines.append("  Key thresholds: Win rate ≥50%  PF ≥1.2  Sharpe ≥0.8  Trades ≥30")
    lines.append("  NOTE: Backtest uses ML probability filter only. Live system adds")
    lines.append("  regime/MTF/session/composite-score gates → expect ~5-20x fewer live")
    lines.append("  trades than shown here. VERDICT reflects ML edge, not exact live count.")
    lines.append(DIVIDER)
    lines.append("")

    output = "\n".join(lines)
    print(output)

    # Save to file
    out_path = "backtest_results.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    logger.info(f"Results saved to: {out_path}")

finally:
    if router:
        try:
            router.shutdown()
        except Exception:
            pass
