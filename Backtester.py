"""
Backtester.py — Upgraded backtest engine (AI EA v4)
Spread/slippage/commission simulation, walk-forward, Monte Carlo.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Per-instrument specifications ─────────────────────────────────────────────
# Each entry defines:
#   spread_pts       : typical spread in price points (NOT pips)
#   slippage_pts     : typical slippage in price points
#   commission_usd   : round-trip commission in USD per lot (0 = spread-only)
#   contract_size    : units per lot (used in PnL: profit = price_diff * contract_size * lot)
#   point            : smallest price movement (tick size)
#
# PnL formula (correct):
#   raw_move  = exit_price - entry_price   (in price units, e.g. 2345.10 - 2344.80)
#   cost_pts  = spread_pts + slippage_pts
#   net_move  = raw_move - cost_pts
#   pnl_usd   = net_move * contract_size * lot - commission_usd
#
# This replaces the old broken formula:
#   pnl = (exit - entry) * 100_000 * point * lot
# which applied a forex multiplier to every instrument regardless of type,
# producing $231M losses on BTC and $12k drawdowns on XAUUSD at 0.01 lot.

SYMBOL_SPECS = {
    # Forex majors/minors — standard 100k contract
    "default":  {"spread_pts": 0.00020, "slippage_pts": 0.00005,
                 "commission_usd": 7.0,  "contract_size": 100_000, "point": 0.00001},
    "JPY":      {"spread_pts": 0.020,   "slippage_pts": 0.005,
                 "commission_usd": 7.0,  "contract_size": 100_000, "point": 0.001},

    # Gold — 100 oz per lot, price in USD/oz
    "XAUUSD":   {"spread_pts": 0.30,    "slippage_pts": 0.10,
                 "commission_usd": 0.0,  "contract_size": 100,     "point": 0.01},

    # Silver — 5000 oz per lot (real CFD spec), price in USD/oz.
    # NOTE: at 0.01 lot (50 oz) the spread of $0.03 = $1.50 cost vs typical
    # $0.20-0.40 ATR/bar — the instrument is structurally unprofitable below
    # 0.10 lot.  Backtest reflects real economics; do not trade XAGUSD below 0.10 lot.
    "XAGUSD":   {"spread_pts": 0.03,    "slippage_pts": 0.01,
                 "commission_usd": 0.0,  "contract_size": 5_000,   "point": 0.001},

    # Bitcoin — 1 BTC per lot, price in USD
    "BTCUSD":   {"spread_pts": 50.0,    "slippage_pts": 20.0,
                 "commission_usd": 0.0,  "contract_size": 1,       "point": 0.01},

    # US indices — $1 per point per lot (mini contract)
    "US100":    {"spread_pts": 1.0,     "slippage_pts": 0.5,
                 "commission_usd": 0.0,  "contract_size": 1,       "point": 0.01},
    "US30":     {"spread_pts": 3.0,     "slippage_pts": 1.0,
                 "commission_usd": 0.0,  "contract_size": 1,       "point": 0.01},
    "US500":    {"spread_pts": 0.5,     "slippage_pts": 0.2,
                 "commission_usd": 0.0,  "contract_size": 1,       "point": 0.01},
    "NAS":      {"spread_pts": 1.0,     "slippage_pts": 0.5,
                 "commission_usd": 0.0,  "contract_size": 1,       "point": 0.01},
    "SPX":      {"spread_pts": 0.5,     "slippage_pts": 0.2,
                 "commission_usd": 0.0,  "contract_size": 1,       "point": 0.01},

    # Oil — 1000 barrels per lot
    "USOIL":    {"spread_pts": 0.05,    "slippage_pts": 0.02,
                 "commission_usd": 2.0,  "contract_size": 1_000,   "point": 0.001},
    "OIL":      {"spread_pts": 0.05,    "slippage_pts": 0.02,
                 "commission_usd": 2.0,  "contract_size": 1_000,   "point": 0.001},
}

def _get_spec(symbol: str) -> Dict:
    """Match symbol to spec using substring matching, longest key wins."""
    s = symbol.upper().replace("..", "").replace("_", "")
    best_key, best_len = "default", 0
    for k in SYMBOL_SPECS:
        if k == "default":
            continue
        if k in s and len(k) > best_len:
            best_key, best_len = k, len(k)
    # JPY cross check
    if best_len == 0 and "JPY" in s:
        return SYMBOL_SPECS["JPY"]
    return SYMBOL_SPECS[best_key]

# Legacy shims so any external code calling _get_costs / _get_point still works
def _get_costs(symbol: str) -> Dict:
    sp = _get_spec(symbol)
    return {
        "spread_pips":        sp["spread_pts"] / sp["point"],
        "commission_per_lot": sp["commission_usd"],
        "slippage_pips":      sp["slippage_pts"] / sp["point"],
    }

def _get_point(symbol: str) -> float:
    return _get_spec(symbol)["point"]

class Backtester:
    def __init__(self):
        pass

    def test_strategy(self, strategy: Dict, market_data: pd.DataFrame,
                      symbol: str = "", lot: float = 0.01) -> Dict:
        try:
            df = market_data.copy()
            signals = self._generate_signals(strategy, df)
            if signals.sum() == 0:
                return self._empty()
            direction = 1 if strategy.get("direction","buy")=="buy" else -1
            spec = _get_spec(symbol)
            cost_pts  = spec["spread_pts"] + spec["slippage_pts"]
            comm      = spec["commission_usd"] * lot          # round-trip per trade
            cs        = spec["contract_size"] * lot           # units traded
            entry  = df["open"].shift(-1).ffill().values
            exit_  = df["close"].shift(-1).ffill().values
            sig    = signals.values
            trades = []
            for i in range(len(sig)-1):
                if sig[i]:
                    net_move = (exit_[i] - (entry[i] + cost_pts)) * direction
                    trades.append(net_move * cs - comm)
            return self._metrics(trades, strategy) if trades else self._empty()
        except Exception as e:
            logger.error(f"test_strategy: {e}")
            return self._empty()

    def walk_forward_test(self, strategy: Dict, df: pd.DataFrame,
                          symbol: str = "", n_windows: int = 5, train_pct: float = 0.7) -> Dict:
        try:
            total = len(df); wsize = total // n_windows; oos = []
            for i in range(n_windows):
                s = i*wsize; e = s+wsize; sp = int(s+(e-s)*train_pct)
                oos_df = df.iloc[sp:e]
                if len(oos_df) >= 20:
                    oos.append(self.test_strategy(strategy, oos_df, symbol))
            if not oos: return self._empty()
            return {
                "type":"walk_forward","n_windows":len(oos),
                "win_rate":float(np.mean([r["win_rate"] for r in oos])),
                "profit_factor":float(np.mean([r["profit_factor"] for r in oos])),
                "max_drawdown":float(np.max([r["max_drawdown"] for r in oos])),
                "sharpe":float(np.mean([r["sharpe"] for r in oos])),
                "score":float(np.mean([r["score"] for r in oos])),
                "profit":float(sum(r["profit"] for r in oos)),
            }
        except Exception as e:
            logger.error(f"walk_forward: {e}"); return self._empty()

    def monte_carlo(self, strategy: Dict, df: pd.DataFrame,
                    symbol: str = "", n_runs: int = 1000, lot: float = 0.01) -> Dict:
        try:
            spec = _get_spec(symbol)
            cost_pts = spec["spread_pts"] + spec["slippage_pts"]
            comm     = spec["commission_usd"] * lot
            cs       = spec["contract_size"] * lot
            d        = df.copy()
            sig      = self._generate_signals(strategy, d).values
            dir_     = 1 if strategy.get("direction","buy")=="buy" else -1
            entry    = d["open"].shift(-1).ffill().values
            exit_    = d["close"].shift(-1).ffill().values
            raw = np.array([
                ((exit_[i] - (entry[i] + cost_pts)) * dir_ * cs) - comm
                for i in range(len(sig)-1) if sig[i]
            ])
            if not len(raw): return {"error":"no_trades"}
            mdd = []
            for _ in range(n_runs):
                eq = np.cumsum(np.random.permutation(raw))
                pk = np.maximum.accumulate(eq)
                mdd.append((pk-eq).max())
            mdd = np.array(mdd)
            return {
                "type":"monte_carlo","n_runs":n_runs,
                "base_profit":float(raw.sum()),
                "max_drawdown_p50":float(np.percentile(mdd,50)),
                "max_drawdown_p95":float(np.percentile(mdd,95)),
                "max_drawdown_p99":float(np.percentile(mdd,99)),
                "win_rate":float((raw>0).mean()*100),
                "profit_factor":float(min(raw[raw>0].sum()/max(abs(raw[raw<=0].sum()),1e-6),99.99)),
            }
        except Exception as e:
            logger.error(f"monte_carlo: {e}"); return {"error":str(e)}

    def _generate_signals(self, strategy: Dict, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0.0, index=df.index); n = 0
        for rule in strategy.get("rules", []):
            # v8 FIX: rules can be dicts (from WF-OPT stub) or strings (legacy).
            # Original code did rule.lower() which crashes on dict rules.
            if isinstance(rule, dict):
                rl = " ".join(str(v) for v in rule.values()).lower()
            else:
                rl = str(rule).lower()
            try:
                if "sma20" in rl or ("sma" in rl and "20" in rl):
                    sma = df["close"].rolling(20).mean()
                    signals += (df["close"]>sma).astype(float)*0.8; n+=1
                elif "rsi" in rl:
                    rsi = df.get("rsi", self._rsi(df["close"]))
                    if "<" in rl: signals += (rsi<30).astype(float)
                    else:         signals -= (rsi>70).astype(float)
                    n+=1
                elif "macd" in rl and "macd_line" in df.columns:
                    ml,ms = df["macd_line"],df["macd_signal"]
                    if "crossover" in rl:  signals += ((ml>ms)&(ml.shift(1)<=ms.shift(1))).astype(float)*1.1
                    elif "crossunder" in rl: signals -= ((ml<ms)&(ml.shift(1)>=ms.shift(1))).astype(float)*1.1
                    n+=1
                elif "bollinger" in rl and "bb_upper" in df.columns:
                    signals += (df["close"]>df["bb_upper"]).astype(float)*0.9
                    n+=1
                elif "volume" in rl:
                    vc = "real_volume" if "real_volume" in df.columns else "tick_volume"
                    if vc in df.columns:
                        signals += (df[vc]>df[vc].rolling(20).mean()*1.5).astype(float)*0.7
                    n+=1
                elif "fair value" in rl and "fvg_bullish" in df.columns:
                    signals += df["fvg_bullish"].astype(float)*1.1; n+=1
                elif "order block" in rl and "ob_bullish" in df.columns:
                    signals += df["ob_bullish"].astype(float)*1.3; n+=1
                elif "liquidity" in rl and "liquidity_grabs" in df.columns:
                    signals += df["liquidity_grabs"].astype(float)*1.2; n+=1
                elif "crosses_above" in rl or ("high" in rl and ("20" in rl or "breakout" in rl)):
                    # ATR breakout above 20-bar high (used by WF-OPT stub)
                    h20 = df["high"].rolling(20).max().shift(1)
                    signals += ((df["close"] > h20) & (df["close"].shift(1) <= h20.shift(1))).astype(float) * 1.2
                    n += 1
                elif "crosses_below" in rl or ("low" in rl and ("20" in rl or "breakout" in rl)):
                    # ATR breakout below 20-bar low
                    l20 = df["low"].rolling(20).min().shift(1)
                    signals -= ((df["close"] < l20) & (df["close"].shift(1) >= l20.shift(1))).astype(float) * 1.2
                    n += 1
                elif "atr" in rl and "atr" in df.columns:
                    signals += (df["atr"]>df["atr"].rolling(14).mean()*1.5).astype(float)*0.8; n+=1
            except Exception as e2:
                logger.debug(f"Rule [{rl}]: {e2}")
        if n==0: return pd.Series(0, index=df.index)
        return (signals > n*0.4).astype(int)

    def _metrics(self, trades: List[float], strategy: Dict) -> Dict:
        arr = np.array(trades); wins = arr[arr>0]; loss = arr[arr<=0]
        wr  = len(wins)/len(arr)*100
        gp  = float(wins.sum()) if len(wins) else 0.0
        gl  = float(abs(loss.sum())) if len(loss) else 0.0
        # FIX: never divide by near-zero — if there are genuinely no losing trades
        # cap PF at 99.99 (extraordinary but plausible) rather than returning
        # infinity / billions from 1e-9 denominator. walk_forward averages these
        # so even one zero-loss window was producing trillion PF in the WF average.
        if gl <= 0:
            pf = 99.99 if gp > 0 else 0.0
        else:
            pf = min(gp / gl, 99.99)   # hard cap: > 99.99 is not credible
        eq  = np.cumsum(arr); pk = np.maximum.accumulate(eq)
        mdd = (pk-eq).max() if len(eq) else 0.0
        sh  = (arr.mean()/arr.std())*np.sqrt(252*6.5) if arr.std()>0 else 0.0
        pr  = float(eq[-1]) if len(eq) else 0.0
        sc  = wr*0.4 + min(pf,5)*0.4*20 + sh*0.2*10
        return {"win_rate":round(wr,2),"profit":round(pr,2),
                "profit_factor":round(pf,3),"max_drawdown":round(mdd,2),
                "sharpe":round(sh,3),"total_trades":len(arr),
                "score":round(sc,2),"equity_curve":eq.tolist()}

    def _empty(self) -> Dict:
        return {"win_rate":0,"profit":0,"profit_factor":0,"max_drawdown":0,
                "sharpe":0,"total_trades":0,"score":0,"equity_curve":[]}

    @staticmethod
    def _rsi(close: pd.Series, p:int=14) -> pd.Series:
        d=close.diff(); g=d.clip(lower=0).rolling(p).mean()
        l=(-d.clip(upper=0)).rolling(p).mean()
        return 100-(100/(1+g/l.replace(0,np.nan)))
