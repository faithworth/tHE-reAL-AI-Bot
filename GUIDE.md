# AI EA v13 — Quick Guide

## 7-Tier MTF Architecture
Signals are filtered through a full cascade before firing:

| Tier | TF  | Role |
|------|-----|------|
| 1    | D1  | Macro bias / weekly trend |
| 2    | H4  | Swing structure / key OBs |
| 3    | H3  | Intermediate BOS/CHoCH layer |
| 4    | H1  | Session entry confirmation |
| 5    | M30 | Sub-session context |
| 6    | M15 | Precision entry + liquidity sweep |
| 7    | M10 | Ultra-precision trigger + FVG |

A trade fires only when **≥4 tiers align**. The score (0–1) weights all 7 layers.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and configure env
cp .env.example .env
# Set BROKER_TYPE, MT5 credentials (or IBKR/cTrader) in .env

# 3. Train models (30 000 bars ≈ 3.4 yr H1)
BROKER_TYPE=mt5 python trainer.py --symbol XAUUSD --bars 30000 --period all

# 4. Backtest
python run_backtest.py --bars 30000

# 5. Run live
python ai_ea.py
```

---

## Training Windows (7-tier)

| Key   | Depth | Bars  | forward_bars | Purpose |
|-------|-------|-------|--------------|---------|
| 365d  | DEEP  | 30000 | 10 | Full macro cycle |
| 90d   | MACRO | 2160  | 8  | Quarterly context |
| 31d   | REGIME| 744   | 5  | Monthly options cycle |
| 14d   | STRUCT| 336   | 4  | Structural microstructure |
| 7d    | SESSION| 168  | 3  | Session swing patterns |
| 3d    | PREC  | 72    | 2  | Precision entry |
| 1d    | ULTRA | 24    | 1  | M10/M15 trigger |

Train a single window: `--period 31d`  
Train all 7 (recommended): `--period all`

---

## Key CLI Flags

```bash
# trainer.py
--symbol XAUUSD        # symbol (or --all-symbols)
--bars 30000           # H1 bars (default: 30000)
--period all           # window: all | 365d | 90d | 31d | 14d | 7d | 3d | 1d
--per-symbol           # save per-symbol .pkl
--check-retrain        # only retrain if live accuracy dropped

# run_backtest.py
--bars 30000           # default 30000
--symbols XAUUSD US100 # symbols to backtest
--lot 0.01             # lot size for P&L simulation
```

---

## Account Scale Guide

| Account | Mode | Expected edge (longer-term) |
|---------|------|-----------------------------|
| Demo    | Testing / model validation | — |
| $100    | Paper-live, 0.01 lot, observe for 1 month | low absolute $ |
| $1 000  | Live, tight risk (1% per trade) | compound slowly |
| $10 000 | Standard live, 7-tier edge fully expressed | meaningful returns |
| $50 000 | Full institutional sizing, partial scale-outs at 1R/2R/3R | optimal |

> The system is designed for **longer-term compounding** — not overnight returns.  
> The 7-tier filter deliberately reduces trade frequency in exchange for higher quality.

---

## Model Files
Saved to `models/signal_model_<SYMBOL>.pkl` after each training run.  
Delete stale models if you change `--bars` significantly — the feature distribution shifts.

---

## Broker Support
Set `BROKER_TYPE` in `.env`:  `mt5` | `ibkr` | `ctrader`

MT5 needs MetaTrader 5 running locally.  
IBKR needs TWS/Gateway on port 7497.  
cTrader needs the Open API credentials in `.env`.
