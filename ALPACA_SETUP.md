# Alpaca Setup Guide — AI EA v20

## Quick Start (5 minutes)

### 1. Create a free Alpaca account
Go to https://alpaca.markets and sign up.  
Paper trading (free) is enabled by default — no money needed to start.

### 2. Get your API keys
Dashboard → **API Keys** → Generate New Key  
Copy both the **API Key ID** and **Secret Key**.

### 3. Install dependencies
```bash
pip install alpaca-py numpy pandas scikit-learn xgboost lightgbm
```

### 4. Configure `.env`
Copy `.env.example` to `.env` and fill in:
```
BROKER_TYPE=alpaca
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_PAPER=true          # start with paper!
ALPACA_DATA_FEED=iex       # free tier
SYMBOLS=SPY,QQQ,AAPL,MSFT,NVDA,BTC/USD,ETH/USD
```

### 5. Run the EA
```bash
python ai_ea.py
```

### 6. Train the ML models (recommended before live trading)
```bash
# Train on all Alpaca symbols (7-tier deep training)
BROKER_TYPE=alpaca python trainer.py --all-symbols --period all
```

---

## Modes

| Mode | Setting | What it does |
|------|---------|-------------|
| **Paper trading** | `ALPACA_PAPER=true` | Real market data, simulated money. Free. |
| **Live trading** | `ALPACA_PAPER=false` | Real money. Fund your account first. |
| **Offline/simulation** | `ALPACA_OFFLINE=true` | No internet needed. Synthetic data. |

---

## Offline / No-Internet Mode

Set `ALPACA_OFFLINE=true` in `.env`.  
No API keys required. The EA generates synthetic market data and simulates fills.  
All ML learning, risk management, and strategy testing work fully offline.

```
BROKER_TYPE=alpaca
ALPACA_OFFLINE=true
SYMBOLS=SPY,QQQ,AAPL,BTC/USD,ETH/USD
SIM_EQUITY=10000          # starting equity for simulation
```

---

## Risk Settings for Small Accounts

The EA **automatically tunes risk to your account size** — all limits are percentages of your equity, so they work in any currency. Leave `RISK_PER_TRADE=0` in your `.env`:

| Account size    | Risk per trade | Daily loss limit | Max drawdown |
|----------------|---------------|-----------------|-------------|
| Under 500      | 1.2%          | 4.5%            | 12%         |
| 500 – 1,000    | 1.0%          | 4.0%            | 10%         |
| 1,000 – 3,000  | 0.8%          | 3.5%            | 9%          |
| 3,000 – 10,000 | 0.7%          | 3.0%            | 8%          |
| 10,000+        | 0.5–0.6%      | 2.5–2.8%        | 7–8%        |

All values are percentages — they apply equally whether your account is in USD, EUR, GBP, ZAR, or any other currency.

---

## What the EA Trades on Alpaca

- **Stocks**: SPY, QQQ, AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL
- **Crypto**: BTC/USD, ETH/USD, SOL/USD (24/7)
- ETFs and anything else in your `SYMBOLS` list

Crypto trades 24/7. Stocks trade 9:30AM–4PM ET Mon–Fri.

---

## Learning from Your Trades (v20 Feature)

The EA automatically:
1. Loads **all your Alpaca trade history** on startup
2. Learns which hours and days you win/lose most
3. Blocks trades during historically bad hours
4. Gets smarter after every trade (win or loss)

No setup needed — it runs automatically in background.

---

## Sharing the Bot with a Friend

1. Give them this entire folder
2. They create their own Alpaca account (free)
3. They fill in their own `.env` with their API keys
4. Each person's learning data is stored separately in `data/` and `models/`

**Never share your `.env` file — it contains your private API keys.**
