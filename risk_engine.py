"""
risk_engine.py
--------------
Institutional-grade risk management engine.  (AI EA v20)

v20 UPGRADES
------------
1. Account-size-aware risk tiers — risk % and daily limits auto-scale to
   actual equity so a $500 account and a $50 000 account both feel right:
     < $1 000  → nano  (1.0% risk, 4% daily loss, 10% DD)
     < $5 000  → micro (0.8% risk, 3.5% daily loss, 9% DD)
     < $20 000 → small (0.7% risk, 3.0% daily loss, 8% DD)
     ≥ $20 000 → std   (0.5% risk, 2.5% daily loss, 7% DD)
   Env-var overrides still work; auto-tier only fires when the var is absent.

2. Alpaca stock lot-sizing — stocks are integer-share, fractional for crypto.
   calculate_lot_size() now accepts asset_class param; for stocks it returns
   whole-share counts and uses share-price-based dollar risk directly.

3. Win-streak position scaling — after 3+ consecutive wins, risk scales up
   to 1.25× (capped). After losses, it scales down to 0.75×. Smooth Kelly-
   inspired adjustment without full Kelly volatility.

4. Daily PnL target lock-in — once daily profit ≥ 2× max_daily_loss amount,
   the engine enters "protect-profits" mode and reduces max risk by 50%.

Responsibilities
----------------
- ATR-based position sizing
- Per-trade risk (auto-tiered or env-configured)
- Daily loss limit (auto-tiered or env-configured)
- Maximum drawdown guard (auto-tiered or env-configured)
- Max concurrent positions (5)
- Max trades per day (10)
- Trade cooldown after losses
- Prop-firm compliance mode
- Emergency stop
"""

import logging
import os
import json
import numpy as np
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Tuple, List

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Risk tiers — auto-selected based on equity unless env vars override
# v20: More granular tiers so a $500 account gets sensible real-$ limits
# -----------------------------------------------------------------------
_RISK_TIERS = [
    # (equity_floor, risk_per_trade, max_daily_loss, max_drawdown, max_concurrent, max_trades)
    # nano accounts: protect capital, learn slowly
    (0,       0.012, 0.045, 0.12, 3,  8),   # nano    < $500   : 1.2% risk, 4.5% daily limit
    (500,     0.010, 0.040, 0.10, 3,  8),   # nano+   < $1 000 : 1.0% risk, 4.0% daily limit
    (1000,    0.008, 0.035, 0.09, 4,  10),  # micro   < $3 000 : 0.8% risk, 3.5% daily limit
    (3000,    0.007, 0.030, 0.08, 4,  10),  # small   < $10 000: 0.7% risk, 3.0% daily limit
    (10000,   0.006, 0.028, 0.08, 5,  12),  # medium  < $30 000: 0.6% risk, 2.8% daily limit
    (30000,   0.005, 0.025, 0.07, 6,  15),  # std     < $100 K : 0.5% risk, 2.5% daily limit
    (100000,  0.004, 0.020, 0.06, 8,  20),  # pro     ≥ $100 K : 0.4% risk, 2.0% daily limit
]

def _auto_tier(equity: float) -> Tuple[float, float, float, int, int]:
    """Return (risk_per_trade, max_daily_loss, max_drawdown, max_concurrent, max_trades_day)."""
    tier = _RISK_TIERS[0][1:]
    for row in _RISK_TIERS:
        if equity >= row[0]:
            tier = row[1:]
    rpt, mdl, mdd, mx_conc, mx_trades = tier
    return rpt, mdl, mdd, int(mx_conc), int(mx_trades)



# -----------------------------------------------------------------------
# Default limits — overridden at runtime via RiskEngine constructor or env
# -----------------------------------------------------------------------
DEFAULT_RISK_PER_TRADE   = float(os.getenv("RISK_PER_TRADE",   "0"))   # 0 = auto-tier
DEFAULT_MAX_DAILY_LOSS   = float(os.getenv("MAX_DAILY_LOSS",   "0"))   # 0 = auto-tier
DEFAULT_MAX_DRAWDOWN     = float(os.getenv("MAX_DRAWDOWN",     "0"))   # 0 = auto-tier
DEFAULT_MAX_TRADES_DAY   = 10
DEFAULT_MAX_CONCURRENT   = 5
DEFAULT_ATR_MULTIPLIER   = 1.5
DEFAULT_COOLDOWN_SECONDS = 900     # 15-min cooldown after 2+ consecutive losses

# Win/loss streak scaling factors
_WIN_SCALE_MAX   = 1.25   # scale up to 25% more risk after a win streak
_LOSS_SCALE_MIN  = 0.75   # scale down to 75% risk after losses
_STREAK_TRIGGER  = 3      # consecutive wins needed to start scaling up

# Protect-profits: if day profit ≥ this multiple of max_daily_loss, reduce risk
_PROTECT_PROFIT_MULT = 2.0
_PROTECT_RISK_SCALE  = 0.50

RISK_STATE_PATH = "data/risk_state.json"


class RiskEngine:
    """
    Central risk manager for the AI EA.
    All trading decisions must pass through `approve_trade()`.
    """

    def __init__(
        self,
        risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
        max_daily_loss: float = DEFAULT_MAX_DAILY_LOSS,
        max_drawdown: float = DEFAULT_MAX_DRAWDOWN,
        max_trades_day: int = DEFAULT_MAX_TRADES_DAY,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        prop_mode: bool = True,
    ):
        # Store configured values (0 = use auto-tier)
        self._cfg_risk_per_trade = risk_per_trade
        self._cfg_max_daily_loss = max_daily_loss
        self._cfg_max_drawdown   = max_drawdown

        # These are active values (updated by _apply_tier)
        self.risk_per_trade = risk_per_trade if risk_per_trade > 0 else 0.007
        self.max_daily_loss = max_daily_loss if max_daily_loss > 0 else 0.03
        self.max_drawdown   = max_drawdown   if max_drawdown   > 0 else 0.08

        self.max_trades_day    = max_trades_day
        self.max_concurrent    = max_concurrent
        self.atr_multiplier    = atr_multiplier
        self.cooldown_seconds  = cooldown_seconds
        self.prop_mode         = prop_mode

        # Runtime state
        self._today: date = date.today()
        self._daily_trades: int = 0
        self._daily_pnl: float = 0.0
        self._peak_equity: float = 0.0
        self._start_equity: float = 0.0
        self._consecutive_losses: int = 0
        self._consecutive_wins: int = 0    # v20: win-streak tracking
        self._last_loss_time: Optional[datetime] = None
        self._emergency_stop: bool = False
        self._open_positions: int = 0
        self._protect_profits_mode: bool = False   # v20: lock-in gains

        # v8: Portfolio correlation tracking
        self._open_lots: Dict[str, float] = {}
        _env_groups = os.getenv("CORR_GROUPS", "")
        try:
            _extra: List[List[str]] = json.loads(_env_groups) if _env_groups else []
        except Exception:
            _extra = []
        self._corr_groups: List[List[str]] = [
            ["XAUUSD", "XAGUSD", "XPTUSD"],          # precious metals
            ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"], # USD longs
            ["USDCHF", "USDJPY", "USDCAD"],           # USD shorts
            ["BTCUSD", "ETHUSD", "LTCUSD"],           # crypto
            ["US30", "US500", "NAS100"],               # US indices
            # Alpaca stock groups
            ["SPY", "QQQ", "IWM"],                    # broad ETFs
            ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],# mega-cap tech
            ["BTC/USD", "ETH/USD", "SOL/USD"],         # Alpaca crypto
        ] + _extra
        self.max_group_risk_pct: float = float(os.getenv("MAX_GROUP_RISK_PCT", "0.06"))

        os.makedirs(os.path.dirname(RISK_STATE_PATH), exist_ok=True)
        self._load_state()

    # ------------------------------------------------------------------
    # v20: Account-size auto-tiering
    # ------------------------------------------------------------------

    def _apply_tier(self, equity: float) -> None:
        """Update risk parameters based on current equity tier (if not env-overridden).  v20."""
        if equity <= 0:
            return
        auto_rpt, auto_mdl, auto_mdd, auto_conc, auto_trades = _auto_tier(equity)
        if self._cfg_risk_per_trade <= 0:
            self.risk_per_trade = auto_rpt
        if self._cfg_max_daily_loss <= 0:
            self.max_daily_loss = auto_mdl
        if self._cfg_max_drawdown <= 0:
            self.max_drawdown = auto_mdd
        # Auto-tier max_concurrent and max_trades_day (unless user set env vars)
        if not os.getenv("MAX_CONCURRENT"):
            self.max_concurrent = auto_conc
        if not os.getenv("MAX_TRADES_DAY"):
            self.max_trades_day = auto_trades

        logger.debug(
            f"[RiskTier] equity={equity:.0f} → risk={self.risk_per_trade*100:.1f}% "
            f"| daily_limit={self.max_daily_loss*100:.1f}% "
            f"| max_conc={self.max_concurrent} | max_trades={self.max_trades_day}"
        )

    def _streak_scale(self) -> float:
        """v20: Return a 0.75–1.25 multiplier based on recent win/loss streak."""
        if self._consecutive_wins >= _STREAK_TRIGGER:
            extra = min(self._consecutive_wins - _STREAK_TRIGGER, 3)
            return min(_WIN_SCALE_MAX, 1.0 + extra * 0.083)   # +8.3% per extra win, cap 1.25
        if self._consecutive_losses >= 2:
            return max(_LOSS_SCALE_MIN, 1.0 - self._consecutive_losses * 0.083)
        return 1.0

    def _effective_risk(self) -> float:
        """v20: risk_per_trade adjusted for streak and protect-profits mode."""
        base = self.risk_per_trade * self._streak_scale()
        if self._protect_profits_mode:
            base *= _PROTECT_RISK_SCALE
        return base

    # ------------------------------------------------------------------
    # Core gate
    # ------------------------------------------------------------------

    def approve_trade(
        self,
        equity: float,
        open_positions: int,
        symbol: str = "",
        signal_prob: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Main gate: returns (approved, reason).
        Call before every trade execution.
        """
        self._refresh_day(equity)
        self._apply_tier(equity)   # v20: update tier with live equity

        if self._emergency_stop:
            return False, "EMERGENCY_STOP active"

        if signal_prob > 0 and signal_prob < 0.35:
            return False, f"Signal probability too low: {signal_prob:.3f}"

        if open_positions >= self.max_concurrent:
            return False, f"Max concurrent positions ({self.max_concurrent}) reached"

        if self._daily_trades >= self.max_trades_day:
            return False, f"Daily trade limit ({self.max_trades_day}) reached"

        daily_loss_pct = self._daily_pnl / self._start_equity if self._start_equity > 0 else 0
        if daily_loss_pct <= -self.max_daily_loss:
            return False, f"Daily loss limit ({self.max_daily_loss*100:.1f}%) breached: {daily_loss_pct*100:.2f}%"

        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown >= self.max_drawdown:
                self._emergency_stop = True
                logger.critical(
                    f"MAX DRAWDOWN {self.max_drawdown*100:.1f}% BREACHED — "
                    f"Emergency stop engaged! Drawdown={drawdown*100:.2f}%"
                )
                self._save_state()
                return False, f"MAX DRAWDOWN ({self.max_drawdown*100:.1f}%) BREACHED — Emergency stop"

        if self._consecutive_losses >= 2 and self._last_loss_time is not None:
            elapsed = (datetime.now() - self._last_loss_time).total_seconds()
            if elapsed < self.cooldown_seconds:
                remaining = int(self.cooldown_seconds - elapsed)
                return False, f"Loss cooldown active: {remaining}s remaining"

        return True, "OK"

    # ------------------------------------------------------------------
    # v8: Correlated-group exposure gate
    # ------------------------------------------------------------------

    def _clean_symbol(self, symbol: str) -> str:
        """Strip broker suffixes (m, .r, .c etc.) for group lookup."""
        import re
        return re.sub(r"[^A-Z]", "", symbol.upper())

    def _get_corr_group(self, symbol: str) -> Optional[List[str]]:
        """Return the correlation group containing symbol, or None."""
        clean = self._clean_symbol(symbol)
        for group in self._corr_groups:
            if any(clean == self._clean_symbol(g) for g in group):
                return group
        return None

    def approve_correlated_trade(
        self,
        symbol: str,
        proposed_lot: float,
        equity: float,
        atr: float,
        symbol_point: float = 0.00001,
        contract_size: float = 100_000,
    ) -> Tuple[bool, str]:
        """
        v8 portfolio-level gate: ensures the total dollar-risk of all
        open positions in the same correlated group (e.g. XAUUSD + XAGUSD)
        does not exceed max_group_risk_pct of equity.

        Call this AFTER approve_trade() and BEFORE place_order().

        Parameters
        ----------
        proposed_lot   : lot size about to be placed
        atr            : current ATR of the symbol (price units)
        symbol_point   : broker point size for this symbol
        contract_size  : units per lot

        Returns (approved, reason)
        """
        group = self._get_corr_group(symbol)
        if group is None or equity <= 0:
            return True, "OK (no group)"

        max_group_risk_dollars = equity * self.max_group_risk_pct

        # Estimate risk in dollars for all open positions in the same group
        # We store lots; risk ≈ lot * ATR * SL_MULT * contract_size * point_value
        # For simplicity we use 1.5×ATR as the assumed SL distance.
        pip_value_per_lot = symbol_point * contract_size
        sl_distance_pips  = (atr * 1.5) / symbol_point if symbol_point > 0 else 0

        group_risk = 0.0
        clean_group = [self._clean_symbol(g) for g in group]
        for open_sym, lot in self._open_lots.items():
            if self._clean_symbol(open_sym) in clean_group:
                group_risk += lot * sl_distance_pips * pip_value_per_lot

        proposed_risk = proposed_lot * sl_distance_pips * pip_value_per_lot

        # ── v19 FIX: ensure cap is never smaller than the proposed trade itself.
        # On small accounts (<$1000) a 3% cap can be less than a single min-lot
        # trade's risk, permanently blocking the entire correlated group.
        # Floor = max(configured %, 1.5× proposed_risk) so at least one trade
        # always fits, and we only block when a SECOND correlated trade would
        # push the group over the cap.
        min_floor = proposed_risk * 1.5
        max_group_risk_dollars = max(max_group_risk_dollars, min_floor)

        total_risk = group_risk + proposed_risk

        if total_risk > max_group_risk_dollars:
            return (
                False,
                f"Correlated group risk cap: group={[self._clean_symbol(g) for g in group]} "
                f"current=${group_risk:.0f} proposed=${proposed_risk:.0f} "
                f"max=${max_group_risk_dollars:.0f} ({self.max_group_risk_pct*100:.1f}%)",
            )
        return True, "OK"

    def record_open_lot(self, symbol: str, lot: float) -> None:
        """Track an opened position for correlated-group accounting."""
        self._open_lots[symbol] = self._open_lots.get(symbol, 0.0) + lot

    def record_close_lot(self, symbol: str, lot: float) -> None:
        """Remove a closed position from correlated-group accounting."""
        remaining = self._open_lots.get(symbol, 0.0) - lot
        if remaining <= 0.0:
            self._open_lots.pop(symbol, None)
        else:
            self._open_lots[symbol] = remaining

    def calculate_lot_size(
        self,
        equity: float,
        atr: float,
        min_lot: float = 0.01,
        max_lot: float = 0.50,
        symbol_point: float = 0.00001,
        contract_size: float = 100_000,
        asset_class: str = "forex",
        share_price: float = 0.0,
    ) -> float:
        """
        ATR-based position sizing.  (v20: supports stocks + streak scaling)

        For forex/crypto/futures:
          lot = (equity * effective_risk) / (ATR * atr_multiplier * contract_size)

        For stocks (asset_class="stock"):
          shares = (equity * effective_risk) / (ATR * atr_multiplier)
          Returns whole-share count; min 1 share.

        Falls back to min_lot on any calculation error.
        """
        try:
            if equity <= 0 or atr <= 0:
                return min_lot

            self._apply_tier(equity)
            risk_amount   = equity * self._effective_risk()
            stop_distance = atr * self.atr_multiplier   # in price units

            if asset_class in ("stock", "equity"):
                # For stocks: risk_amount / stop_distance = number of shares
                price = share_price if share_price > 0 else (atr * 10)   # rough fallback
                shares = risk_amount / stop_distance
                shares = max(1.0, round(shares))
                shares = min(shares, max_lot)
                logger.debug(
                    f"Stock sizing: equity={equity:.2f}, risk={risk_amount:.2f}, "
                    f"ATR={atr:.4f}, stop={stop_distance:.4f} → {shares:.0f} shares"
                )
                return shares

            # Forex / crypto / futures
            dollar_per_unit = contract_size
            if dollar_per_unit <= 0 or stop_distance <= 0:
                return min_lot

            lot = risk_amount / (stop_distance * dollar_per_unit)
            lot = round(max(min_lot, min(lot, max_lot)), 2)
            logger.debug(
                f"Lot sizing: equity={equity:.2f}, risk={risk_amount:.2f}, "
                f"ATR={atr:.5f}, stop={stop_distance:.5f}, cs={dollar_per_unit} → lot={lot} "
                f"(streak_scale={self._streak_scale():.2f})"
            )
            return lot

        except Exception as e:
            logger.error(f"Lot size calculation error: {e}")
            return min_lot

    # ------------------------------------------------------------------
    # Trade result feedback
    # ------------------------------------------------------------------

    def record_trade_open(self) -> None:
        self._daily_trades += 1
        self._open_positions += 1
        self._save_state()

    def record_trade_close(self, pnl: float, equity: float) -> None:
        """Called after a trade closes. Updates daily P&L, streaks, and protect-profits mode."""
        self._daily_pnl += pnl
        self._open_positions = max(0, self._open_positions - 1)

        if pnl < 0:
            self._consecutive_losses += 1
            self._consecutive_wins = 0        # v20: reset win streak on any loss
            self._last_loss_time = datetime.now()
            logger.info(f"Loss recorded ${pnl:.2f}. Consecutive losses: {self._consecutive_losses} | streak_scale={self._streak_scale():.2f}")
        else:
            self._consecutive_losses = 0
            self._consecutive_wins += 1       # v20: track win streak
            self._last_loss_time = None
            logger.info(f"Win recorded ${pnl:.2f}. Consecutive wins: {self._consecutive_wins} | streak_scale={self._streak_scale():.2f}")

        # Update peak equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        # v20: Protect-profits — once daily profit ≥ 2× daily loss limit, reduce risk 50%
        if self._start_equity > 0:
            daily_profit_pct = self._daily_pnl / self._start_equity
            protect_threshold = self.max_daily_loss * _PROTECT_PROFIT_MULT
            if daily_profit_pct >= protect_threshold and not self._protect_profits_mode:
                self._protect_profits_mode = True
                logger.info(
                    f"[RiskEngine] PROTECT-PROFITS mode ON — daily P&L={self._daily_pnl:.2f} "
                    f"({daily_profit_pct*100:.1f}%) >= {protect_threshold*100:.1f}% threshold. Risk halved."
                )

        # Prop-mode: log warning when daily loss limit is hit
        if self.prop_mode:
            daily_loss_pct = self._daily_pnl / self._start_equity if self._start_equity > 0 else 0
            if daily_loss_pct <= -self.max_daily_loss:
                logger.warning(
                    f"[PROP MODE] Daily loss limit hit: {daily_loss_pct*100:.2f}% "
                    f"(pnl={self._daily_pnl:.2f}). No more trades today."
                )

        self._save_state()

    def set_equity_baseline(self, equity: float) -> None:
        """Call once at session start with current account equity."""
        if self._start_equity == 0:
            self._start_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity
        self._save_state()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self, equity: float) -> Dict:
        self._refresh_day(equity)
        drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0
        daily_loss_pct = self._daily_pnl / self._start_equity if self._start_equity > 0 else 0
        return {
            "emergency_stop":       self._emergency_stop,
            "daily_trades":         self._daily_trades,
            "max_trades_day":       self.max_trades_day,
            "daily_pnl":            round(self._daily_pnl, 2),
            "daily_loss_pct":       round(daily_loss_pct * 100, 2),
            "max_daily_loss_pct":   round(self.max_daily_loss * 100, 2),
            "drawdown_pct":         round(drawdown * 100, 2),
            "max_drawdown_pct":     round(self.max_drawdown * 100, 2),
            "consecutive_losses":   self._consecutive_losses,
            "consecutive_wins":     self._consecutive_wins,
            "streak_scale":         round(self._streak_scale(), 3),
            "prop_mode":            self.prop_mode,
            "open_positions":        self._open_positions,
            "equity_baseline":       round(self._start_equity, 2),
            "protect_profits_mode":  self._protect_profits_mode,
        }

    def reset_emergency_stop(self) -> None:
        """Manual override — use only after reviewing positions."""
        logger.warning("Emergency stop manually reset.")
        self._emergency_stop = False
        self._save_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_day(self, equity: float) -> None:
        """Reset daily counters when the calendar day rolls over."""
        today = date.today()
        if today != self._today:
            logger.info(f"New trading day: {today}. Resetting daily risk counters.")
            self._today = today
            self._daily_trades = 0
            self._daily_pnl = 0.0
            self._start_equity = equity
            self._consecutive_losses = 0
            self._last_loss_time = None
            # Do NOT reset _emergency_stop on day roll — require manual override
            self._save_state()

    def _save_state(self) -> None:
        state = {
            "today":               str(self._today),
            "daily_trades":        self._daily_trades,
            "daily_pnl":           self._daily_pnl,
            "peak_equity":         self._peak_equity,
            "start_equity":        self._start_equity,
            "consecutive_losses":  self._consecutive_losses,
            "consecutive_wins":    self._consecutive_wins,       # v20
            "protect_profits_mode":self._protect_profits_mode,   # v20
            "last_loss_time":      self._last_loss_time.isoformat() if self._last_loss_time else None,
            "emergency_stop":      self._emergency_stop,
            "open_positions":      self._open_positions,
        }
        try:
            with open(RISK_STATE_PATH, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save risk state: {e}")

    def _load_state(self) -> None:
        if not os.path.exists(RISK_STATE_PATH):
            return
        try:
            with open(RISK_STATE_PATH, "r") as f:
                state = json.load(f)
            # Only restore if same calendar day
            if state.get("today") == str(date.today()):
                self._daily_trades        = state.get("daily_trades", 0)
                self._daily_pnl           = state.get("daily_pnl", 0.0)
                self._consecutive_losses  = state.get("consecutive_losses", 0)
                self._consecutive_wins    = state.get("consecutive_wins", 0)    # v20
                self._protect_profits_mode= state.get("protect_profits_mode", False)  # v20
                llt = state.get("last_loss_time")
                self._last_loss_time = datetime.fromisoformat(llt) if llt else None
                self._emergency_stop = state.get("emergency_stop", False)
                self._open_positions = state.get("open_positions", 0)
            self._peak_equity  = state.get("peak_equity", 0.0)
            self._start_equity = state.get("start_equity", 0.0)
            logger.info("Risk state loaded from disk.")
        except Exception as e:
            logger.warning(f"Could not load risk state: {e}")
