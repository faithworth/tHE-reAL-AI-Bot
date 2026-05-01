"""
risk_engine.py
--------------
Institutional-grade risk management engine.

Responsibilities
----------------
- ATR-based position sizing
- Per-trade risk (0.5% – 1% of equity)
- Daily loss limit (3% equity)
- Maximum drawdown guard (8% equity)
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
# Default limits — overridden at runtime via RiskEngine constructor
# -----------------------------------------------------------------------
DEFAULT_RISK_PER_TRADE   = 0.007   # 0.7 % of equity
DEFAULT_MAX_DAILY_LOSS   = 0.03    # 5  % of equity
DEFAULT_MAX_DRAWDOWN     = 0.08    # 10  % of equity
DEFAULT_MAX_TRADES_DAY   = 10
DEFAULT_MAX_CONCURRENT   = 5
DEFAULT_ATR_MULTIPLIER   = 1.5     # lot = risk_amount / (ATR * multiplier)
DEFAULT_COOLDOWN_SECONDS = 900     # 15-min cooldown after 2+ consecutive losses

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
        self.risk_per_trade = risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.max_trades_day = max_trades_day
        self.max_concurrent = max_concurrent
        self.atr_multiplier = atr_multiplier
        self.cooldown_seconds = cooldown_seconds
        self.prop_mode = prop_mode

        # Runtime state
        self._today: date = date.today()
        self._daily_trades: int = 0
        self._daily_pnl: float = 0.0
        self._peak_equity: float = 0.0
        self._start_equity: float = 0.0
        self._consecutive_losses: int = 0
        self._last_loss_time: Optional[datetime] = None
        self._emergency_stop: bool = False
        self._open_positions: int = 0

        # v8: Portfolio correlation tracking
        # Stores {symbol: lot_size} for all currently open positions.
        # Used to cap total correlated-group exposure.
        self._open_lots: Dict[str, float] = {}
        # Hard-coded high-correlation groups (extend via CORR_GROUPS env var JSON)
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
        ] + _extra
        # Maximum fraction of equity that a single correlated group may risk
        # Default 6%: allows 1 trade per group on small accounts; set lower in .env for stricter control
        self.max_group_risk_pct: float = float(os.getenv("MAX_GROUP_RISK_PCT", "0.06"))

        os.makedirs(os.path.dirname(RISK_STATE_PATH), exist_ok=True)
        self._load_state()

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

        if self._emergency_stop:
            return False, "EMERGENCY_STOP active"

        # Signal quality guard — v8 FIX: lowered from 0.65 to 0.35.
        # In a 3-class calibrated model, raw prob of 0.65 is extremely high
        # (implies model is almost certain). Real edge appears at 0.36+.
        # Primary quality filtering is done upstream in ai_ea.py via composite score.
        if signal_prob > 0 and signal_prob < 0.35:
            return False, f"Signal probability too low: {signal_prob:.3f}"

        # Concurrent positions
        if open_positions >= self.max_concurrent:
            return False, f"Max concurrent positions ({self.max_concurrent}) reached"

        # Daily trade count
        if self._daily_trades >= self.max_trades_day:
            return False, f"Daily trade limit ({self.max_trades_day}) reached"

        # Daily loss limit
        daily_loss_pct = self._daily_pnl / self._start_equity if self._start_equity > 0 else 0
        if daily_loss_pct <= -self.max_daily_loss:
            return False, f"Daily loss limit ({self.max_daily_loss*100:.1f}%) breached: {daily_loss_pct*100:.2f}%"

        # Drawdown limit
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

        # Cooldown after consecutive losses
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
    ) -> float:
        """
        ATR-based position sizing.
        lot = (equity * risk_per_trade) / (ATR * atr_multiplier * contract_size * point_value)

        For simplicity we treat monetary risk directly:
          risk_dollars = equity * risk_per_trade
          stop_distance = atr * atr_multiplier  (in price)
          lot = risk_dollars / (stop_distance / symbol_point * point_value)

        Falls back to min_lot on any calculation error.
        """
        try:
            if equity <= 0 or atr <= 0:
                return min_lot

            risk_amount   = equity * self.risk_per_trade
            stop_distance = atr * self.atr_multiplier   # in price units

            # Dollar value of 1 price-unit move per lot:
            #   forex standard:  point=0.00001, cs=100000 → $1 per pip ($0.10 per point)
            #   gold (XAUUSD):   point=0.01,    cs=100    → $1 per point
            #   BTC:             point=0.01,    cs=1      → $0.01 per point
            #   US30 index:      point=0.01,    cs=1      → $0.01 per point
            # dollar_per_price_unit = contract_size * symbol_point  ... NO — that's
            # pip-value, not price-value.  Correct formula:
            #   dollar_per_price_unit = contract_size   (1 USD per price $ per lot for BTC)
            #   BUT for forex: price move of 0.0001 = 1 pip; cs=100000 → $10/pip = $10/0.0001 = $100k/unit
            # Unified: dollar at risk = stop_distance * contract_size * lot
            # → lot = risk_amount / (stop_distance * contract_size)
            # This is always correct regardless of asset class.
            dollar_per_unit = contract_size   # $/price-unit/lot
            if dollar_per_unit <= 0 or stop_distance <= 0:
                return min_lot

            lot = risk_amount / (stop_distance * dollar_per_unit)
            lot = round(max(min_lot, min(lot, max_lot)), 2)
            logger.debug(
                f"Lot sizing: equity={equity:.2f}, risk={risk_amount:.2f}, "
                f"ATR={atr:.5f}, stop={stop_distance:.5f}, cs={dollar_per_unit} → lot={lot}"
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
        """Called after a trade closes. Updates daily P&L and loss streak."""
        self._daily_pnl += pnl
        self._open_positions = max(0, self._open_positions - 1)

        if pnl < 0:
            self._consecutive_losses += 1
            self._last_loss_time = datetime.now()
            logger.info(f"Loss recorded. Consecutive losses: {self._consecutive_losses}")
        else:
            self._consecutive_losses = 0
            self._last_loss_time = None

        # Update peak equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Prop-mode: check daily loss after every close
        if self.prop_mode:
            daily_loss_pct = self._daily_pnl / self._start_equity if self._start_equity > 0 else 0
            if daily_loss_pct <= -self.max_daily_loss:
                logger.warning(
                    f"[PROP MODE] Daily loss limit hit: {daily_loss_pct*100:.2f}%. "
                    "No more trades today."
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
            "emergency_stop": self._emergency_stop,
            "daily_trades": self._daily_trades,
            "max_trades_day": self.max_trades_day,
            "daily_pnl": round(self._daily_pnl, 2),
            "daily_loss_pct": round(daily_loss_pct * 100, 2),
            "max_daily_loss_pct": round(self.max_daily_loss * 100, 2),
            "drawdown_pct": round(drawdown * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "consecutive_losses": self._consecutive_losses,
            "prop_mode": self.prop_mode,
            "open_positions": self._open_positions,
            # FIX: equity_baseline exposed so ai_ea.py perf_monitor can read it
            "equity_baseline": round(self._start_equity, 2),
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
            "today": str(self._today),
            "daily_trades": self._daily_trades,
            "daily_pnl": self._daily_pnl,
            "peak_equity": self._peak_equity,
            "start_equity": self._start_equity,
            "consecutive_losses": self._consecutive_losses,
            "last_loss_time": self._last_loss_time.isoformat() if self._last_loss_time else None,
            "emergency_stop": self._emergency_stop,
            "open_positions": self._open_positions,
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
                self._daily_trades = state.get("daily_trades", 0)
                self._daily_pnl = state.get("daily_pnl", 0.0)
                self._consecutive_losses = state.get("consecutive_losses", 0)
                llt = state.get("last_loss_time")
                self._last_loss_time = datetime.fromisoformat(llt) if llt else None
                self._emergency_stop = state.get("emergency_stop", False)
                self._open_positions = state.get("open_positions", 0)
            self._peak_equity = state.get("peak_equity", 0.0)
            self._start_equity = state.get("start_equity", 0.0)
            logger.info("Risk state loaded from disk.")
        except Exception as e:
            logger.warning(f"Could not load risk state: {e}")
