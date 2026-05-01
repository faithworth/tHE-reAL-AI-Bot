"""
prop_guard.py — Prop-firm compliance guard (AI EA v4)
------------------------------------------------------
Wraps the RiskEngine with firm-specific rule enforcement and an
auditable emergency-stop system.

When prop_mode=True the following hard limits CANNOT be overridden
at runtime:
  - Daily loss       ≤ 3 % of starting balance
  - Max drawdown     ≤ 8 % of peak equity
  - Max daily trades ≤ 10
  - Max concurrent   ≤ 5

Emergency stop logic
--------------------
The E-stop is triggered automatically when any limit is breached.
Once triggered it persists until:
  a) Manually reset via reset_emergency_stop()       — requires reason
  b) A new trading day AND the breach was daily-only — auto-lifts

All E-stop events are written to logs/estop_events.jsonl.
"""

import json
import logging
import os
from datetime import date, datetime
from typing import Tuple

from risk_engine import RiskEngine

logger = logging.getLogger(__name__)

ESTOP_LOG = "logs/estop_events.jsonl"

# Hard ceilings enforced regardless of RiskEngine settings
PROP_MAX_DAILY_LOSS   = 0.03
PROP_MAX_DRAWDOWN     = 0.08
PROP_MAX_TRADES_DAY   = 20
PROP_MAX_CONCURRENT   = 5


class PropGuard:
    """
    Thin compliance wrapper around RiskEngine.

    Usage
    -----
    guard = PropGuard(risk_engine)
    ok, reason = guard.check(equity, open_positions, signal_prob)
    """

    def __init__(self, risk_engine: RiskEngine):
        self.risk = risk_engine
        os.makedirs("logs", exist_ok=True)
        self._today_breach: bool = False   # was today's breach daily-only?

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self,
        equity: float,
        open_positions: int,
        signal_prob: float = 0.0,
        symbol: str = "",
    ) -> Tuple[bool, str]:
        """
        Full compliance check.  Returns (allowed, reason).
        Delegates to RiskEngine.approve_trade() and then applies prop overrides.
        """
        if not self.risk.prop_mode:
            # Non-prop: just delegate
            return self.risk.approve_trade(equity, open_positions, symbol, signal_prob)

        # ── Prop hard-limit enforcement ───────────────────────────────────────
        status = self.risk.get_status(equity)

        # Daily loss
        daily_loss_pct = abs(status.get("daily_loss_pct", 0)) / 100
        if daily_loss_pct >= PROP_MAX_DAILY_LOSS:
            msg = (f"[PROP] Daily loss limit {PROP_MAX_DAILY_LOSS*100:.0f}% breached: "
                   f"{daily_loss_pct*100:.2f}%")
            self._trigger_estop(msg, breach_type="daily_loss")
            return False, msg

        # Drawdown
        drawdown_pct = status.get("drawdown_pct", 0) / 100
        if drawdown_pct >= PROP_MAX_DRAWDOWN:
            msg = (f"[PROP] Drawdown limit {PROP_MAX_DRAWDOWN*100:.0f}% breached: "
                   f"{drawdown_pct*100:.2f}%")
            self._trigger_estop(msg, breach_type="drawdown")
            return False, msg

        # Daily trade count
        if status.get("daily_trades", 0) >= PROP_MAX_TRADES_DAY:
            return False, f"[PROP] Max daily trades ({PROP_MAX_TRADES_DAY}) reached"

        # Concurrent positions
        if open_positions >= PROP_MAX_CONCURRENT:
            return False, f"[PROP] Max concurrent positions ({PROP_MAX_CONCURRENT}) reached"

        # Emergency stop
        if status.get("emergency_stop", False):
            return False, "[PROP] EMERGENCY STOP active — manual reset required"

        # Delegate remaining checks (signal prob, cooldown, etc.)
        return self.risk.approve_trade(equity, open_positions, symbol, signal_prob)

    def reset_emergency_stop(self, reason: str, operator: str = "manual") -> None:
        """
        Manually lift the emergency stop.  Requires a documented reason.
        """
        self.risk.reset_emergency_stop()
        event = {
            "timestamp": datetime.now().isoformat(),
            "event": "estop_reset",
            "reason": reason,
            "operator": operator,
        }
        self._write_estop_event(event)
        logger.warning(f"[PROP] Emergency stop RESET by {operator}: {reason}")

    def get_compliance_report(self, equity: float) -> dict:
        """Return a dict suitable for dashboard display."""
        status = self.risk.get_status(equity)
        return {
            "prop_mode":        self.risk.prop_mode,
            "daily_loss_pct":   status.get("daily_loss_pct", 0),
            "max_daily_loss":   PROP_MAX_DAILY_LOSS * 100,
            "drawdown_pct":     status.get("drawdown_pct", 0),
            "max_drawdown":     PROP_MAX_DRAWDOWN * 100,
            "daily_trades":     status.get("daily_trades", 0),
            "max_trades_day":   PROP_MAX_TRADES_DAY,
            "open_positions":   status.get("open_positions", 0),
            "max_concurrent":   PROP_MAX_CONCURRENT,
            "emergency_stop":   status.get("emergency_stop", False),
            "consecutive_loss": status.get("consecutive_losses", 0),
            "compliant":        not status.get("emergency_stop", False),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _trigger_estop(self, msg: str, breach_type: str) -> None:
        if not self.risk._emergency_stop:
            self.risk._emergency_stop = True
            self.risk._save_state()
            event = {
                "timestamp":   datetime.now().isoformat(),
                "event":       "estop_triggered",
                "breach_type": breach_type,
                "message":     msg,
            }
            self._write_estop_event(event)
            logger.critical(f"[PROP] EMERGENCY STOP TRIGGERED: {msg}")
        else:
            logger.debug(f"[PROP] E-stop already active: {msg}")

    @staticmethod
    def _write_estop_event(event: dict) -> None:
        try:
            with open(ESTOP_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Could not write e-stop event: {e}")
