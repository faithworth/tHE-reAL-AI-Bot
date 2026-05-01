"""
secure_config.py — Environment-variable based secure configuration (AI EA v17)
------------------------------------------------------------------------------
Replaces hardcoded MT5 credentials with a layered config system:

Priority (highest → lowest)
  1. Environment variables  (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, …)
  2. .env file              (project root, git-ignored)
  3. Encrypted config       (config/config.enc via config_loader.py)
  4. Safe defaults          (non-sensitive settings only)

NEVER hardcode credentials in source code.
Add .env to .gitignore immediately.

Usage
-----
from secure_config import get_config
cfg = get_config()
login    = cfg.mt5_login
password = cfg.mt5_password
server   = cfg.mt5_server
symbols  = cfg.symbols
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ── .env file loader (no external dependency) ─────────────────────────────────

# Keys that .env should ALWAYS override, even if already set in os.environ.
# This prevents stale Windows system/user env vars left over from a previous
# bot version from silently overriding your current .env settings.
_DOTENV_FORCE_KEYS = {
    "SYMBOLS", "MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER",
    "RISK_PER_TRADE", "MAX_DAILY_LOSS", "MAX_DRAWDOWN",
    "MAX_TRADES_DAY", "MAX_CONCURRENT", "PROP_MODE",
    "MIN_SIGNAL_PROB", "SLEEP_INTERVAL", "BARS", "LOT_SIZE",
    "BROKER_TYPE", "LOG_LEVEL", "ATR_MULTIPLIER",
    "MAX_POSITIONS_SYMBOL", "WF_OPTIMIZE_INTERVAL",
    "MAX_GROUP_RISK_PCT", "CORR_GROUPS",
}


def _load_dotenv(path: str = ".env") -> None:
    """
    Parse a KEY=VALUE .env file and inject into os.environ.

    Priority rules
    --------------
    - For keys in _DOTENV_FORCE_KEYS: .env ALWAYS wins, overwriting any
      stale system/user environment variable set by a previous bot version.
    - For all other keys: existing os.environ value takes precedence
      (standard dotenv behaviour), so unrelated system vars are untouched.

    Also handles:
      - Quoted values:   KEY="my value"  or  KEY='my value'
      - Inline comments: KEY=value  # this is ignored
      - Blank lines and # comment-only lines are skipped
    """
    env_path = Path(path)
    if not env_path.exists():
        logger.warning(
            f".env file not found at {Path(path).resolve()} — "
            "using environment variables / defaults only."
        )
        return

    loaded: dict = {}
    with open(env_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, rest = line.partition("=")
            key = key.strip()
            # Strip inline comments before removing quotes
            if "#" in rest:
                rest = rest[:rest.index("#")]
            value = rest.strip().strip("'\"")
            if key:
                loaded[key] = value

    overwritten: list = []
    for key, value in loaded.items():
        if key in _DOTENV_FORCE_KEYS:
            old = os.environ.get(key)
            os.environ[key] = value
            if old is not None and old != value:
                overwritten.append(f"{key}: {old!r} -> {value!r}")
        elif key not in os.environ:
            os.environ[key] = value

    if overwritten:
        logger.warning(
            f"[secure_config] .env overrode {len(overwritten)} stale system "
            f"env var(s): " + "; ".join(overwritten)
        )
    logger.debug(f".env loaded from {env_path.resolve()} ({len(loaded)} keys)")


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class BotConfig:
    # MT5 credentials
    mt5_login:    int    = 0
    mt5_password: str    = ""
    mt5_server:   str    = ""

    # Trading settings — UPDATED to v7 broker symbols (no spaces in list)
    symbols:              List[str] = field(default_factory=lambda: [
        "XAUUSD..", "BTCUSD..", "US100..", "US30..", "US500..", "XAGUSD.."
    ])
    lot_size:             float     = 0.01
    max_positions_symbol: int       = 2
    sleep_interval:       int       = 300      # seconds
    # 3-month H1 context (90d × 24h = 2160 bars)
    bars:                 int       = 2160

    # Risk settings
    risk_per_trade:   float = 0.007   # 0.7%
    max_daily_loss:   float = 0.03    # 3%
    max_drawdown:     float = 0.08    # 8%
    max_trades_day:   int   = 10
    max_concurrent:   int   = 5
    atr_multiplier:   float = 1.5

    # Prop-firm mode
    prop_mode:        bool  = True

    # Signal quality
    min_signal_prob:  float = 0.35

    # Logging
    log_level: str = "INFO"

    def is_valid(self) -> bool:
        """
        Return True if all critical credentials for the active broker are set.
        MT5 creds only checked when BROKER_TYPE=mt5.
        IBKR/cTrader credentials are validated by their adapters directly.
        """
        import os as _os
        broker = _os.environ.get("BROKER_TYPE", "mt5").lower().strip()
        if broker == "mt5":
            return bool(self.mt5_login and self.mt5_password and self.mt5_server)
        elif broker in ("ibkr", "ib", "interactivebrokers"):
            return True   # IBKRAdapter validates its own env vars
        elif broker in ("ctrader", "spotware"):
            return True   # CTraderAdapter validates its own env vars
        return True       # Unknown broker — let the adapter handle it

    def summary(self) -> str:
        return (
            f"MT5 account={self.mt5_login} server={self.mt5_server} | "
            f"symbols={self.symbols} | prop_mode={self.prop_mode} | "
            f"risk={self.risk_per_trade*100:.1f}%/trade | "
            f"max_dd={self.max_drawdown*100:.0f}%"
        )


# ── Config factory ────────────────────────────────────────────────────────────

_config: Optional[BotConfig] = None


def get_config(reload: bool = False) -> BotConfig:
    """
    Build and return the singleton BotConfig.
    Reads environment variables (after loading .env if present).
    """
    global _config
    if _config is not None and not reload:
        return _config

    # Step 1: load .env file if present
    _load_dotenv(".env")

    cfg = BotConfig()

    # ── MT5 credentials ───────────────────────────────────────────────────────
    # Only warn about missing MT5 credentials when BROKER_TYPE=mt5
    _broker_type = os.environ.get("BROKER_TYPE", "mt5").lower().strip()
    _mt5_mode = (_broker_type == "mt5")

    login_raw = os.environ.get("MT5_LOGIN", "")
    if login_raw.isdigit():
        cfg.mt5_login = int(login_raw)
    elif _mt5_mode:
        logger.warning("MT5_LOGIN not set or not numeric in environment.")

    cfg.mt5_password = os.environ.get("MT5_PASSWORD", "")
    if not cfg.mt5_password and _mt5_mode:
        logger.warning("MT5_PASSWORD not set in environment.")

    cfg.mt5_server = os.environ.get("MT5_SERVER", "")
    if not cfg.mt5_server and _mt5_mode:
        logger.warning("MT5_SERVER not set in environment.")

    # ── Trading settings ──────────────────────────────────────────────────────
    symbols_raw = os.environ.get("SYMBOLS", "")
    if symbols_raw:
        # Strip spaces around commas so "US100.., US30.." works correctly
        cfg.symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]

    cfg.lot_size             = float(os.environ.get("LOT_SIZE", cfg.lot_size))
    cfg.max_positions_symbol = int(os.environ.get("MAX_POSITIONS_SYMBOL", cfg.max_positions_symbol))
    cfg.sleep_interval       = int(os.environ.get("SLEEP_INTERVAL", cfg.sleep_interval))
    cfg.bars                 = int(os.environ.get("BARS", cfg.bars))

    # ── Risk settings ─────────────────────────────────────────────────────────
    cfg.risk_per_trade  = float(os.environ.get("RISK_PER_TRADE",  cfg.risk_per_trade))
    cfg.max_daily_loss  = float(os.environ.get("MAX_DAILY_LOSS",  cfg.max_daily_loss))
    cfg.max_drawdown    = float(os.environ.get("MAX_DRAWDOWN",    cfg.max_drawdown))
    cfg.max_trades_day  = int(os.environ.get("MAX_TRADES_DAY",    cfg.max_trades_day))
    cfg.max_concurrent  = int(os.environ.get("MAX_CONCURRENT",    cfg.max_concurrent))
    cfg.atr_multiplier  = float(os.environ.get("ATR_MULTIPLIER",  cfg.atr_multiplier))
    cfg.prop_mode       = os.environ.get("PROP_MODE", "true").lower() in ("1", "true", "yes")
    cfg.min_signal_prob = float(os.environ.get("MIN_SIGNAL_PROB", cfg.min_signal_prob))
    cfg.log_level       = os.environ.get("LOG_LEVEL", cfg.log_level).upper()

    if not cfg.is_valid():
        import os as _os2
        _bt = _os2.environ.get("BROKER_TYPE", "mt5").lower()
        logger.error(
            f"Bot config is INVALID for broker_type={_bt!r}. "
            "Check your .env file or environment variables."
        )
    else:
        logger.info(f"Config loaded: {cfg.summary()}")

    _config = cfg
    return _config


def generate_env_template(path: str = ".env.example") -> None:
    """Write a .env.example template to disk so users know what to fill in."""
    template = """\
# AI EA v17 — Environment Variables
# Copy this file to .env and fill in your credentials.
# NEVER commit .env to version control.

# ── MT5 Credentials ──────────────────────────────────────────────────────────
MT5_LOGIN=YOUR_ACCOUNT_NUMBER
MT5_PASSWORD=YOUR_MT5_PASSWORD
MT5_SERVER=YOUR_BROKER_SERVER

# ── Trading Settings ─────────────────────────────────────────────────────────
# No spaces between comma-separated symbols
SYMBOLS=XAUUSD..,BTCUSD..,US100..,US30..,US500..,XAGUSD..
LOT_SIZE=0.01
MAX_POSITIONS_SYMBOL=2
SLEEP_INTERVAL=300
# 3-month H1 bars for deep ML context (90d × 24h)
BARS=2160

# ── Risk Settings ────────────────────────────────────────────────────────────
RISK_PER_TRADE=0.007
MAX_DAILY_LOSS=0.03
MAX_DRAWDOWN=0.08
MAX_TRADES_DAY=10
MAX_CONCURRENT=5
ATR_MULTIPLIER=1.5

# ── Prop-Firm Mode ───────────────────────────────────────────────────────────
PROP_MODE=true
MIN_SIGNAL_PROB=0.35

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL=INFO
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(template)
    print(f"[secure_config] Template written to {path}")


if __name__ == "__main__":
    # Set up basic logging so the "Config loaded:" line actually prints
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    generate_env_template()
    cfg = get_config()
    # Also print directly so it's visible even if the logger level is wrong
    print(f"Config loaded: {cfg.summary()}")
