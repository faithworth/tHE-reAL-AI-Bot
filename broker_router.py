"""
broker_router.py — Universal Broker Router (AI EA v20)
======================================================
Selects and initialises the correct broker adapter at runtime based on
the BROKER_TYPE environment variable (or explicit argument).

Supported broker types:
    mt5       → MT5Adapter       (MetaTrader 5)
    ibkr      → IBKRAdapter      (Interactive Brokers via ib_insync)
    ctrader   → CTraderAdapter   (Spotware cTrader Open API)
    alpaca    → AlpacaAdapter    (Alpaca Markets — stocks/crypto, paper/live/offline)

Usage
-----
from broker_router import BrokerRouter

router = BrokerRouter()           # reads BROKER_TYPE from env
broker = router.get_broker()      # returns connected BaseBroker instance

Or force a specific type:
    router = BrokerRouter(broker_type="alpaca")
    broker = router.get_broker()

Environment variables read per broker:
  MT5:
    BROKER_TYPE=mt5
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

  IBKR:
    BROKER_TYPE=ibkr
    IBKR_HOST (default 127.0.0.1)
    IBKR_PORT (default 7497)
    IBKR_CLIENT_ID (default 1)
    IBKR_ACCOUNT (optional — auto-detected if blank)
    IBKR_PAPER (default true)

  cTrader:
    BROKER_TYPE=ctrader
    CTRADER_CLIENT_ID
    CTRADER_CLIENT_SECRET
    CTRADER_ACCESS_TOKEN
    CTRADER_ACCOUNT_ID
    CTRADER_DEMO (default true)

  Alpaca:
    BROKER_TYPE=alpaca
    ALPACA_API_KEY        — from alpaca.markets dashboard
    ALPACA_SECRET_KEY     — from alpaca.markets dashboard
    ALPACA_PAPER=true     — false for live funded account
    ALPACA_DATA_FEED=iex  — iex (free) or sip (paid, real-time)
    ALPACA_OFFLINE=false  — true = fully simulated, zero API calls (friend mode)
"""

import logging
import os
from typing import Optional

from base_broker import BaseBroker

logger = logging.getLogger(__name__)


class BrokerRouter:
    """
    Factory that builds the correct BaseBroker subclass from env vars.
    Stores the singleton broker instance after first connect.
    """

    def __init__(
        self,
        broker_type: Optional[str] = None,
        risk_engine=None,
    ):
        self._broker_type = (
            (broker_type or os.getenv("BROKER_TYPE", "mt5")).lower().strip()
        )
        self._risk_engine = risk_engine
        self._broker: Optional[BaseBroker] = None

        logger.info(f"[BrokerRouter] Selected broker type: {self._broker_type!r}")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def get_broker(self) -> BaseBroker:
        """
        Return a connected BaseBroker instance.
        Raises RuntimeError if connection fails.
        """
        if self._broker is not None and self._broker.connected:
            return self._broker

        broker = self._build_broker()
        connected = broker.connect()

        if not connected:
            raise RuntimeError(
                f"[BrokerRouter] Failed to connect to broker: {self._broker_type!r}. "
                f"Check credentials, connectivity, and that the platform is running."
            )

        self._broker = broker
        logger.info(
            f"[BrokerRouter] {broker.broker_name} connected successfully "
            f"| equity={broker.get_equity():.2f}"
        )
        return broker

    def get_broker_type(self) -> str:
        """Return the current broker type string."""
        return self._broker_type

    def reconnect(self) -> bool:
        """Force a fresh reconnect on the active broker."""
        if self._broker is not None:
            try:
                self._broker.disconnect()
            except Exception:
                pass
        try:
            broker = self._build_broker()
            if broker.connect():
                self._broker = broker
                logger.info(f"[BrokerRouter] Reconnected to {self._broker_type}")
                return True
        except Exception as e:
            logger.error(f"[BrokerRouter] Reconnect failed: {e}")
        return False

    def shutdown(self) -> None:
        """Gracefully disconnect the active broker."""
        if self._broker is not None:
            try:
                self._broker.disconnect()
            except Exception as e:
                logger.warning(f"[BrokerRouter] shutdown error: {e}")
            self._broker = None
            logger.info("[BrokerRouter] Broker disconnected.")

    # ─────────────────────────────────────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────────────────────────────────────

    def _build_broker(self) -> BaseBroker:
        """Instantiate (not yet connected) broker adapter from env vars."""
        t = self._broker_type

        if t == "mt5":
            return self._build_mt5()
        elif t in ("ibkr", "ib", "interactivebrokers"):
            return self._build_ibkr()
        elif t in ("ctrader", "spotware"):
            return self._build_ctrader()
        elif t in ("alpaca", "alpaca_markets"):
            return self._build_alpaca()
        else:
            raise ValueError(
                f"[BrokerRouter] Unknown broker type: {t!r}. "
                f"Valid values: mt5, ibkr, ctrader, alpaca"
            )

    def _build_mt5(self) -> BaseBroker:
        from mt5_adapter import MT5Adapter

        login    = int(os.getenv("MT5_LOGIN", "0"))
        password = os.getenv("MT5_PASSWORD", "")
        server   = os.getenv("MT5_SERVER", "")

        if not login:
            logger.warning("[BrokerRouter] MT5_LOGIN not set in environment.")
        if not password:
            logger.warning("[BrokerRouter] MT5_PASSWORD not set in environment.")

        logger.info(
            f"[BrokerRouter] Building MT5Adapter "
            f"login={login} server={server!r}"
        )
        return MT5Adapter(
            login=login,
            password=password,
            server=server,
            risk_engine=self._risk_engine,
        )

    def _build_ibkr(self) -> BaseBroker:
        from ibkr_adapter import IBKRAdapter

        host        = os.getenv("IBKR_HOST", "127.0.0.1")
        port        = int(os.getenv("IBKR_PORT", "7497"))
        client_id   = int(os.getenv("IBKR_CLIENT_ID", "1"))
        account     = os.getenv("IBKR_ACCOUNT", "")
        paper       = os.getenv("IBKR_PAPER", "true").lower() in ("1", "true", "yes")

        logger.info(
            f"[BrokerRouter] Building IBKRAdapter "
            f"host={host}:{port} clientId={client_id} paper={paper}"
        )
        return IBKRAdapter(
            host=host,
            port=port,
            client_id=client_id,
            account=account,
            risk_engine=self._risk_engine,
            paper_trading=paper,
        )

    def _build_ctrader(self) -> BaseBroker:
        from ctrader_adapter import CTraderAdapter

        client_id     = os.getenv("CTRADER_CLIENT_ID", "")
        client_secret = os.getenv("CTRADER_CLIENT_SECRET", "")
        access_token  = os.getenv("CTRADER_ACCESS_TOKEN", "")
        account_id    = int(os.getenv("CTRADER_ACCOUNT_ID", "0"))
        demo          = os.getenv("CTRADER_DEMO", "true").lower() in ("1", "true", "yes")

        if not client_id:
            logger.warning("[BrokerRouter] CTRADER_CLIENT_ID not set.")
        if not access_token:
            logger.warning("[BrokerRouter] CTRADER_ACCESS_TOKEN not set.")
        if not account_id:
            logger.warning("[BrokerRouter] CTRADER_ACCOUNT_ID not set.")

        logger.info(
            f"[BrokerRouter] Building CTraderAdapter "
            f"account={account_id} demo={demo}"
        )
        return CTraderAdapter(
            client_id=client_id,
            client_secret=client_secret,
            access_token=access_token,
            account_id=account_id,
            demo=demo,
            risk_engine=self._risk_engine,
        )

    def _build_alpaca(self) -> BaseBroker:
        from alpaca_adapter import AlpacaAdapter

        api_key    = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        paper      = os.getenv("ALPACA_PAPER", "true").lower() in ("1", "true", "yes")
        data_feed  = os.getenv("ALPACA_DATA_FEED", "iex")
        offline    = os.getenv("ALPACA_OFFLINE", "false").lower() in ("1", "true", "yes")

        if not offline and (not api_key or not secret_key):
            logger.warning(
                "[BrokerRouter] ALPACA_API_KEY / ALPACA_SECRET_KEY not set. "
                "Switching to offline simulation mode automatically."
            )
            offline = True

        mode = "OFFLINE" if offline else ("PAPER" if paper else "LIVE")
        logger.info(
            f"[BrokerRouter] Building AlpacaAdapter mode={mode} "
            f"data_feed={data_feed}"
        )
        return AlpacaAdapter(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
            data_feed=data_feed,
            offline=offline,
            risk_engine=self._risk_engine,
        )

    def __repr__(self) -> str:
        return (
            f"<BrokerRouter type={self._broker_type!r} "
            f"connected={self._broker.connected if self._broker else False}>"
        )
