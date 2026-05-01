"""
watchdog.py — Global training watchdog (AI EA v14)
===================================================
A lightweight singleton that kills the process if no progress heartbeat
is received within TIMEOUT seconds.  Import and call reset() at the start
of any potentially long operation.  signal_engine, feature_engineering, and
trainer all import this — NOT trainer itself — so there are zero circular deps.
"""
import os
import sys
import signal
import threading
import time as _time
import logging

logger = logging.getLogger(__name__)

TIMEOUT_SECS = 600   # 10 minutes hard limit per operation


class Watchdog:
    """
    Background-thread watchdog.  daemon=True so it never blocks clean exit.
    Sends SIGTERM on timeout, then os._exit(1) after 2s grace if ignored.
    """
    def __init__(self, timeout: int = TIMEOUT_SECS):
        self.timeout     = timeout
        self._last_reset = _time.monotonic()
        self._label      = "startup"
        self._lock       = threading.Lock()
        self._stop_evt   = threading.Event()
        self._thread     = threading.Thread(target=self._run, daemon=True, name="watchdog")
        self._thread.start()

    def reset(self, label: str = "") -> None:
        """Call this at the start of each major operation to prove progress."""
        with self._lock:
            self._last_reset = _time.monotonic()
            self._label      = label
        if label:
            logger.debug(f"[watchdog] reset — {label}")

    def stop(self) -> None:
        """Call in finally block to let the process exit cleanly."""
        self._stop_evt.set()

    def _run(self) -> None:
        while not self._stop_evt.wait(timeout=30):
            with self._lock:
                elapsed = _time.monotonic() - self._last_reset
                label   = self._label
            if elapsed > self.timeout:
                logger.error(
                    f"[watchdog] TIMEOUT — '{label}' has not progressed in "
                    f"{elapsed:.0f}s (limit={self.timeout}s). Terminating."
                )
                _time.sleep(0.5)
                try:
                    os.kill(os.getpid(), signal.SIGTERM)
                except Exception:
                    pass
                _time.sleep(2)
                os._exit(1)


# Module-level singleton — import watchdog and call watchdog.instance.reset()
instance: Watchdog = None   # type: ignore  # set to None until activate() called


def activate(timeout: int = TIMEOUT_SECS) -> Watchdog:
    """Create and start the watchdog.  Call once from trainer.py main."""
    global instance
    if instance is None:
        instance = Watchdog(timeout=timeout)
        logger.info(f"[watchdog] activated — timeout={timeout}s per operation")
    return instance
