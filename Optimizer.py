"""
Optimizer.py — Strategy Parameter Optimizer (AI EA v5)
------------------------------------------------------
Finds optimal SL/TP parameters for a strategy using walk-forward
grid search over historical backtest data.
"""
import logging
from typing import Dict
from Backtester import Backtester
import pandas as pd

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    def __init__(self):
        self.backtester = Backtester()

    def optimize_parameters(
        self,
        strategy: Dict,
        market_data: pd.DataFrame,
        sl_values: list = None,
        tp_values: list = None,
    ) -> Dict:
        """
        Find optimal stop loss and take profit ATR multiples.
        
        Parameters
        ----------
        strategy    : Strategy dict (passed through to Backtester)
        market_data : OHLCV DataFrame
        sl_values   : SL ATR multiples to test  (default: [1.0, 1.5, 2.0, 2.5])
        tp_values   : TP ATR multiples to test  (default: [2.0, 2.5, 3.0, 3.5, 4.0])

        Returns
        -------
        dict with keys: sl, tp, profit, win_rate
        """
        if sl_values is None:
            sl_values = [1.0, 1.5, 2.0, 2.5]
        if tp_values is None:
            tp_values = [2.0, 2.5, 3.0, 3.5, 4.0]

        best_profit = float("-inf")
        best_params: Dict = {"sl": sl_values[0], "tp": tp_values[0]}

        for sl in sl_values:
            for tp in tp_values:
                if tp <= sl:
                    continue   # skip negative R:R configs
                strategy_var = {**strategy, "sl": sl, "tp": tp}
                try:
                    result = self.backtester.test_strategy(strategy_var, market_data)
                    profit = result.get("profit", float("-inf"))
                    if profit > best_profit:
                        best_profit = profit
                        best_params = {
                            "sl": sl,
                            "tp": tp,
                            "profit": round(profit, 2),
                            "win_rate": round(result.get("win_rate", 0.0), 4),
                        }
                except Exception as exc:
                    logger.debug(f"Backtest failed for sl={sl} tp={tp}: {exc}")

        logger.info(
            f"Optimized: SL={best_params['sl']} TP={best_params['tp']} "
            f"profit={best_params.get('profit', '?')}"
        )
        return best_params
