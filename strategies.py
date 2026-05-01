# strategies.py
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict
import copy

class StrategyManager:
    def __init__(self, path: str = "data/strategies.json"):
        self.path = path
        self.weights_path = "data/strategy_weights.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump({"user": [], "generated": []}, f)
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        if not os.path.exists(self.weights_path):
            with open(self.weights_path, "w") as f:
                json.dump({}, f)

        # keep per-symbol bias map (None/buy/sell)
        self.symbol_bias = {}

    def load_strategies(self) -> Dict[str, List[Dict]]:
        try:
            with open(self.path, "r") as file:
                return json.load(file)
        except Exception as e:
            logging.error(f"Failed to load strategies: {str(e)}")
            return {"user": [], "generated": []}

    def load_weights(self) -> Dict[str, float]:
        try:
            with open(self.weights_path, "r") as file:
                return json.load(file)
        except Exception as e:
            logging.error(f"Failed to load strategy weights: {str(e)}")
            return {}

    def save_strategies(self, strategies: dict) -> None:
        try:
            with open(self.path, "w") as file:
                json.dump(strategies, file, indent=2)
        except Exception as e:
            logging.error(f"Failed to save strategies: {str(e)}")

    def save_weights(self, weights: Dict[str, float]) -> None:
        try:
            with open(self.weights_path, "w") as file:
                json.dump(weights, file, indent=2)
        except Exception as e:
            logging.error(f"Failed to save strategy weights: {str(e)}")

    def get_all_strategies(self) -> List[Dict]:
        data = self.load_strategies()
        return data.get("user", []) + data.get("generated", [])

    def get_active_strategies(self, bias: Optional[str] = None) -> List[Dict]:
        """
        Return active strategies. If bias is provided ('buy' or 'sell'),
        prefer strategies matching that direction (strategy['direction'] or strategy tag).
        If none match bias, return all active strategies.
        This method returns copies (does not mutate stored JSON).
        """
        data = self.load_strategies()
        weights = self.load_weights()
        all_strats = data.get("user", []) + data.get("generated", [])
        active = []
        buy_pref = []
        sell_pref = []

        for s in all_strats:
            if not s.get("active", True):
                continue
            s_copy = copy.deepcopy(s)
            name = s_copy.get("name", "")
            if name in weights:
                s_copy["weight"] = weights[name]
            # canonicalize direction if present
            direction = s_copy.get("direction")
            tags = s_copy.get("tags", [])
            if direction in ("buy", "sell"):
                if direction == "buy":
                    buy_pref.append(s_copy)
                else:
                    sell_pref.append(s_copy)
            else:
                # infer from tags
                if "bull" in tags or "long" in tags:
                    buy_pref.append(s_copy)
                elif "bear" in tags or "short" in tags:
                    sell_pref.append(s_copy)
                else:
                    active.append(s_copy)

        # If bias specified, return matching subset first + a few neutrals
        if bias == "buy":
            result = buy_pref + active + sell_pref
            return result
        elif bias == "sell":
            result = sell_pref + active + buy_pref
            return result

        # no bias -> return all active with weights applied
        return buy_pref + sell_pref + active

    def add_user_strategy(self, strategy: Dict) -> None:
        data = self.load_strategies()
        strategy["version"] = 1
        strategy["created"] = datetime.now().isoformat()
        strategy["active"] = True
        data.setdefault("user", []).append(strategy)
        self.save_strategies(data)

    def add_generated_strategies(self, new_strategies: List[Dict]) -> None:
        data = self.load_strategies()
        data.setdefault("generated", []).extend(new_strategies)
        self.save_strategies(data)

    def update_strategy_weights(self, loss_patterns: Dict) -> None:
        weights = self.load_weights()
        strategies = self.get_all_strategies()
        for strategy in strategies:
            sname = strategy.get("name", "")
            if sname and sname not in weights:
                weights[sname] = 1.0
        if 'strategy_patterns' in loss_patterns:
            for sname, loss_ratio in loss_patterns['strategy_patterns'].items():
                if sname in weights:
                    weights[sname] *= max(0.2, 1 - loss_ratio)
        if 'time_of_day' in loss_patterns:
            current_hour = datetime.now().hour
            hour_loss_ratio = loss_patterns['time_of_day'].get(current_hour, 0)
            if hour_loss_ratio > 0.2:
                for strategy in strategies:
                    if strategy.get("active", True):
                        sname = strategy.get("name", "")
                        if sname in weights:
                            weights[sname] *= max(0.4, 1 - hour_loss_ratio)
        if 'market_conditions' in loss_patterns:
            market_conditions = loss_patterns['market_conditions']
            if market_conditions.get('high_volatility', 0) > 0.1:
                for strategy in strategies:
                    if "high_volatility" in strategy.get("tags", []):
                        sname = strategy.get("name", "")
                        if sname in weights:
                            weights[sname] *= 0.5
            if market_conditions.get('low_volatility', 0) > 0.1:
                for strategy in strategies:
                    if "low_volatility" in strategy.get("tags", []):
                        sname = strategy.get("name", "")
                        if sname in weights:
                            weights[sname] *= 0.5
        for sname in list(weights.keys()):
            weights[sname] = max(0.1, min(4.0, weights[sname]))
        self.save_weights(weights)
        logging.info("Updated strategy weights based on loss patterns")

    def refresh_strategies(self, results: List[Dict], trade_history: Optional[List[Dict]] = None) -> None:
        data = self.load_strategies()
        weights = self.load_weights()
        strategy_performance = {}
        for result in results:
            strategy_name = result.get("name", "")
            if strategy_name:
                strategy_performance[strategy_name] = {
                    'score': result.get("score", 0),
                    'win_rate': result.get("win_rate", 0),
                    'profit': result.get("profit", 0)
                }
        if trade_history:
            trade_counts = defaultdict(int)
            trade_profits = defaultdict(float)
            for trade in trade_history:
                sname = trade.get("strategy", "")
                if sname:
                    trade_counts[sname] += 1
                    trade_profits[sname] += trade.get("profit", 0)
            for sname in trade_counts:
                if sname in strategy_performance:
                    strategy_performance[sname]['trade_count'] = trade_counts[sname]
                    strategy_performance[sname]['realized_profit'] = trade_profits[sname]
        for strategy in data.get("generated", []):
            name = strategy.get("name", "")
            if name in strategy_performance:
                perf = strategy_performance[name]
                if perf.get('win_rate', 0) < 40 or perf.get('profit', 0) < 0:
                    strategy["active"] = False
                    weights[name] = max(0.1, weights.get(name, 1.0) * 0.5)
                elif perf.get('win_rate', 0) > 50 and perf.get('profit', 0) > 0:
                    strategy["active"] = True
                    weights[name] = min(4.0, weights.get(name, 1.0) * 1.2)
        top_performers = sorted(strategy_performance.items(), key=lambda x: x[1].get('score', 0), reverse=True)[:5]
        for strategy_name, _ in top_performers:
            for strategy in data.get("generated", []):
                if strategy.get("name", "") == strategy_name:
                    strategy["active"] = True
                    weights[strategy_name] = min(4.0, weights.get(strategy_name, 1.0) * 1.5)
                    break
        self.save_strategies(data)
        self.save_weights(weights)
        logging.info("Refreshed strategy pool and weights based on performance")

    def log_performance(self, results: List[Dict]) -> None:
        log_dir = "logs/performance"
        os.makedirs(log_dir, exist_ok=True)
        filename = f"performance_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, "a") as f:
                weights = self.load_weights()
                for result in results:
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "strategy": result.get("name"),
                        "symbol": result.get("symbol", "N/A"),
                        "win_rate": result.get("win_rate", 0),
                        "profit": result.get("profit", 0),
                        "direction": result.get("direction", "N/A"),
                        "score": result.get("score", 0),
                        "weight": weights.get(result.get("name"), 1.0)
                    }
                    f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logging.error(f"Failed to log performance: {str(e)}")

    # --- new small API for bias handling ---
    def set_bias_for_symbol(self, symbol: str, bias: Optional[str]) -> None:
        if bias not in (None, "buy", "sell"):
            bias = None
        self.symbol_bias[symbol] = bias

    def get_bias_for_symbol(self, symbol: str) -> Optional[str]:
        return self.symbol_bias.get(symbol) 