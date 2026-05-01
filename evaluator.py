import os
import logging
import json
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple, Union
from Backtester import Backtester

PERFORMANCE_FILE = "data/performance.json"

class StrategyEvaluator:
    def __init__(self):
        os.makedirs(os.path.dirname(PERFORMANCE_FILE), exist_ok=True)
        self.backtester = Backtester()
        self.indicators_config = {
            'rsi': {'period': 14},
            'atr': {'period': 14},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger': {'period': 20, 'std_dev': 2},
            'stoch': {'k_period': 14, 'd_period': 3},
            'ict': {
                'fvg_lookback': 3,
                'ob_wick_ratio': 0.7,
                'liquidity_window': 20,
                'volume_spike_threshold': 2.0,
                'killzones': {
                    'london': (7, 9),
                    'new_york': (8, 10),
                    'tokyo': (0, 2)  # Added Tokyo session
                }
            }
        }
        
    # ========== Core Methods (Maintained Exactly) ==========
    def add_market_indicators(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced to include ICT/SMC specific indicators while maintaining original functionality
        """
        if not isinstance(market_data, pd.DataFrame) or len(market_data) < 20:
            return market_data
            
        try:
            df = market_data.copy()
            
            # Original indicators
            df['rsi'] = self._calculate_rsi(df['close'], self.indicators_config['rsi']['period'])
            df['atr'] = self._calculate_atr(df, self.indicators_config['atr']['period'])
            df['macd_line'], df['macd_signal'] = self._calculate_macd(df['close'])
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
            df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df)
            df['volatility'] = df['close'].pct_change().rolling(14).std() * 100
            df['trend_strength'] = self._calculate_adx(df)
            
            # Price action metrics
            df['body'] = df['close'] - df['open']
            df['range'] = df['high'] - df['low']
            df['body_pct'] = abs(df['body']) / df['range']
            
            if 'real_volume' in df.columns:
                df['volume_ma'] = df['real_volume'].rolling(20).mean()
                df['volume_ratio'] = df['real_volume'] / df['volume_ma']
            
            # Enhanced ICT/SMC indicators
            df['fvg_bullish'], df['fvg_bearish'] = self._calculate_enhanced_fvg(df)
            df['ob_bullish'], df['ob_bearish'] = self._calculate_order_blocks(df)
            df['liquidity_pools'] = self._calculate_liquidity_pools(df)
            df['mitigation_blocks'] = self._calculate_mitigation_blocks(df)
            df['liquidity_grabs'] = self._calculate_liquidity_grabs(df)
            
            return df
            
        except Exception as e:
            logging.error(f"Failed to add market indicators: {e}")
            return market_data

    def evaluate_strategy(self, strategy: Dict, market_data: pd.DataFrame, symbol: str,
                         market_context: Optional[Dict] = None) -> Dict:
        """
        Enhanced evaluation with ICT/SMC strategy support while maintaining original functionality
        """
        try:
            # Add indicators if not present
            if 'rsi' not in market_data.columns:
                market_data = self.add_market_indicators(market_data)
            
            # Check ICT-specific conditions first if this is an ICT strategy
            if strategy.get('strategy_type') in ['ict', 'smc']:
                ict_valid = self._validate_ict_conditions(strategy, market_data)
                if not ict_valid:
                    return self._create_failed_result(strategy, symbol)
            
            # Run standard backtest (original functionality preserved)
            result = self.backtester.test_strategy(strategy, market_data, symbol)
            
            # Enhanced confidence calculation
            confidence = self._calculate_enhanced_confidence(result, market_context, strategy)
            
            return {
                "name": strategy["name"],
                "win_rate": result["win_rate"],
                "profit": result["profit"],
                "direction": strategy["direction"],
                "symbol": symbol,
                "score": result["score"],
                "confidence": confidence,
                "market_conditions": self._get_market_conditions(market_data),
                "ict_metrics": self._get_ict_metrics(market_data) if strategy.get('strategy_type') in ['ict', 'smc'] else None
            }
        except Exception as e:
            logging.error(f"Evaluation failed for {strategy['name']}: {e}")
            return self._create_failed_result(strategy, symbol)

    # ========== Enhanced ICT/SMC Methods ==========
    def _validate_ict_conditions(self, strategy: Dict, market_data: pd.DataFrame) -> bool:
        """Enhanced ICT condition validation with multiple confirmation checks"""
        conditions = []
        
        # Time-based conditions
        if 'time_filter' in strategy:
            conditions.append(self._check_killzone(strategy['time_filter']))
        
        # Price action conditions
        if 'required_fvg' in strategy:
            conditions.append(self._check_fvg_condition(market_data, strategy['required_fvg']))
        
        if 'required_ob' in strategy:
            conditions.append(self._check_ob_condition(market_data, strategy['required_ob']))
        
        if 'required_liquidity' in strategy:
            conditions.append(self._check_liquidity_condition(market_data, strategy['required_liquidity']))
        
        # Market structure conditions
        if 'required_trend' in strategy:
            current_trend = "up" if market_data['close'].iloc[-1] > market_data['ema50'].iloc[-1] else "down"
            conditions.append(current_trend == strategy['required_trend'])
        
        return all(conditions)
    
    def _check_killzone(self, time_filter: str) -> bool:
        """Enhanced killzone check with multiple sessions"""
        if time_filter.lower() not in self.indicators_config['ict']['killzones']:
            logging.warning(f"Unknown killzone: {time_filter}")
            return False
            
        tz_map = {
            'london': 'Europe/London',
            'new_york': 'America/New_York',
            'tokyo': 'Asia/Tokyo'
        }
        
        tz = pytz.timezone(tz_map[time_filter.lower()])
        now = datetime.now(tz).time()
        start_hour, end_hour = self.indicators_config['ict']['killzones'][time_filter.lower()]
        
        return time(start_hour, 0) <= now <= time(end_hour, 0)
    
    def _calculate_enhanced_fvg(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Vectorised FVG detection — no Python row loop."""
        lookback = self.indicators_config['ict']['fvg_lookback']
        low   = df['low']
        high  = df['high']
        close = df['close']
        fvg_bullish = (low > high.shift(lookback)) & (close > high.shift(1))
        fvg_bearish = (high < low.shift(lookback))  & (close < low.shift(1))
        return fvg_bullish.fillna(False), fvg_bearish.fillna(False)

    def _calculate_mitigation_blocks(self, df: pd.DataFrame) -> pd.Series:
        """Vectorised ICT Mitigation Block detection."""
        prev_close = df['close'].shift(1)
        prev_open  = df['open'].shift(1)
        prev_low   = df['low'].shift(1)
        prev_high  = df['high'].shift(1)
        bullish_mb = (prev_close < prev_open) & (df['close'] > prev_low)
        bearish_mb = (prev_close > prev_open) & (df['close'] < prev_high)
        return (bullish_mb | bearish_mb).fillna(False)

    def _calculate_liquidity_grabs(self, df: pd.DataFrame) -> pd.Series:
        """Vectorised liquidity sweep detection."""
        low2  = df['low'].shift(2)
        high2 = df['high'].shift(2)
        bullish_sweep = (df['low'] < low2)   & (df['close'] > low2)
        bearish_sweep = (df['high'] > high2) & (df['close'] < high2)
        return (bullish_sweep | bearish_sweep).fillna(False)

    # ========== Original Methods (Maintained Exactly) ==========
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Original RSI calculation preserved exactly"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Original ATR calculation preserved exactly"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Original MACD calculation preserved exactly"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
        """Original Bollinger Bands calculation preserved exactly"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple:
        """Original Stochastic calculation preserved exactly"""
        low_min = df['low'].rolling(k_period).min()
        high_max = df['high'].rolling(k_period).max()
        
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d = k.rolling(d_period).mean()
        return k, d

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Original ADX calculation preserved exactly"""
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self._calculate_atr(df, period)
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        return dx.ewm(alpha=1/period).mean()

    def save_performance(self, performance_data: Dict) -> None:
        """Original performance saving preserved exactly"""
        try:
            with open(PERFORMANCE_FILE, "w") as f:
                json.dump(performance_data, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save performance: {e}")

    def load_performance(self) -> Dict:
        """Original performance loading preserved exactly"""
        try:
            if os.path.exists(PERFORMANCE_FILE):
                with open(PERFORMANCE_FILE, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"Failed to load performance: {e}")
            return {}

    # ========== Original ICT Methods (Maintained with Enhancements) ==========
    def _calculate_fvg(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Original FVG detection now calls enhanced version"""
        return self._calculate_enhanced_fvg(df)
    
    def _calculate_order_blocks(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Vectorised order block detection with volume confirmation."""
        wick_ratio  = self.indicators_config['ict']['ob_wick_ratio']
        candle_range = df['range']
        body         = df['close'] - df['open']
        next_close   = df['close'].shift(-1)

        # Bullish OB: bearish body > wick_ratio of range, next close above current high
        bull_cond = (
            (df['close'] < df['open']) &
            ((-body) > candle_range * wick_ratio) &
            (next_close > df['high'])
        )
        # Bearish OB: bullish body > wick_ratio of range, next close below current low
        bear_cond = (
            (df['close'] > df['open']) &
            (body > candle_range * wick_ratio) &
            (next_close < df['low'])
        )
        if 'volume_ratio' in df.columns:
            vol_ok   = df['volume_ratio'] > 1.0
            bull_cond = bull_cond & vol_ok
            bear_cond = bear_cond & vol_ok
        return bull_cond.fillna(False), bear_cond.fillna(False)

    def _calculate_liquidity_pools(self, df: pd.DataFrame) -> pd.Series:
        """Liquidity pool detection with volume confirmation"""
        window = self.indicators_config['ict']['liquidity_window']
        is_swing_high = (df['high'] == df['high'].rolling(window, center=True).max())
        is_swing_low = (df['low'] == df['low'].rolling(window, center=True).min())
        
        if 'volume_ratio' in df.columns:
            volume_threshold = self.indicators_config['ict']['volume_spike_threshold']
            volume_confirmed = df['volume_ratio'] > volume_threshold
            return (is_swing_high | is_swing_low) & volume_confirmed
            
        return is_swing_high | is_swing_low

    def _check_fvg_condition(self, df: pd.DataFrame, condition: Dict) -> bool:
        """Original FVG condition check preserved"""
        direction = condition.get('direction', 'both')
        lookback = condition.get('lookback', 5)
        
        if direction == 'bullish':
            return df['fvg_bullish'].rolling(lookback).sum() > 0
        elif direction == 'bearish':
            return df['fvg_bearish'].rolling(lookback).sum() > 0
        else:
            return (df['fvg_bullish'].rolling(lookback).sum() > 0 or 
                    df['fvg_bearish'].rolling(lookback).sum() > 0)

    def _check_ob_condition(self, df: pd.DataFrame, condition: Dict) -> bool:
        """Original OB condition check preserved"""
        direction = condition.get('direction', 'both')
        lookback = condition.get('lookback', 5)
        
        if direction == 'bullish':
            return df['ob_bullish'].rolling(lookback).sum() > 0
        elif direction == 'bearish':
            return df['ob_bearish'].rolling(lookback).sum() > 0
        else:
            return (df['ob_bullish'].rolling(lookback).sum() > 0 or 
                    df['ob_bearish'].rolling(lookback).sum() > 0)

    def _check_liquidity_condition(self, df: pd.DataFrame, condition: Dict) -> bool:
        """Original liquidity condition check preserved"""
        if 'liquidity_pools' not in df.columns:
            return False
            
        volume_threshold = condition.get('volume_threshold', 1.5)
        lookback = condition.get('lookback', 5)
        
        if 'volume_ratio' in df.columns:
            return (df['liquidity_pools'].rolling(lookback).sum() > 0 and 
                    df['volume_ratio'].iloc[-1] > volume_threshold)
        return df['liquidity_pools'].rolling(lookback).sum() > 0

    def _calculate_enhanced_confidence(self, result: Dict, market_context: Optional[Dict], 
                                     strategy: Dict) -> float:
        """Original confidence calculation with ICT enhancements"""
        base_confidence = min(1.0, max(0.0, result.get('win_rate', 0) / 100))
        
        if not market_context:
            return base_confidence
            
        volatility = market_context.get('volatility', 0)
        trend_strength = market_context.get('trend_strength', 0)
        
        if strategy.get('strategy_type') in ['ict', 'smc']:
            if 'time_filter' in strategy and self._check_killzone(strategy['time_filter']):
                base_confidence *= 1.2
            if 'required_fvg' in strategy:
                base_confidence *= 1.1
                
        if volatility > 1.5:
            base_confidence *= 0.8
        elif volatility < 0.5:
            base_confidence *= 0.9
            
        if trend_strength > 0.7:
            base_confidence *= 1.1
        elif trend_strength < 0.3:
            base_confidence *= 0.9
            
        return min(1.0, max(0.1, base_confidence))

    def _get_market_conditions(self, market_data: pd.DataFrame) -> Dict:
        """Original market conditions check preserved"""
        return {
            "volatility": market_data['volatility'].iloc[-1] if 'volatility' in market_data.columns else 0,
            "trend_strength": market_data['trend_strength'].iloc[-1] if 'trend_strength' in market_data.columns else 0,
            "rsi": market_data['rsi'].iloc[-1] if 'rsi' in market_data.columns else 0,
            "atr": market_data['atr'].iloc[-1] if 'atr' in market_data.columns else 0
        }

    def _get_ict_metrics(self, market_data: pd.DataFrame) -> Dict:
        """Original ICT metrics check preserved"""
        return {
            "fvg_bullish": market_data['fvg_bullish'].iloc[-1] if 'fvg_bullish' in market_data.columns else False,
            "fvg_bearish": market_data['fvg_bearish'].iloc[-1] if 'fvg_bearish' in market_data.columns else False,
            "ob_bullish": market_data['ob_bullish'].iloc[-1] if 'ob_bullish' in market_data.columns else False,
            "ob_bearish": market_data['ob_bearish'].iloc[-1] if 'ob_bearish' in market_data.columns else False,
            "liquidity_pools": market_data['liquidity_pools'].iloc[-1] if 'liquidity_pools' in market_data.columns else False,
            "volume_ratio": market_data['volume_ratio'].iloc[-1] if 'volume_ratio' in market_data.columns else 0
        }

    def _create_failed_result(self, strategy: Dict, symbol: str) -> Dict:
        """Original failed result creation preserved"""
        return {
            "name": strategy["name"],
            "win_rate": 0,
            "profit": 0,
            "direction": strategy["direction"],
            "symbol": symbol,
            "score": 0,
            "confidence": 0,
            "market_conditions": {},
            "ict_metrics": None
        }            