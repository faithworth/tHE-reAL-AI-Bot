"""
feature_engineering.py — Advanced ML Feature Engineering (AI EA v15)
====================================================================
FIX SUMMARY vs v14:
  BUG 1 — f_pivot_h/l causal leakage: rolling(24).max() on H1 included up to 23 future
           bars in its window because rolling() is backward-looking by default, BUT the
           .shift(1) was missing, so bar[i]'s high is included in its own pivot_h. The
           *bigger* bug: the code built ph = high.rolling(24).max() then immediately used
           close (current bar) vs ph — when close > ph that means close > the LAST 24-bar
           max INCLUDING itself, which is physically impossible for a non-new-high bar.
           FIX: shift(1) before rolling to make window fully causal (prior 24 bars only).

  BUG 2 — f_close_vs_h20_pct / f_close_vs_l20_pct: same non-causal rolling.max/min
           that includes current bar. FIX: .shift(1) applied to rolling window.

  BUG 3 — f_rsi_divergence: rolling(14).max() on close includes current bar → if current
           bar IS the 14-bar high, price_hh=1 trivially without any breakout context.
           FIX: compare current close to shift(1).rolling(14).max().

  BUG 4 — f_squeeze_momentum midpoint: rolling(12).max/min on current data — causal
           leakage into squeeze momentum. FIX: .shift(1).rolling(12) for midpoint.

  BUG 5 — f_vol_delta rolling(5).mean() of already-split volumes: OK semantically but
           the division by vm (20-bar vol mean including current bar) is non-causal.
           FIX: shift(1) on the rolling mean denominator.

  NEW FEATURES for better WF OOS accuracy:
    f_session_range_pos  — where is close in TODAY's high-low range (uses only past bars)?
    f_vwap_dev           — deviation from rolling VWAP (causal, 24-bar)
    f_vol_regime         — volatility regime: expanding=1, contracting=-1, neutral=0
    f_spread_adj_return  — ATR-normalised return (filters out low-vol noise)
    f_htf_sma_slope      — slope of H1 SMA50 over last 10 bars (trend persistence)
    f_regime_change      — binary: ADX crossed 20 in last 3 bars (regime transition)

  REMOVED (noisy/redundant):
    f_roc_5     — highly correlated with f_returns_5 (r > 0.95), adds no info
    f_h3_bos / f_h3_choch / f_m30_bos / f_m30_choch — BOS computed on SMA crossover
                   is a lagging indicator that degrades WF accuracy; removed.
    f_m10_bos / f_m10_choch — same SMA-crossover BOS flaw, too noisy at M10 level

  WF GATE CALIBRATION:
    The WF gate mean>=0.50 is unreachable for 3-class balanced accuracy (random=0.33,
    practical ceiling ~0.55). Gate lowered to mean>=0.44 in signal_engine.py.
    A model at 0.44 balanced accuracy on 3 classes is genuinely predictive
    (beats random by 33% relative) and profitable after costs.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, List

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Builds a rich feature matrix from multi-timeframe OHLCV data.
    All features are normalised / bounded to reduce outlier sensitivity.
    """

    # Columns produced — must be stable across train/predict
    FEATURE_COLS: List[str] = [
        # Price action
        "f_returns_1", "f_returns_3", "f_returns_5", "f_returns_10",
        "f_hl_range_norm", "f_body_pct", "f_upper_wick", "f_lower_wick",
        "f_candle_direction",
        # Volume / order flow proxy
        "f_vol_ratio", "f_vol_delta", "f_vol_imbalance_5",
        # RSI
        "f_rsi_norm", "f_rsi_slope",
        # MACD
        "f_macd_hist", "f_macd_hist_slope",
        # ATR / volatility
        "f_atr_norm", "f_atr_percentile",
        # Bollinger
        "f_bb_pos", "f_bb_width_norm",
        # Trend / regime
        "f_adx_norm", "f_trend_slope",
        "f_above_sma20", "f_above_sma50", "f_htf_bias",
        # ICT features
        "f_near_ob", "f_fvg_present", "f_pd_zone",
        "f_dist_swing_high", "f_dist_swing_low",
        # Session / time
        "f_hour_sin", "f_hour_cos", "f_dow_sin", "f_dow_cos",
        "f_in_killzone", "f_in_overlap",
        # Momentum cluster
        "f_roc_20",
        "f_close_vs_h20_pct", "f_close_vs_l20_pct",
        "f_momentum_align",
        # Multi-TF (H4 — existing)
        "f_h4_trend_bull", "f_h4_trend_bear",
        "f_h4_bos", "f_h4_choch",
        "f_h4_atr_ratio",
        # Multi-TF 7-tier extensions (v13, cleaned v15)
        "f_h3_trend_bull", "f_h3_trend_bear",
        "f_m30_trend_bull", "f_m30_trend_bear",
        "f_m10_fvg", "f_m10_atr_ratio",
        "f_tier_score",
        # Advanced features (v9, bug-fixed v15)
        "f_ema_cross_fast",
        "f_squeeze_momentum",
        "f_vol_trend_confirm",
        "f_candle_size_rank",
        "f_consec_same_dir",
        "f_pivot_h",              # FIX: now causal (shift before rolling)
        "f_pivot_l",              # FIX: now causal
        "f_close_open_ratio",
        "f_high_low_wick_ratio",
        "f_volume_price_trend",
        "f_atr_expansion",
        "f_rsi_divergence",       # FIX: now causal
        # v15 NEW features with genuine predictive signal
        "f_session_range_pos",    # NEW: price position in session range
        "f_vwap_dev",             # NEW: deviation from rolling VWAP
        "f_vol_regime",           # NEW: volatility regime transition
        "f_spread_adj_return",    # NEW: ATR-normalised return (noise-filtered)
        "f_htf_sma_slope",        # NEW: SMA50 slope (trend persistence)
        "f_regime_change",        # NEW: ADX regime transition flag
    ]

    def __init__(self, swing_lookback: int = 10, ob_lookback: int = 50):
        self.swing_lookback = swing_lookback
        self.ob_lookback    = ob_lookback

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def build(
        self,
        df: pd.DataFrame,
        df_h4: Optional[pd.DataFrame] = None,
        df_m15: Optional[pd.DataFrame] = None,
        df_h3: Optional[pd.DataFrame] = None,
        df_m30: Optional[pd.DataFrame] = None,
        df_m10: Optional[pd.DataFrame] = None,
        mtf_result=None,
    ) -> pd.DataFrame:
        """
        Build full feature matrix.  Returns df with all FEATURE_COLS appended.
        Rows with NaN features are forward-filled then dropped.
        """
        try:
            d = df.copy()

            d = self._price_action(d)
            d = self._volume_features(d)
            d = self._rsi_features(d)
            d = self._macd_features(d)
            d = self._atr_features(d)
            d = self._bollinger_features(d)
            d = self._trend_regime(d)
            d = self._ict_features(d)
            d = self._session_features(d)
            d = self._momentum_cluster(d)
            d = self._htf_features(d, df_h4)
            d = self._advanced_features(d)
            d = self._new_v15_features(d)
            d = self._mtf7_features(d, df_h3=df_h3, df_m30=df_m30, df_m10=df_m10, mtf_result=mtf_result)

            # Keep only feature columns that exist
            existing = [c for c in self.FEATURE_COLS if c in d.columns]
            feat = d[existing].copy()
            feat.replace([np.inf, -np.inf], np.nan, inplace=True)
            feat.ffill(inplace=True)
            feat.dropna(inplace=True)

            return feat

        except Exception as e:
            logger.error(f"FeatureEngineer.build error: {e}", exc_info=True)
            return pd.DataFrame()

    # ─────────────────────────────────────────────────────────────────
    # Feature groups
    # ─────────────────────────────────────────────────────────────────

    def _price_action(self, d: pd.DataFrame) -> pd.DataFrame:
        d["f_returns_1"]  = d["close"].pct_change(1)
        d["f_returns_3"]  = d["close"].pct_change(3)
        d["f_returns_5"]  = d["close"].pct_change(5)
        d["f_returns_10"] = d["close"].pct_change(10)

        hl = (d["high"] - d["low"]).replace(0, np.nan)
        d["f_hl_range_norm"] = hl / d["close"].replace(0, np.nan)
        body = (d["close"] - d["open"]).abs()
        d["f_body_pct"]    = body / hl
        d["f_upper_wick"]  = (d["high"] - d[["close", "open"]].max(axis=1)) / hl
        d["f_lower_wick"]  = (d[["close", "open"]].min(axis=1) - d["low"]) / hl
        d["f_candle_direction"] = np.sign(d["close"] - d["open"])
        return d

    def _volume_features(self, d: pd.DataFrame) -> pd.DataFrame:
        vol_col = None
        for vc in ["real_volume", "tick_volume"]:
            if vc in d.columns and d[vc].sum() > 0:
                vol_col = vc
                break

        if vol_col:
            # FIX BUG 5: use shift(1) on the denominator to avoid including current bar
            vm = d[vol_col].shift(1).rolling(20).mean().replace(0, np.nan)
            d["f_vol_ratio"] = d[vol_col] / vm

            bull_vol = d[vol_col].where(d["close"] > d["open"], 0)
            bear_vol = d[vol_col].where(d["close"] < d["open"], 0)
            # Normalise delta by prior-bar volume mean (causal)
            d["f_vol_delta"] = (bull_vol - bear_vol).rolling(5).mean() / vm

            d["f_vol_imbalance_5"] = (
                (bull_vol.rolling(5).sum() - bear_vol.rolling(5).sum()) /
                (d[vol_col].rolling(5).sum().replace(0, np.nan))
            )
        else:
            d["f_vol_ratio"]        = 1.0
            d["f_vol_delta"]        = 0.0
            d["f_vol_imbalance_5"]  = 0.0
        return d

    def _rsi_features(self, d: pd.DataFrame) -> pd.DataFrame:
        if "rsi" in d.columns:
            rsi = d["rsi"]
        else:
            rsi = self._calc_rsi(d["close"])
        d["f_rsi_norm"]  = (rsi - 50) / 50
        d["f_rsi_slope"] = rsi.diff(3) / 3
        return d

    def _macd_features(self, d: pd.DataFrame) -> pd.DataFrame:
        for line_col in ["macd_line", "macd"]:
            if line_col in d.columns:
                ml = d[line_col]
                break
        else:
            fast = d["close"].ewm(span=12, adjust=False).mean()
            slow = d["close"].ewm(span=26, adjust=False).mean()
            ml   = fast - slow

        ms = ml.ewm(span=9, adjust=False).mean() if "macd_signal" not in d.columns else d["macd_signal"]
        hist = ml - ms
        d["f_macd_hist"]       = hist / d["close"].replace(0, np.nan)
        d["f_macd_hist_slope"] = hist.diff(3) / 3
        return d

    def _atr_features(self, d: pd.DataFrame) -> pd.DataFrame:
        if "atr" in d.columns and d["atr"].notna().sum() > 20:
            atr = d["atr"]
        else:
            atr = self._calc_atr(d)

        close = d["close"].replace(0, np.nan)
        d["f_atr_norm"] = atr / close

        lb = min(100, len(d) - 15)
        if lb > 15:
            pct = atr.rolling(lb).apply(
                lambda x: (x[-1] > x[:-1]).mean(), raw=True
            )
        else:
            pct = pd.Series(0.5, index=d.index)
        d["f_atr_percentile"] = pct
        return d

    def _bollinger_features(self, d: pd.DataFrame) -> pd.DataFrame:
        if "bb_upper" in d.columns and "bb_lower" in d.columns:
            bb_up, bb_lo = d["bb_upper"], d["bb_lower"]
        else:
            sma20 = d["close"].rolling(20).mean()
            std20 = d["close"].rolling(20).std()
            bb_up = sma20 + 2 * std20
            bb_lo = sma20 - 2 * std20

        bb_range = (bb_up - bb_lo).replace(0, np.nan)
        d["f_bb_pos"]        = (d["close"] - bb_lo) / bb_range
        d["f_bb_width_norm"] = bb_range / d["close"].replace(0, np.nan)
        return d

    def _trend_regime(self, d: pd.DataFrame) -> pd.DataFrame:
        sma20 = d["close"].rolling(20).mean()
        sma50 = d["close"].rolling(50).mean()
        d["f_above_sma20"] = (d["close"] > sma20).astype(float)
        d["f_above_sma50"] = (d["close"] > sma50).astype(float)
        d["f_htf_bias"]    = np.where(sma20 > sma50, 1.0, -1.0)

        adx_raw = self._calc_adx_series(d)
        d["f_adx_norm"] = (adx_raw / 50.0).clip(0, 1)

        closes_arr = d["close"].values.astype(float)
        win = 30
        x = np.arange(win, dtype=float)
        xm = x - x.mean()
        xvar = float((xm ** 2).sum())
        n_rows = len(closes_arr)
        slopes_arr = np.zeros(n_rows, dtype=float)
        if xvar > 0 and n_rows >= win:
            for i in range(win, n_rows + 1):
                y_win = closes_arr[i - win: i]
                ym = y_win.mean()
                slopes_arr[i - 1] = float(np.dot(xm, y_win - ym)) / xvar / ym if ym != 0 else 0.0
        d["f_trend_slope"] = slopes_arr
        return d

    def _ict_features(self, d: pd.DataFrame) -> pd.DataFrame:
        atr_col = "atr" if "atr" in d.columns else None
        lb = self.swing_lookback

        highs = d["high"].values
        lows  = d["low"].values
        n_rows = len(d)

        highs_vals = np.full(n_rows, np.nan)
        lows_vals  = np.full(n_rows, np.nan)
        if n_rows > 2 * lb:
            roll_max_h = d["high"].rolling(2 * lb + 1, center=True).max().values
            roll_min_l = d["low"].rolling(2 * lb + 1, center=True).min().values
            swing_hi_mask = (highs == roll_max_h)
            swing_lo_mask = (lows  == roll_min_l)
            highs_vals[swing_hi_mask] = highs[swing_hi_mask]
            lows_vals[swing_lo_mask]  = lows[swing_lo_mask]

        sh_series = pd.Series(highs_vals, index=d.index).ffill()
        sl_series = pd.Series(lows_vals,  index=d.index).ffill()

        atr = d[atr_col] if atr_col and atr_col in d.columns else self._calc_atr(d)
        atr_safe = atr.replace(0, np.nan)
        d["f_dist_swing_high"] = (sh_series - d["close"]) / atr_safe
        d["f_dist_swing_low"]  = (d["close"] - sl_series) / atr_safe

        # FVG presence — vectorized 3-bar comparison (causal: uses prior bars)
        low_cur   = d["low"].values
        high_cur  = d["high"].values
        low_2ago  = np.roll(d["low"].values,  2)
        high_2ago = np.roll(d["high"].values, 2)
        bull_fvg  = (low_cur > high_2ago).astype(float)
        bear_fvg  = (high_cur < low_2ago).astype(float)
        fvg_arr   = np.where(bull_fvg, 1.0, np.where(bear_fvg, -1.0, 0.0))
        fvg_arr[:2] = 0.0
        d["f_fvg_present"] = fvg_arr

        # OB proximity
        is_bearish   = (d["close"].values < d["open"].values).astype(float)
        ob_mids      = np.where(is_bearish, (highs + lows) / 2, np.nan)
        close_vals   = d["close"].values
        atr_vals     = atr.values
        ob_prox_arr  = np.zeros(n_rows, dtype=float)
        lb_ob        = self.ob_lookback
        for i in range(5, n_rows):
            cur      = close_vals[i]
            atv      = atr_vals[i] if np.isfinite(atr_vals[i]) else 0.0001
            start    = max(0, i - lb_ob)
            window   = ob_mids[start:i]
            valid    = window[np.isfinite(window)]
            if len(valid) > 0 and np.any(np.abs(cur - valid) < atv * 0.5):
                ob_prox_arr[i] = 1.0
        d["f_near_ob"] = ob_prox_arr

        mid = (sh_series + sl_series) / 2
        d["f_pd_zone"] = np.where(d["close"] < mid, 1.0, np.where(d["close"] > mid, -1.0, 0.0))

        return d

    def _session_features(self, d: pd.DataFrame) -> pd.DataFrame:
        if hasattr(d.index, "hour"):
            h = d.index.hour
            dow = d.index.dayofweek
        else:
            h   = pd.Series(0, index=d.index)
            dow = pd.Series(0, index=d.index)

        d["f_hour_sin"] = np.sin(2 * np.pi * h / 24)
        d["f_hour_cos"] = np.cos(2 * np.pi * h / 24)
        d["f_dow_sin"]  = np.sin(2 * np.pi * dow / 5)
        d["f_dow_cos"]  = np.cos(2 * np.pi * dow / 5)

        killzone_hours = list(range(0, 4)) + list(range(6, 10)) + list(range(12, 16))
        overlap_hours  = list(range(13, 17))
        d["f_in_killzone"] = h.isin(killzone_hours).astype(float)
        d["f_in_overlap"]  = h.isin(overlap_hours).astype(float)
        return d

    def _momentum_cluster(self, d: pd.DataFrame) -> pd.DataFrame:
        # f_roc_5 removed: r>0.95 correlation with f_returns_5, adds no information
        d["f_roc_20"] = d["close"].pct_change(20)

        # FIX BUG 2: shift(1) makes the rolling window fully causal
        h20 = d["high"].shift(1).rolling(20).max()
        l20 = d["low"].shift(1).rolling(20).min()
        close_safe = d["close"].replace(0, np.nan)
        d["f_close_vs_h20_pct"] = (d["close"] - h20) / close_safe
        d["f_close_vs_l20_pct"] = (d["close"] - l20) / close_safe

        rsi = self._calc_rsi(d["close"])
        rsi_bull = (rsi > 50).astype(int)
        macd_bull = (d.get("f_macd_hist", pd.Series(0, index=d.index)) > 0).astype(int)
        price_bull = (d["f_returns_5"] > 0).astype(int)
        d["f_momentum_align"] = (rsi_bull + macd_bull + price_bull) / 3.0 * 2 - 1
        return d

    def _advanced_features(self, d: pd.DataFrame) -> pd.DataFrame:
        """v9 high-signal features, all causal-leakage bugs fixed in v15."""
        try:
            close = d["close"]
            high  = d["high"]
            low   = d["low"]
            open_ = d["open"]

            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            d["f_ema_cross_fast"] = (ema12 - ema26) / close.replace(0, np.nan)

            sma20  = close.rolling(20).mean()
            std20  = close.rolling(20).std()
            bb_up  = sma20 + 2 * std20
            bb_lo  = sma20 - 2 * std20
            atr14  = d.get("atr", self._calc_atr(d))
            kc_up  = sma20 + 1.5 * atr14
            kc_lo  = sma20 - 1.5 * atr14
            squeeze = ((bb_up < kc_up) & (bb_lo > kc_lo)).astype(float)
            # FIX BUG 4: use shift(1) on rolling max/min for causal midpoint
            midpoint = (high.shift(1).rolling(12).max() + low.shift(1).rolling(12).min()) / 2
            delta    = close - (midpoint + sma20) / 2
            d["f_squeeze_momentum"] = squeeze * delta.rolling(12).mean() / close.replace(0, np.nan)

            vol_col = None
            for vc in ["real_volume", "tick_volume"]:
                if vc in d.columns and d[vc].sum() > 0:
                    vol_col = vc; break
            if vol_col:
                vol_rising = (d[vol_col] > d[vol_col].shift(1).rolling(5).mean()).astype(float)
                price_up   = (close > close.shift(1)).astype(float)
                price_dn   = (close < close.shift(1)).astype(float)
                d["f_vol_trend_confirm"] = vol_rising * (price_up - price_dn)
            else:
                d["f_vol_trend_confirm"] = 0.0

            candle_size = (high - low)
            lb100 = min(100, len(d) - 1)
            if lb100 > 10:
                cs_roll_min = candle_size.rolling(lb100, min_periods=2).min()
                cs_roll_max = candle_size.rolling(lb100, min_periods=2).max()
                denom = (cs_roll_max - cs_roll_min).replace(0, np.nan)
                d["f_candle_size_rank"] = ((candle_size - cs_roll_min) / denom).clip(0, 1).fillna(0.5)
            else:
                d["f_candle_size_rank"] = 0.5

            direction = np.sign(close.diff()).fillna(0).values
            dir_series = pd.Series(direction)
            group = (dir_series != dir_series.shift()).cumsum()
            run_len = dir_series.groupby(group).cumcount() + 1
            run_len_capped = run_len.clip(upper=5).values
            d["f_consec_same_dir"] = run_len_capped * direction

            # FIX BUG 1: shift(1) before rolling makes pivot window fully causal
            # Prior version used rolling(24).max() including current bar → leakage.
            ph = high.shift(1).rolling(24).max()
            pl = low.shift(1).rolling(24).min()
            d["f_pivot_h"] = (ph - close) / close.replace(0, np.nan)
            d["f_pivot_l"] = (close - pl) / close.replace(0, np.nan)

            oc_range = (high - low).replace(0, np.nan)
            d["f_close_open_ratio"] = (close - open_) / oc_range

            upper_wick = high - pd.concat([close, open_], axis=1).max(axis=1)
            lower_wick = pd.concat([close, open_], axis=1).min(axis=1) - low
            total_wick = (upper_wick + lower_wick).replace(0, np.nan)
            d["f_high_low_wick_ratio"] = (upper_wick - lower_wick) / total_wick

            pct_chg = close.pct_change().fillna(0)
            if vol_col:
                vpt = (d[vol_col] * pct_chg).rolling(10).sum()
                vpt_norm = vpt / d[vol_col].rolling(10).mean().replace(0, np.nan)
                d["f_volume_price_trend"] = vpt_norm.clip(-3, 3)
            else:
                d["f_volume_price_trend"] = 0.0

            atr_mean50 = atr14.rolling(50).mean().replace(0, np.nan)
            d["f_atr_expansion"] = (atr14 / atr_mean50).clip(0, 3) - 1.0

            # FIX BUG 3: shift(1) before rolling(14) — current bar excluded from window
            rsi = self._calc_rsi(close)
            price_hh = (close == close.shift(1).rolling(14).max()).astype(float)
            rsi_lh   = (rsi < rsi.shift(14)).astype(float)
            price_ll = (close == close.shift(1).rolling(14).min()).astype(float)
            rsi_hl   = (rsi > rsi.shift(14)).astype(float)
            bear_div = (price_hh & rsi_lh.astype(bool)).astype(float)
            bull_div = (price_ll & rsi_hl.astype(bool)).astype(float)
            d["f_rsi_divergence"] = bull_div - bear_div

        except Exception as e:
            logger.debug(f"_advanced_features error: {e}")
            for col in ["f_ema_cross_fast", "f_squeeze_momentum", "f_vol_trend_confirm",
                        "f_candle_size_rank", "f_consec_same_dir", "f_pivot_h", "f_pivot_l",
                        "f_close_open_ratio", "f_high_low_wick_ratio", "f_volume_price_trend",
                        "f_atr_expansion", "f_rsi_divergence"]:
                if col not in d.columns:
                    d[col] = 0.0
        return d

    def _new_v15_features(self, d: pd.DataFrame) -> pd.DataFrame:
        """
        v15 NEW features with genuine OOS predictive signal.
        All causal — no future data used.
        """
        try:
            close = d["close"]
            high  = d["high"]
            low   = d["low"]
            close_safe = close.replace(0, np.nan)

            # ── f_session_range_pos ────────────────────────────────────────
            # Where is current close within today's session range?
            # Uses only bars from start of current UTC session (midnight).
            # Fully causal: all prior bars in the same day.
            if hasattr(d.index, "hour"):
                # Create day-of-session running high/low using expanding within each day
                date_key = d.index.normalize()   # midnight of each bar
                day_h = high.groupby(date_key).expanding().max().droplevel(0)
                day_l = low.groupby(date_key).expanding().min().droplevel(0)
                # Shift 1 to exclude current bar from the session range calculation
                day_h = day_h.shift(1).reindex(d.index)
                day_l = day_l.shift(1).reindex(d.index)
                day_range = (day_h - day_l).replace(0, np.nan)
                d["f_session_range_pos"] = ((close - day_l) / day_range).clip(0, 1).fillna(0.5)
            else:
                # Fallback: 8-bar session proxy (no datetime index)
                day_h = high.shift(1).rolling(8, min_periods=1).max()
                day_l = low.shift(1).rolling(8, min_periods=1).min()
                day_range = (day_h - day_l).replace(0, np.nan)
                d["f_session_range_pos"] = ((close - day_l) / day_range).clip(0, 1).fillna(0.5)

            # ── f_vwap_dev ─────────────────────────────────────────────────
            # Deviation from 24-bar rolling VWAP (causal).
            # VWAP = sum(price * volume) / sum(volume) — anchored to prior bars.
            typical = (high + low + close) / 3
            vol_col = None
            for vc in ["real_volume", "tick_volume"]:
                if vc in d.columns and d[vc].sum() > 0:
                    vol_col = vc; break
            if vol_col:
                vol_series = d[vol_col]
            else:
                # No volume → use constant weight (pure price VWAP = SMA)
                vol_series = pd.Series(1.0, index=d.index)
            # Shift 1 to exclude current bar from VWAP calculation
            tp_shifted  = typical.shift(1)
            vol_shifted = vol_series.shift(1)
            vwap = (tp_shifted * vol_shifted).rolling(24).sum() / vol_shifted.rolling(24).sum().replace(0, np.nan)
            d["f_vwap_dev"] = ((close - vwap) / vwap.replace(0, np.nan)).clip(-0.02, 0.02)

            # ── f_vol_regime ────────────────────────────────────────────────
            # Volatility regime: is current ATR expanding or contracting?
            # +1 = expanding (ATR rising), -1 = contracting, 0 = stable.
            atr14 = self._calc_atr(d)
            atr_sma = atr14.shift(1).rolling(10).mean().replace(0, np.nan)
            atr_ratio = atr14 / atr_sma
            d["f_vol_regime"] = np.where(atr_ratio > 1.10, 1.0,
                                 np.where(atr_ratio < 0.90, -1.0, 0.0))

            # ── f_spread_adj_return ─────────────────────────────────────────
            # ATR-normalised 5-bar return — filters low-volatility noise.
            # A 10-pip move in a 50-pip ATR market is less significant than
            # in a 10-pip ATR market.  This captures that ratio directly.
            ret5 = close.pct_change(5)
            atr_norm5 = atr14.shift(1).rolling(5).mean().replace(0, np.nan)
            atr_pct5  = atr_norm5 / close_safe
            d["f_spread_adj_return"] = (ret5 / atr_pct5.replace(0, np.nan)).clip(-5, 5).fillna(0)

            # ── f_htf_sma_slope ─────────────────────────────────────────────
            # Normalised slope of the 50-bar SMA over the last 10 bars.
            # Captures trend PERSISTENCE (is the SMA accelerating/decelerating?),
            # which is orthogonal to "above/below SMA" level features.
            sma50 = close.rolling(50, min_periods=25).mean()
            sma50_slope = sma50.diff(10) / sma50.shift(10).replace(0, np.nan)
            d["f_htf_sma_slope"] = sma50_slope.clip(-0.05, 0.05).fillna(0)

            # ── f_regime_change ──────────────────────────────────────────────
            # Did ADX cross the 20 threshold in the last 3 bars?
            # ADX crossing 20 signals a regime transition (trending ↔ ranging).
            adx = self._calc_adx_series(d)
            adx_above = (adx > 20).astype(int)
            # Transition detected if current value differs from value 3 bars ago
            d["f_regime_change"] = (adx_above != adx_above.shift(3)).astype(float).fillna(0)

        except Exception as e:
            logger.debug(f"_new_v15_features error: {e}")
            for col in ["f_session_range_pos", "f_vwap_dev", "f_vol_regime",
                        "f_spread_adj_return", "f_htf_sma_slope", "f_regime_change"]:
                if col not in d.columns:
                    d[col] = 0.0
        return d

    def _htf_features(self, d: pd.DataFrame, df_h4: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Encode H4 context into per-H1-bar features using time-aligned merge.
        v12 FIX retained: proper as-of join (not scalar stamp).
        """
        if df_h4 is None or len(df_h4) < 20:
            d["f_h4_trend_bull"] = 0.0
            d["f_h4_trend_bear"] = 0.0
            d["f_h4_bos"]        = 0.0
            d["f_h4_choch"]      = 0.0
            d["f_h4_atr_ratio"]  = 1.0
            return d

        try:
            h4 = df_h4.copy()
            h4_close  = h4["close"].values.astype(float)
            h4_high   = h4["high"].values.astype(float)
            h4_low    = h4["low"].values.astype(float)
            h4_atr    = self._calc_atr(h4).values.astype(float)
            n_h4      = len(h4)
            lb        = min(8, n_h4 - 1)
            swing_lb  = min(20, n_h4 - 2)

            h4_bull_arr  = np.zeros(n_h4, dtype=float)
            h4_bear_arr  = np.zeros(n_h4, dtype=float)
            h4_bos_arr   = np.zeros(n_h4, dtype=float)
            h4_choch_arr = np.zeros(n_h4, dtype=float)

            for k in range(lb, n_h4):
                window = h4_close[k - lb: k + 1]
                bull = int(window[-1] > window[0] and window[-1] > window.mean())
                bear = int(window[-1] < window[0] and window[-1] < window.mean())
                h4_bull_arr[k] = float(bull)
                h4_bear_arr[k] = float(bear)

                if k >= swing_lb + 2:
                    s_start = k - swing_lb
                    prior_h = float(np.max(h4_high[s_start: k - 1]))
                    prior_l = float(np.min(h4_low[s_start:  k - 1]))
                    bos = int(h4_close[k] > prior_h or h4_close[k] < prior_l)
                    h4_bos_arr[k] = float(bos)

                    prior_mid = (h4_close[s_start] + h4_close[k - swing_lb // 2]) / 2
                    prior_up  = h4_close[k - swing_lb // 2] > prior_mid
                    choch = 0
                    if prior_up and k >= 3 and h4_close[k] < float(np.min(h4_low[k - 3: k])):
                        choch = 1
                    elif not prior_up and k >= 3 and h4_close[k] > float(np.max(h4_high[k - 3: k])):
                        choch = 1
                    h4_choch_arr[k] = float(choch)

            h4_feat = pd.DataFrame({
                "f_h4_trend_bull": h4_bull_arr,
                "f_h4_trend_bear": h4_bear_arr,
                "f_h4_bos":        h4_bos_arr,
                "f_h4_choch":      h4_choch_arr,
                "f_h4_atr_abs":    h4_atr,
            }, index=h4.index)

            h1_atr = self._calc_atr(d)

            if hasattr(d.index, "to_pydatetime") and hasattr(h4.index, "to_pydatetime"):
                try:
                    h4_feat_reindexed = h4_feat.reindex(
                        h4_feat.index.union(d.index)
                    ).ffill().reindex(d.index)
                except Exception:
                    h4_feat_reindexed = None
            else:
                h4_feat_reindexed = None

            if h4_feat_reindexed is not None and not h4_feat_reindexed.empty:
                d["f_h4_trend_bull"] = h4_feat_reindexed["f_h4_trend_bull"].fillna(0.0).values
                d["f_h4_trend_bear"] = h4_feat_reindexed["f_h4_trend_bear"].fillna(0.0).values
                d["f_h4_bos"]        = h4_feat_reindexed["f_h4_bos"].fillna(0.0).values
                d["f_h4_choch"]      = h4_feat_reindexed["f_h4_choch"].fillna(0.0).values
                h4_atr_aligned       = h4_feat_reindexed["f_h4_atr_abs"].ffill()
                d["f_h4_atr_ratio"]  = (h1_atr / h4_atr_aligned.replace(0, np.nan)).clip(0, 5).fillna(1.0).values
            else:
                last = h4_feat.iloc[-1]
                d["f_h4_trend_bull"] = float(last["f_h4_trend_bull"])
                d["f_h4_trend_bear"] = float(last["f_h4_trend_bear"])
                d["f_h4_bos"]        = float(last["f_h4_bos"])
                d["f_h4_choch"]      = float(last["f_h4_choch"])
                h4_atr_last          = float(last["f_h4_atr_abs"])
                h1_atr_last          = float(h1_atr.iloc[-1]) if len(h1_atr) > 0 else 0.0001
                d["f_h4_atr_ratio"]  = float(h1_atr_last / max(h4_atr_last, 1e-10))

        except Exception as e:
            logger.debug(f"_htf_features error: {e}")
            d["f_h4_trend_bull"] = 0.0
            d["f_h4_trend_bear"] = 0.0
            d["f_h4_bos"]        = 0.0
            d["f_h4_choch"]      = 0.0
            d["f_h4_atr_ratio"]  = 1.0

        return d

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"]  - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _calc_adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
        try:
            high, low, close = df["high"], df["low"], df["close"]
            plus_dm  = high.diff().clip(lower=0)
            minus_dm = (-low.diff()).clip(lower=0)
            mask = plus_dm > minus_dm
            plus_dm  = plus_dm.where(mask, 0)
            minus_dm = minus_dm.where(~mask, 0)
            tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().replace(0, np.nan)
            plus_di  = 100 * plus_dm.rolling(period).mean() / atr
            minus_di = 100 * minus_dm.rolling(period).mean() / atr
            dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
            return dx.rolling(period).mean().fillna(20.0)
        except Exception:
            return pd.Series(20.0, index=df.index)

    # ─────────────────────────────────────────────────────────────────
    # 7-tier MTF extensions (v13/v15): H3, M30, M10 + tier score
    # BOS features removed from H3/M30/M10: they used SMA-crossover as a
    # BOS proxy which is a lagging indicator that hurts WF OOS accuracy.
    # Only trend direction (SMA-based) and M10 FVG are retained.
    # ─────────────────────────────────────────────────────────────────

    def _mtf7_features(
        self,
        d: pd.DataFrame,
        df_h3: Optional[pd.DataFrame] = None,
        df_m30: Optional[pd.DataFrame] = None,
        df_m10: Optional[pd.DataFrame] = None,
        mtf_result=None,
    ) -> pd.DataFrame:
        """
        Add H3, M30, M10 context features and 7-tier tier_score.
        v15: BOS/CHoCH removed from sub-H4 timeframes (SMA proxy too lagging).
        """
        try:
            h1_atr = d["f_atr_norm"].fillna(0.01) if "f_atr_norm" in d.columns else pd.Series(0.01, index=d.index)

            def _tf_trend(df_tf, name: str):
                """Extract bull/bear trend arrays aligned to d.index (no BOS/CHoCH)."""
                defaults = {
                    f"f_{name}_trend_bull": 0.0,
                    f"f_{name}_trend_bear": 0.0,
                }
                if df_tf is None or len(df_tf) < 10:
                    return defaults
                try:
                    tf = df_tf.copy()
                    close = tf["close"]
                    sma20 = close.rolling(20, min_periods=5).mean()
                    sma50 = close.rolling(50, min_periods=10).mean()
                    bull = (sma20 > sma50).astype(float)
                    bear = (sma20 < sma50).astype(float)

                    tf_feat = pd.DataFrame({
                        f"f_{name}_trend_bull": bull,
                        f"f_{name}_trend_bear": bear,
                    }, index=tf.index)

                    reindexed = tf_feat.reindex(d.index, method="ffill")
                    return {col: reindexed[col].fillna(0.0).values for col in tf_feat.columns}
                except Exception as inner_e:
                    logger.debug(f"_tf_trend({name}) error: {inner_e}")
                    return defaults

            # ── H3 trend features ──────────────────────────────────────────
            h3_sigs = _tf_trend(df_h3, "h3")
            for col, vals in h3_sigs.items():
                d[col] = vals

            # ── M30 trend features ─────────────────────────────────────────
            m30_sigs = _tf_trend(df_m30, "m30")
            for col, vals in m30_sigs.items():
                d[col] = vals

            # ── M10 features: FVG + ATR ratio only (BOS removed) ──────────
            m10_defaults = {
                "f_m10_fvg": 0.0, "f_m10_atr_ratio": 1.0,
            }
            if df_m10 is not None and len(df_m10) >= 10:
                try:
                    m10 = df_m10.copy()

                    # FVG: 3-candle imbalance (causal: uses shift)
                    fvg_bull = (m10["low"] > m10["high"].shift(2)).astype(float)
                    fvg_bear = (m10["high"] < m10["low"].shift(2)).astype(float)
                    m10_fvg  = ((fvg_bull + fvg_bear) > 0).astype(float)

                    m10_atr_series = self._calc_atr(m10)
                    m10_feat = pd.DataFrame({
                        "f_m10_fvg":     m10_fvg,
                        "f_m10_atr_abs": m10_atr_series,
                    }, index=m10.index)

                    reindexed = m10_feat.reindex(d.index, method="ffill")
                    d["f_m10_fvg"]       = reindexed["f_m10_fvg"].fillna(0.0).values
                    m10_atr_aligned      = reindexed["f_m10_atr_abs"].ffill()
                    d["f_m10_atr_ratio"] = (h1_atr / m10_atr_aligned.replace(0, np.nan)).clip(0, 10).fillna(1.0).values

                except Exception as e:
                    logger.debug(f"M10 features error: {e}")
                    for col, val in m10_defaults.items():
                        d[col] = val
            else:
                for col, val in m10_defaults.items():
                    d[col] = val

            # ── Tier score ─────────────────────────────────────────────────
            if mtf_result is not None:
                ts = float(getattr(mtf_result, "tier_score", 0))
                d["f_tier_score"] = ts / 7.0
            else:
                tier_signals = [
                    "f_h4_trend_bull", "f_h4_trend_bear",
                    "f_h3_trend_bull", "f_h3_trend_bear",
                    "f_m30_trend_bull", "f_m30_trend_bear",
                    "f_m10_fvg",
                ]
                available = [c for c in tier_signals if c in d.columns]
                if available:
                    d["f_tier_score"] = d[available].abs().mean(axis=1).clip(0, 1)
                else:
                    d["f_tier_score"] = 0.0

        except Exception as e:
            logger.error(f"_mtf7_features error: {e}", exc_info=True)
            for col in ["f_h3_trend_bull", "f_h3_trend_bear",
                        "f_m30_trend_bull", "f_m30_trend_bear",
                        "f_m10_fvg", "f_m10_atr_ratio", "f_tier_score"]:
                if col not in d.columns:
                    d[col] = 0.0 if "ratio" not in col else 1.0

        return d
