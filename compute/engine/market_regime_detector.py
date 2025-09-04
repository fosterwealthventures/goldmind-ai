"""
market_regime_detector.py — GoldMIND AI
--------------------------------------
Lightweight, dependency-minimal regime detection that plays nicely with the rest
of your stack. No external TA libs required.

Features:
- Multi-timeframe analysis (short/medium/long windows)
- Trend, volatility (ATR-like), and momentum heuristics
- Volume confirmation & liquidity sanity checks
- Robust classification into: trending (bull/bear), volatile, sideways
- Confidence scoring and suggested portfolio adjustments
- SQLite-friendly, but stateless (no DB required)
- Dashboard helpers for compact summaries

Public surface:
    mrd = MarketRegimeDetector(config, data_framework=None, lstm=None)
    regime = await mrd.detect_for_symbol("XAU/USD", days=180)       # pulls via FinancialDataFramework if provided
    regime2 = await mrd.detect_current_regime(df)                   # pass your own OHLCV DataFrame
    card = MarketRegimeDetector.to_dashboard_block(regime)          # -> {regime, sentiment, confidence, ...}

Expected OHLCV schema (normalized):
    Index: DatetimeIndex (ascending)
    Columns: Open, High, Low, Close, Volume (Price optional; Close used for Price)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import math
import numpy as np
import pandas as pd

# Optional imports wired with soft fallbacks
try:
    from financial_data_framework import FinancialDataFramework
except Exception:  # pragma: no cover
    FinancialDataFramework = None  # type: ignore

try:
    from lstm_temporal_analysis import LSTMTemporalAnalysis
except Exception:  # pragma: no cover
    LSTMTemporalAnalysis = None  # type: ignore

log = logging.getLogger("goldmind.market_regime")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RegimeConfig:
    short_window: int = 20
    medium_window: int = 50
    long_window: int = 120
    vol_window: int = 14
    atr_k: float = 1.0                     # ATR-like sensitivity
    trend_threshold: float = 0.04          # 4% move over short window -> trend
    bearish_threshold: float = -0.04
    vol_threshold: float = 0.03            # stdev(close)/mean(close)
    min_rows: int = 60                     # minimal data length for stable signals
    pos_size_trending: float = 1.1
    pos_size_volatile: float = 0.7
    pos_size_sideways: float = 1.0
    # sentiment blending with LSTM if provided
    lstm_weight: float = 0.25


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class MarketRegimeDetector:
    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        data_framework: Optional["FinancialDataFramework"] = None,
        lstm: Optional["LSTMTemporalAnalysis"] = None,
    ) -> None:
        self.cfg = RegimeConfig(**(config or {}))
        self.fdf = data_framework
        self.lstm = lstm

    # ---------------- Public entry points ----------------

    async def detect_for_symbol(self, symbol: str, days: int = 180, interval: str = "1day") -> Dict[str, Any]:
        """
        Pull data from FinancialDataFramework if available and run detection.
        """
        if not self.fdf:
            log.warning("FinancialDataFramework not provided; generating synthetic series for %s", symbol)
            # minimal synthetic series as fallback
            idx = pd.date_range(end=datetime.utcnow(), periods=days, freq="D")
            base = np.linspace(100, 110, len(idx))
            df = pd.DataFrame({"Open": base, "High": base * 1.002, "Low": base * 0.998, "Close": base, "Volume": 100000}, index=idx)
        else:
            df = await self.fdf.get_processed_data(symbol, days_back=days, interval=interval)
            if df is None or df.empty:
                raise RuntimeError(f"No OHLCV data for {symbol}.")
            # Normalize expected columns
            df = df.rename(columns=str.capitalize)
            if "Close" not in df.columns and "Price" in df.columns:
                df["Close"] = df["Price"]
            if "Volume" not in df.columns:
                df["Volume"] = 0
        return await self.detect_current_regime(df, symbol=symbol)

    async def detect_current_regime(self, market_data: pd.DataFrame, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute regime and return a structured dict.
        """
        if market_data is None or market_data.empty or len(market_data) < self.cfg.min_rows:
            return {
                "regime": "unknown",
                "sentiment": "unknown",
                "confidence": 0.0,
                "recommended_adjustments": {},
                "features": {},
                "symbol": symbol or "",
                "rows": 0,
                "note": "insufficient data",
            }

        df = market_data.copy()
        # ensure datetime index and ascending order
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass
        df = df.sort_index()

        features = self._compute_features(df)
        regime, sentiment, conf = self._classify(features)

        # Optionally blend sentiment with LSTM directional suggestion
        if self.lstm and hasattr(self.lstm, "predict_stock_price") and symbol:
            try:
                lstm_out = await self.lstm.predict_stock_price(symbol)
                lstm_dir = str(lstm_out.get("action", "HOLD")).upper()
                lstm_conf = float(lstm_out.get("confidence", 0.5))
                blend = self.cfg.lstm_weight * lstm_conf
                # nudge confidence/sentiment with LSTM
                if lstm_dir == "BUY":
                    sentiment = self._blend_sentiment(sentiment, "bullish", blend)
                elif lstm_dir == "SELL":
                    sentiment = self._blend_sentiment(sentiment, "bearish", blend)
                conf = float(min(1.0, conf + blend * 0.15))
                features["lstm_hint"] = {"dir": lstm_dir, "conf": lstm_conf}
            except Exception as e:
                log.warning("LSTM blend skipped: %s", e)

        adjustments = self._recommended_adjustments(regime)
        out = {
            "regime": regime,
            "sentiment": sentiment,
            "confidence": conf,
            "recommended_adjustments": adjustments,
            "features": features,
            "symbol": symbol or "",
            "rows": int(len(df)),
            "generated_at": datetime.utcnow().isoformat(),
        }
        return out

    # ---------------- Internals ----------------

    def _compute_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        cfg = self.cfg
        close = df["Close"].astype(float).copy()

        # returns & momentum
        ret1 = close.pct_change().fillna(0.0)
        mom_s = (close / close.shift(cfg.short_window) - 1.0).iloc[-1]
        mom_m = (close / close.shift(cfg.medium_window) - 1.0).iloc[-1] if len(close) >= cfg.medium_window + 1 else 0.0
        mom_l = (close / close.shift(cfg.long_window) - 1.0).iloc[-1] if len(close) >= cfg.long_window + 1 else 0.0

        # moving averages
        ma_s = close.rolling(cfg.short_window).mean().iloc[-1]
        ma_m = close.rolling(cfg.medium_window).mean().iloc[-1] if len(close) >= cfg.medium_window else ma_s
        ma_l = close.rolling(cfg.long_window).mean().iloc[-1] if len(close) >= cfg.long_window else ma_m

        # volatility: stdev/mean and ATR-like
        vol_pct = float(ret1.rolling(cfg.vol_window).std().iloc[-1] * math.sqrt(252)) if len(ret1) >= cfg.vol_window else float(ret1.std()*math.sqrt(252))
        if all(k in df.columns for k in ("High", "Low")):
            tr = (df["High"] - df["Low"]).abs()
            atr = tr.rolling(cfg.vol_window).mean().iloc[-1] if len(tr) >= cfg.vol_window else tr.mean()
        else:
            atr = float(close.rolling(cfg.vol_window).std().iloc[-1]) if len(close) >= cfg.vol_window else float(close.std())
        atr_pct = float(atr / max(1e-9, close.iloc[-1]))

        # volume confirmation
        if "Volume" in df.columns:
            v_ma = float(df["Volume"].rolling(cfg.short_window).mean().iloc[-1])
            v_last = float(df["Volume"].iloc[-1])
            vol_confirm = 1.0 if v_ma == 0 else float(v_last / max(1.0, v_ma))
        else:
            vol_confirm = 1.0

        # trend slope (OLS on last medium window)
        try:
            n = min(cfg.medium_window, len(close))
            y = close.iloc[-n:].values
            x = np.arange(n, dtype=float)
            x = (x - x.mean()) / (x.std() + 1e-9)
            slope = float(np.dot(x, y - y.mean()) / (n - 1) / (y.std() + 1e-9))
        except Exception:
            slope = 0.0

        return {
            "momentum": {"short": float(mom_s), "medium": float(mom_m), "long": float(mom_l)},
            "moving_averages": {"short": float(ma_s), "medium": float(ma_m), "long": float(ma_l)},
            "volatility": {"stdev_annualized": float(vol_pct), "atr_pct": float(atr_pct)},
            "volume_confirm": float(vol_confirm),
            "trend_slope": float(slope),
            "price": float(close.iloc[-1]),
        }

    def _classify(self, f: Dict[str, Any]) -> tuple[str, str, float]:
        cfg = self.cfg
        mom_s = f["momentum"]["short"]
        vol_pct = f["volatility"]["stdev_annualized"]
        atr_pct = f["volatility"]["atr_pct"]
        slope = f.get("trend_slope", 0.0)
        vol_confirm = f.get("volume_confirm", 1.0)

        # Primary trend decision using short momentum
        if mom_s >= cfg.trend_threshold:
            regime = "trending"
            sentiment = "bullish"
        elif mom_s <= cfg.bearish_threshold:
            regime = "trending"
            sentiment = "bearish"
        else:
            # Volatility vs sideways
            if vol_pct > cfg.vol_threshold or atr_pct > cfg.vol_threshold:
                regime = "volatile"
                sentiment = "uncertain"
            else:
                regime = "sideways"
                sentiment = "neutral"

        # Confidence heuristic
        base_conf = 0.6 if regime == "trending" else 0.55 if regime == "sideways" else 0.5
        # boost confidence if volume confirms and slope aligns
        if regime == "trending":
            align = 1 if (sentiment == "bullish" and slope > 0) or (sentiment == "bearish" and slope < 0) else -1
            base_conf += 0.1 * max(0.0, align) + 0.05 * min(2.0, vol_confirm)
        elif regime == "sideways":
            base_conf += 0.05 * (1.5 - min(1.5, vol_confirm))
        else:  # volatile
            base_conf -= 0.05 * min(2.0, vol_confirm)

        return regime, sentiment, float(max(0.0, min(1.0, base_conf)))

    def _blend_sentiment(self, a: str, b: str, weight: float) -> str:
        # simple blend: if strong weight, adopt b; else keep a
        return b if weight >= 0.15 else a

    def _recommended_adjustments(self, regime: str) -> Dict[str, Any]:
        if regime == "trending":
            return {"position_size_multiplier": self.cfg.pos_size_trending, "preferred_pathways": ["ANALYTICAL"]}
        if regime == "volatile":
            return {"position_size_multiplier": self.cfg.pos_size_volatile, "preferred_pathways": ["LSTM_TEMPORAL"]}
        return {"position_size_multiplier": self.cfg.pos_size_sideways, "preferred_pathways": ["ANALYTICAL", "LSTM_TEMPORAL"]}

    # ---------------- Dashboard helper ----------------

    @staticmethod
    def to_dashboard_block(regime: Dict[str, Any]) -> Dict[str, Any]:
        vol = regime.get("features", {}).get("volatility", {})
        vol_label = "High" if (vol.get("stdev_annualized", 0) > 0.3 or vol.get("atr_pct", 0) > 0.03) else "Low/Moderate"
        return {
            "regime": str(regime.get("regime", "unknown")).title(),
            "sentiment": str(regime.get("sentiment", "unknown")).title(),
            "confidence": float(regime.get("confidence", 0.0)),
            "volatility": vol_label,
            "note": f"{regime.get('rows', 0)} bars analyzed; pos-size x{regime.get('recommended_adjustments',{}).get('position_size_multiplier',1.0)}",
            "symbol": regime.get("symbol", ""),
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def _demo():
        # Synthetic example
        idx = pd.date_range(end=datetime.utcnow(), periods=180, freq="D")
        prices = np.cumsum(np.random.normal(0, 0.3, len(idx))) + 100
        df = pd.DataFrame({
            "Open": prices + np.random.normal(0, 0.1, len(idx)),
            "High": prices + np.random.uniform(0.1, 0.5, len(idx)),
            "Low":  prices - np.random.uniform(0.1, 0.5, len(idx)),
            "Close": prices,
            "Volume": np.random.randint(50_000, 120_000, len(idx)),
        }, index=idx)

        mrd = MarketRegimeDetector({})
        out = await mrd.detect_current_regime(df, symbol="XAU/USD")
        print("Regime:", out)
        print("Card:", MarketRegimeDetector.to_dashboard_block(out))

    asyncio.run(_demo())
