"""
analytical_pathway.py — GoldMIND AI
----------------------------------
Rule-based technical pathway aligned with new dashboard/API shapes.
- SMA(20/50) crossover + momentum/volatility context
- Deterministic targets/SL from last price (no RNG)
- Returns a normalized recommendation block
- Includes dashboard mappers (card/summary)

Public surface:
    ap = AnalyticalPathway(config)
    rec = ap.get_recommendation(df)  # df columns: Close [+ Open, High, Low, Volume optional]
    card = AnalyticalPathway.to_recommendation_card(rec, symbol="XAU/USD")
    summary = AnalyticalPathway.to_summary_block(rec, symbol="XAU/USD")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


log = logging.getLogger("goldmind.analytical")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


@dataclass
class APConfig:
    short_ma: int = 20
    long_ma: int = 50
    min_rows: int = 60
    buy_tp_pct: float = 0.04        # +4% target for BUY
    sell_tp_pct: float = -0.03      # -3% target for SELL (relative to last price)
    buy_sl_pct: float = -0.04       # -4% stop for BUY
    sell_sl_pct: float = 0.03       # +3% stop for SELL
    default_position_size: float = 0.15
    min_confidence: float = 0.5
    cross_confidence: float = 0.8
    hold_confidence: float = 0.55


class AnalyticalPathway:
    """
    Moving-average crossover with light momentum/volatility context.
    Produces a dashboard-ready recommendation dict.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.cfg = APConfig(**(config or {}))
        log.info("AnalyticalPathway initialized: short=%s long=%s", self.cfg.short_ma, self.cfg.long_ma)

    # ---------------- Core logic ----------------

    def get_recommendation(self, processed_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Parameters
        ----------
        processed_data : pd.DataFrame
            Expected columns: Close (float). Optional: Open/High/Low/Volume.
            Index should be ascending DatetimeIndex; method will attempt to coerce.

        Returns
        -------
        dict
            {
                "action": "BUY|SELL|HOLD",
                "confidence": float [0..1],
                "target_price": float,
                "stop_loss": float,
                "position_size": float,
                "reasoning": str,
                "risk_score": float [0..1]
            }
        """
        df = self._normalize(processed_data)
        if df.empty or len(df) < self.cfg.min_rows:
            return self._empty("Insufficient data for technical analysis.")

        # Indicators
        df["SMA_S"] = df["Close"].rolling(self.cfg.short_ma).mean()
        df["SMA_L"] = df["Close"].rolling(self.cfg.long_ma).mean()
        df["RET1"] = df["Close"].pct_change().fillna(0.0)
        df["VOL_PCT"] = df["RET1"].rolling(20).std().fillna(0.0)  # simple proxy

        # Latest values
        last = float(df["Close"].iloc[-1])
        sma_s, sma_l = float(df["SMA_S"].iloc[-1]), float(df["SMA_L"].iloc[-1])
        prev_s, prev_l = float(df["SMA_S"].iloc[-2]), float(df["SMA_L"].iloc[-2])
        vol_pct = float(df["VOL_PCT"].iloc[-1])

        # Decision
        action = "HOLD"
        confidence = self.cfg.hold_confidence
        reason = "No confirmed crossover."

        if np.isfinite(sma_s) and np.isfinite(sma_l) and np.isfinite(prev_s) and np.isfinite(prev_l):
            # Golden/Death cross detection
            if sma_s > sma_l and prev_s <= prev_l:
                action = "BUY"
                confidence = self.cfg.cross_confidence
                reason = "Golden Cross: SMA(short) crossed above SMA(long)."
            elif sma_s < sma_l and prev_s >= prev_l:
                action = "SELL"
                confidence = self.cfg.cross_confidence
                reason = "Death Cross: SMA(short) crossed below SMA(long)."
            else:
                # momentum tilt
                momentum = float(df["Close"].iloc[-1] / df["Close"].iloc[-20] - 1.0) if len(df) >= 21 else 0.0
                if momentum > 0.02:
                    action, reason = "BUY", "Mild positive momentum tilt; no cross."
                    confidence = 0.62
                elif momentum < -0.02:
                    action, reason = "SELL", "Mild negative momentum tilt; no cross."
                    confidence = 0.62

        # Targets & stops (deterministic percentages from last price)
        if action == "BUY":
            target = last * (1.0 + abs(self.cfg.buy_tp_pct))
            stop = last * (1.0 + self.cfg.buy_sl_pct)
        elif action == "SELL":
            target = last * (1.0 + self.cfg.sell_tp_pct)  # negative pct lowers target
            stop = last * (1.0 + abs(self.cfg.sell_sl_pct))  # protective stop above price
        else:
            target = last
            stop = last * 0.99  # tight stop placeholder

        # Risk score heuristic (higher vol → higher risk)
        risk = float(min(0.9, max(0.1, vol_pct * 10)))

        return {
            "action": action,
            "confidence": float(max(self.cfg.min_confidence, min(1.0, confidence))),
            "target_price": float(round(target, 4)),
            "stop_loss": float(round(stop, 4)),
            "position_size": float(self.cfg.default_position_size),
            "reasoning": reason,
            "risk_score": risk,
        }

    # ---------------- Dashboard mappers ----------------

    @staticmethod
    def to_recommendation_card(rec: Dict[str, Any], symbol: str = "") -> Dict[str, Any]:
        return {
            "action": str(rec.get("action", "HOLD")).upper(),
            "confidence": float(rec.get("confidence", 0.5)),
            "entry": None,
            "target": float(rec.get("target_price", 0.0)),
            "note": rec.get("reasoning", ""),
            "symbol": symbol,
        }

    @staticmethod
    def to_summary_block(rec: Dict[str, Any], symbol: str = "") -> Dict[str, Any]:
        action = str(rec.get("action", "HOLD")).upper()
        pc = float(rec.get("confidence", 0.5)) * 100.0
        if action == "BUY" and pc >= 65:
            regime = "Bullish Trend"
        elif action == "SELL" and pc >= 65:
            regime = "Bearish Trend"
        else:
            regime = "Neutral / Range"
        # Approximate volatility label from risk_score
        r = float(rec.get("risk_score", 0.5))
        volatility = "High" if r >= 0.7 else "Moderate" if r >= 0.4 else "Low"
        return {"regime": regime, "volatility": volatility, "note": rec.get("reasoning", ""), "symbol": symbol}

    # ---------------- Helpers ----------------

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        # unify capitalization & ensure Close
        out = out.rename(columns={c: str(c).capitalize() for c in out.columns})
        if "Close" not in out.columns:
            # try common variants
            for alt in ("Adj close", "Adjusted close", "Price"):
                if alt in out.columns:
                    out["Close"] = out[alt]
                    break
        # ensure datetime index ascending
        if not isinstance(out.index, pd.DatetimeIndex):
            try:
                out.index = pd.to_datetime(out.index, utc=False)
            except Exception:
                pass
        return out.sort_index()

    def _empty(self, msg: str) -> Dict[str, Any]:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "target_price": 0.0,
            "stop_loss": 0.0,
            "position_size": self.cfg.default_position_size,
            "reasoning": msg,
            "risk_score": 0.5,
        }


# ---------------- Demo ----------------
if __name__ == "__main__":
    import numpy as np

    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=120, freq="D")
    base = np.cumsum(np.random.normal(0, 0.5, len(idx))) + 100
    df = pd.DataFrame({"Close": base}, index=idx)

    ap = AnalyticalPathway({})
    rec = ap.get_recommendation(df)
    print("Recommendation:", rec)
    print("Card:", AnalyticalPathway.to_recommendation_card(rec, symbol="XAU/USD"))
    print("Summary:", AnalyticalPathway.to_summary_block(rec, symbol="XAU/USD"))
