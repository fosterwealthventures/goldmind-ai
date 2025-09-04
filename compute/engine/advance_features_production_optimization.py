"""
Advanced Features & Production Optimization for GoldMIND AI
----------------------------------------------------------
- Resilient imports (safe fallbacks)
- User personalization (SQLite-backed with in‑memory cache)
- Market regime detection (volatility/trend heuristic)
- Adaptive learning stubs
- Orchestrator that plays nicely with the dashboard and API shapes

This module is SAFE TO IMPORT even if optional deps are missing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# FinancialDataFramework (placeholder first, overwritten by real if available)
# ============================================================================

class FinancialDataFramework:
    """Minimal placeholder that provides a SQLite store and simple data access."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or ":memory:"
        self._init_mock_database()

    def _init_mock_database(self) -> None:
        try:
            if self.db_path != ":memory:":
                os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS recommendations (
                        id TEXT PRIMARY KEY,
                        user_id INTEGER,
                        final_action TEXT,
                        final_confidence REAL,
                        detailed_reasoning TEXT,
                        bias_analysis TEXT,
                        timestamp TEXT,
                        predicted_price REAL,
                        actual_price REAL,
                        entry_price REAL,
                        prediction_accuracy REAL,
                        actual_return REAL,
                        predicted_return REAL,
                        pathway_used TEXT,
                        bias_detected INTEGER,
                        bias_adjusted INTEGER,
                        conflict_severity TEXT,
                        resolution_method TEXT,
                        pathway_weights TEXT,
                        alternative_scenarios TEXT,
                        market_regime TEXT,
                        personalization_applied INTEGER,
                        final_target_price REAL,
                        final_stop_loss REAL,
                        final_position_size REAL,
                        followed_recommendation INTEGER
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id INTEGER PRIMARY KEY,
                        risk_tolerance REAL,
                        time_horizon TEXT,
                        preferred_pathway TEXT,
                        profile_data TEXT
                    )
                    """
                )
                conn.commit()
            logger.info("Mock FDF DB ready at %s", self.db_path)
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to init mock DB: %s", e)

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    async def get_stock_data(self, ticker: str, days: int = 120) -> pd.DataFrame:
        # Deterministic synthetic OHLCV so callers have something to work with
        rng = np.random.default_rng(abs(hash(ticker)) % (2**32 - 1))
        idx = pd.date_range(end=datetime.now(), periods=days, freq="D")
        base = np.cumsum(rng.normal(0, 0.5, size=days)) + 100.0
        df = pd.DataFrame(
            {
                "open": base + rng.normal(0, 0.2, size=days),
                "high": base + rng.uniform(0.1, 0.6, size=days),
                "low": base - rng.uniform(0.1, 0.6, size=days),
                "close": base,
                "volume": rng.integers(50_000, 150_000, size=days),
            },
            index=idx,
        )
        return df

    async def get_gold_ohlcv_data(self, *_, **__) -> pd.DataFrame:
        return await self.get_stock_data("XAUUSD", 180)

    def get_usage_report(self) -> Dict[str, Any]:
        return {"apis": {}}


# Try to overwrite placeholder with the real one, if available
try:  # pragma: no cover
    from financial_data_framework import FinancialDataFramework as RealFDF  # type: ignore
    FinancialDataFramework = RealFDF  # noqa: F811
    logger.info("✅ Loaded real FinancialDataFramework")
except Exception as e:  # pragma: no cover
    logger.warning("Using placeholder FinancialDataFramework (%s)", e)


# ============================================================================
# Dual-system / LSTM placeholders (overwritten if real modules exist)
# ============================================================================

@dataclass
class ConflictResolution:
    final_action: str
    final_confidence: float
    final_target_price: float
    final_stop_loss: float
    final_position_size: float
    conflict_severity: str
    resolution_method: str
    pathway_weights: Dict[str, float]
    consensus_score: float
    uncertainty_factor: float
    detailed_reasoning: str
    alternative_scenarios: List[Dict]


class DualSystemConflictResolver:
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.pathway_weights = {"ANALYTICAL": 0.4, "LSTM_TEMPORAL": 0.6}

    def resolve_recommendations(self, analytical: Dict[str, Any], lstm: Dict[str, Any]) -> ConflictResolution:
        a_c = float(analytical.get("confidence", 0.5))
        l_c = float(lstm.get("confidence", 0.5))
        winner = analytical if a_c >= l_c else lstm
        return ConflictResolution(
            final_action=str(winner.get("action", "HOLD")).upper(),
            final_confidence=float(winner.get("confidence", 0.5)),
            final_target_price=float(winner.get("target_price", 0.0)),
            final_stop_loss=float(winner.get("stop_loss", 0.0)),
            final_position_size=float(winner.get("position_size", 0.0)),
            conflict_severity="LOW",
            resolution_method="max_confidence_stub",
            pathway_weights=self.pathway_weights.copy(),
            consensus_score=max(a_c, l_c),
            uncertainty_factor=1.0 - max(a_c, l_c),
            detailed_reasoning=winner.get("reasoning", "max confidence"),
            alternative_scenarios=[],
        )


class LSTMTemporalAnalysis:
    def __init__(self, financial_data_framework: FinancialDataFramework):
        self.fdf = financial_data_framework

    def _gen(self, days: int) -> pd.DataFrame:
        idx = pd.date_range(end=datetime.now(), periods=days, freq="D")
        base = np.linspace(100, 110, days)
        return pd.DataFrame({"close": base, "volume": np.random.randint(100_000, 200_000, days)}, index=idx)

    async def predict_stock_price(self, ticker: str) -> Dict[str, Any]:
        df = self._gen(60)
        action = ["BUY", "SELL", "HOLD"][int(np.random.randint(0, 3))]
        conf = float(np.random.uniform(0.5, 0.75))
        return {"action": action, "confidence": conf, "target_price": df["close"].iloc[-1] * (1.05 if action == "BUY" else 0.97)}

    async def train_models(self, *_, **__) -> bool:
        return True

    def save_models(self) -> None:
        return None

    def load_models(self) -> None:
        return None


try:  # pragma: no cover
    from dual_system_conflict_resolution import DualSystemConflictResolver as RealResolver  # type: ignore
    DualSystemConflictResolver = RealResolver  # noqa: F811
    from lstm_temporal_analysis import LSTMTemporalAnalysis as RealLSTM  # type: ignore
    LSTMTemporalAnalysis = RealLSTM  # noqa: F811
    logger.info("✅ Loaded real DualSystemConflictResolver & LSTMTemporalAnalysis")
except Exception as e:  # pragma: no cover
    logger.warning("Using placeholder resolver/LSTM (%s)", e)


# ============================================================================
# Advanced Analytics (placeholder first)
# ============================================================================

class AdvancedAnalyticsManager:
    def __init__(self, data_framework: FinancialDataFramework):
        self.df = data_framework

    def track_recommendation_performance(self, recommendation_id: str, user_id: int, actual_outcome: Dict, recommendation_output: Dict) -> Optional[float]:
        logger.info("Mock analytics tracking for %s (user %s)", recommendation_id, user_id)
        return 0.9

    def get_system_metrics(self) -> Dict[str, Any]:
        return {"total_recommendations": 0, "average_accuracy": 0.0}


try:  # pragma: no cover
    from advanced_analytics import AdvancedAnalyticsManager as RealAnalytics  # type: ignore
    AdvancedAnalyticsManager = RealAnalytics  # noqa: F811
    logger.info("✅ Loaded real AdvancedAnalyticsManager")
except Exception as e:  # pragma: no cover
    logger.warning("Using placeholder AdvancedAnalyticsManager (%s)", e)


# ============================================================================
# User Personalization
# ============================================================================

class UserPersonalizationEngine:
    def __init__(self, db_manager: FinancialDataFramework):
        self.db = db_manager
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._ttl_sec = 300

    async def build_user_profile(self, user_id: int) -> Dict[str, Any]:
        now = datetime.utcnow()
        c = self._cache.get(user_id)
        if c and (now - c["_at"]).total_seconds() < self._ttl_sec:
            return c["profile"]

        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT profile_data, risk_tolerance, time_horizon, preferred_pathway
                FROM user_profiles
                WHERE user_id = ?
                """,
                (user_id,),
            )
            row = cur.fetchone()

        profile = {
            "user_id": user_id,
            "risk_preferences": {"avg_risk_tolerance": float(row["risk_tolerance"]) if row and row["risk_tolerance"] is not None else 0.5},
            "time_horizon": (row["time_horizon"] if row and row["time_horizon"] else "medium"),
            "preferred_pathway": (row["preferred_pathway"] if row and row["preferred_pathway"] else None),
            "last_updated": now.isoformat(),
        }
        if row and row["profile_data"]:
            try:
                profile.update(json.loads(row["profile_data"]))
            except Exception:
                pass

        self._cache[user_id] = {"_at": now, "profile": profile}
        return profile


# ============================================================================
# Market Regime Detector
# ============================================================================

class MarketRegimeDetector:
    def __init__(self, data_framework: FinancialDataFramework, lstm: LSTMTemporalAnalysis):
        self.df = data_framework
        self.lstm = lstm

    async def detect_current_regime(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        if market_data is None or market_data.empty:
            market_data = await self.df.get_stock_data("XAUUSD", 120)
        tail = market_data["close"].tail(20)
        if tail.empty:
            return {"regime": "unknown", "sentiment": "unknown", "confidence": 0.0, "recommended_adjustments": {}}

        pct = tail.iloc[-1] / tail.iloc[0] - 1.0
        vol = float(tail.std() / max(1e-9, tail.mean()))
        if pct > 0.05:
            regime, sentiment = "trending", "bullish"
        elif pct < -0.05:
            regime, sentiment = "trending", "bearish"
        elif vol > 0.03:
            regime, sentiment = "volatile", "uncertain"
        else:
            regime, sentiment = "sideways", "neutral"

        return {
            "regime": regime,
            "sentiment": sentiment,
            "confidence": 0.8,
            "recommended_adjustments": {
                "position_size_multiplier": 0.7 if regime == "volatile" else 1.1 if regime == "trending" else 1.0,
                "preferred_pathways": ["ANALYTICAL"] if regime == "trending" else ["LSTM_TEMPORAL"],
            },
        }


# ============================================================================
# Adaptive Learning
# ============================================================================

class AdaptiveLearningSystem:
    def __init__(self, resolver: DualSystemConflictResolver, analytics: AdvancedAnalyticsManager):
        self.resolver = resolver
        self.analytics = analytics

    def adapt_pathway_weights(self) -> Dict[str, float]:
        w = self.resolver.pathway_weights.copy()
        # Tiny nudge to demonstrate adaptation
        w["ANALYTICAL"] = min(0.6, w.get("ANALYTICAL", 0.4) + 0.01)
        w["LSTM_TEMPORAL"] = max(0.2, w.get("LSTM_TEMPORAL", 0.6) - 0.01)
        s = sum(w.values())
        return {k: v / s for k, v in w.items()}


# ============================================================================
# Ultimate Orchestrator
# ============================================================================

class UltimateBiasAwareManager:
    """
    Orchestrates advanced features to generate personalized, bias-aware
    and regime-adaptive recommendations.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        financial_data_framework: FinancialDataFramework,
        lstm_temporal_analyzer: Optional[LSTMTemporalAnalysis] = None,
    ) -> None:
        self.config = config or {}
        self.fdf = financial_data_framework
        self.lstm = lstm_temporal_analyzer or LSTMTemporalAnalysis(self.fdf)
        self.resolver = DualSystemConflictResolver(self.config.get("dual_system", {}))
        self.analytics = AdvancedAnalyticsManager(self.fdf)
        self.personalization = UserPersonalizationEngine(self.fdf)
        self.regime_detector = MarketRegimeDetector(self.fdf, self.lstm)
        self._store_lock = asyncio.Lock()

    async def generate_recommendation(self, ticker: str, user_id: int, user_input: str = "") -> Dict[str, Any]:
        data = await self.fdf.get_stock_data(ticker, self.config.get("lookback_days", 120))
        if data is None or data.empty:
            return {"success": False, "error": f"No market data for {ticker}"}

        profile = await self.personalization.build_user_profile(user_id)
        regime = await self.regime_detector.detect_current_regime(data)

        # Pathway A (analytical) — simple momentum mock
        tail = data["close"].tail(20)
        momentum = float(tail.iloc[-1] / tail.iloc[0] - 1.0) if len(tail) >= 2 else 0.0
        a_action = "BUY" if momentum > 0.02 else "SELL" if momentum < -0.02 else "HOLD"
        analytical = {

            "confidence": float(min(0.75, max(0.5, abs(momentum) * 5 + 0.5))),
            "target_price": float(tail.iloc[-1] * (1.04 if a_action == "BUY" else 0.97 if a_action == "SELL" else 1.0)),
            "stop_loss": float(tail.iloc[-1] * (0.96 if a_action == "BUY" else 1.03 if a_action == "SELL" else 0.99)),
            "position_size": 0.15,
            "reasoning": f"Momentum {momentum:+.2%} → {a_action}.",
            "risk_score": 0.45,
        }

        # Pathway B (LSTM) — placeholder or real
        lstm_pred = await self.lstm.predict_stock_price(ticker)
        lstm = {
            "action": str(lstm_pred.get("action", "HOLD")).upper(),
            "confidence": float(lstm_pred.get("confidence", 0.55)),
            "target_price": float(lstm_pred.get("target_price", tail.iloc[-1])),
            "stop_loss": float(tail.iloc[-1] * (0.95 if lstm_pred.get("action") == "BUY" else 1.03 if lstm_pred.get("action") == "SELL" else 0.99)),
            "position_size": 0.15,
            "reasoning": f"LSTM suggests {lstm_pred.get('action')} with {float(lstm_pred.get('confidence', 0.55)):.2f} confidence.",
            "risk_score": 0.50,
        }

        # Conflict resolution
        cr = self.resolver.resolve_recommendations(analytical, lstm)

        # Regime & personalization adjustments
        ps_mul = float(regime.get("recommended_adjustments", {}).get("position_size_multiplier", 1.0))
        adj_pos = float(max(0.05, min(0.5, cr.final_position_size * ps_mul)))

        # Final recommendation block
        final_block = {
            "final_action": cr.final_action,
            "final_confidence": float(max(0.0, min(1.0, cr.final_confidence))),
            "final_target_price": cr.final_target_price,
            "final_stop_loss": cr.final_stop_loss,
            "final_position_size": adj_pos,
            "resolution_method": cr.resolution_method,
            "conflict_severity": cr.conflict_severity,
            "pathway_weights": cr.pathway_weights,
            "consensus_score": cr.consensus_score,
            "uncertainty_factor": cr.uncertainty_factor,
            "detailed_reasoning": cr.detailed_reasoning,
            "alternative_scenarios": cr.alternative_scenarios,
        }

        # Store to DB (best-effort)
        rec_id = str(uuid.uuid4())
        try:
            async with self._store_lock:
                with self.fdf.get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute(
                        """
                        INSERT INTO recommendations (
                            id, user_id, final_action, final_confidence,
                            detailed_reasoning, bias_analysis, timestamp,
                            predicted_price, actual_price, entry_price,
                            prediction_accuracy, actual_return, predicted_return,
                            pathway_used, bias_detected, bias_adjusted,
                            conflict_severity, resolution_method, pathway_weights,
                            alternative_scenarios, market_regime, personalization_applied,
                            final_target_price, final_stop_loss, final_position_size,
                            followed_recommendation
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            rec_id,
                            user_id,
                            final_block["final_action"],
                            final_block["final_confidence"],
                            final_block["detailed_reasoning"],
                            json.dumps({"detector": "n/a"}),
                            datetime.utcnow().isoformat(),
                            final_block["final_target_price"],
                            None,  # actual_price unknown at time of rec
                            None,  # entry_price could be set by executor
                            None,  # prediction_accuracy TBD
                            None,  # actual_return TBD
                            None,  # predicted_return TBD
                            "AUTO",
                            0,
                            0,
                            final_block["conflict_severity"],
                            final_block["resolution_method"],
                            json.dumps(final_block["pathway_weights"]),
                            json.dumps(final_block["alternative_scenarios"]),
                            json.dumps(regime),
                            1,
                            final_block["final_target_price"],
                            final_block["final_stop_loss"],
                            final_block["final_position_size"],
                            0,
                        ),
                    )
                    conn.commit()
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to persist recommendation: %s", e)

        return {
            "success": True,
            "id": rec_id,
            "ticker": ticker,
            "user_id": user_id,
            "generated_at": datetime.utcnow().isoformat(),
            "profile": profile,
            "regime": regime,
            "pathways": {"analytical": analytical, "lstm": lstm},
            "final": final_block,
        }

    # ---------------- Dashboard mappers ----------------

    @staticmethod
    def to_recommendation_card(result: Dict[str, Any]) -> Dict[str, Any]:
        final = result.get("final", {})
        # use ticker price proxy if known (not always present in this module)
        entry = None
        target = final.get("final_target_price")
        return {
            "action": str(final.get("final_action", "HOLD")).upper(),
            "confidence": float(final.get("final_confidence", 0.5)),
            "entry": entry,
            "target": target,
            "note": final.get("detailed_reasoning", ""),
            "symbol": result.get("ticker", ""),
        }

    @staticmethod
    def to_bias_summary(_: Dict[str, Any]) -> Dict[str, Any]:
        # This module doesn't run a full bias detector; keep neutral summary
        return {"level": "LOW", "headline": "No major bias detected", "confidence": 0.5}

    @staticmethod
    def to_summary_block(result: Dict[str, Any]) -> Dict[str, Any]:
        card = UltimateBiasAwareManager.to_recommendation_card(result)
        action = card.get("action", "HOLD").upper()
        pc = float(card.get("confidence", 0.5)) * 100.0

        if action == "BUY" and pc >= 65:
            regime = "Bullish Trend"
        elif action == "SELL" and pc >= 65:
            regime = "Bearish Trend"
        else:
            regime = "Neutral / Range"

        # Use regime detector output if available for volatility flavor
        rg = result.get("regime", {})
        sentiment = (rg.get("sentiment") or "neutral").capitalize()
        volatility = "High" if rg.get("regime") == "volatile" else "Moderate" if rg.get("regime") == "trending" else "Low"
        note = f"{sentiment} conditions. {card.get('note','')}".strip()
        return {"regime": regime, "volatility": volatility, "note": note, "symbol": result.get("ticker", "")}


# ============================================================================
# Demo harness
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def _demo():
        logging.basicConfig(level=logging.INFO)
        fdf = FinancialDataFramework(db_path=str(Path("./gm_demo.db")))
        mgr = UltimateBiasAwareManager({"lookback_days": 120}, fdf)
        out = await mgr.generate_recommendation("XAUUSD", user_id=1, user_input="testing advanced features")
        print("=== Full Output ===")
        print(json.dumps(out, indent=2))
        print("\n=== Card ===")
        print(json.dumps(UltimateBiasAwareManager.to_recommendation_card(out), indent=2))
        print("\n=== Summary ===")
        print(json.dumps(UltimateBiasAwareManager.to_summary_block(out), indent=2))

    asyncio.run(_demo())
