"""
advanced_analytics.py — Advanced Analytics Manager for GoldMIND AI

Purpose
-------
- Track recommendation performance and user/system activity.
- Provide system metrics with caching.
- Offer small helpers that map model/bias outputs into shapes the *new dashboard* expects.

Compatibility
-------------
- Backwards compatible with prior method names where possible.
- Safe defaults and table auto-creation so missing schemas won't crash the service.

Env (optional)
--------------
- ANALYTICS_CACHE_TTL_SEC (default 3600)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FinancialDataFramework import (graceful fallback)
# ---------------------------------------------------------------------------
try:
    from financial_data_framework import FinancialDataFramework
except ImportError:
    logger.critical("❌ Could not import FinancialDataFramework in advanced_analytics.py. Using mock.")
    class FinancialDataFramework:  # minimal mock for parsing
        def __init__(self, db_path=":memory:"):
            self.db_path = db_path
            self._init_minimal()
        def _init_minimal(self):
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("CREATE TABLE IF NOT EXISTS system_analytics (id INTEGER PRIMARY KEY, analytics_data TEXT, created_at TIMESTAMP)");
                cur.execute("CREATE TABLE IF NOT EXISTS recommendation_performance (id INTEGER PRIMARY KEY)");
                cur.execute("CREATE TABLE IF NOT EXISTS recommendations (id TEXT PRIMARY KEY, user_id INTEGER, final_confidence REAL)");
                cur.execute("CREATE TABLE IF NOT EXISTS user_activities (id INTEGER PRIMARY KEY, user_id INTEGER, activity_type TEXT, activity_data TEXT, created_at TIMESTAMP)");
                cur.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT)");
                conn.commit()
        def get_connection(self):
            return sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
        def get_usage_report(self):
            return {"apis": {}}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PerformanceRecord:
    recommendation_id: str
    user_id: int
    predicted_target_price: Optional[float]
    exit_price: Optional[float]
    entry_price: Optional[float]
    prediction_accuracy: float
    actual_return: float
    predicted_return: float
    pathway_used: str
    bias_detected: bool
    bias_adjusted: bool

# ---------------------------------------------------------------------------
# Advanced Analytics Manager
# ---------------------------------------------------------------------------
class AdvancedAnalyticsManager:
    """Manages analytics for GoldMIND AI: performance, usage, and system metrics."""
    def __init__(self, data_framework: FinancialDataFramework, cache_ttl_sec: Optional[int] = None):
        self.data_framework = data_framework
        self.metrics_cache: Dict[str, Any] = {}
        self.cache_timeout = int(cache_ttl_sec or int(__import__("os").getenv("ANALYTICS_CACHE_TTL_SEC", "3600")))
        self._ensure_tables()
        logger.info("AdvancedAnalyticsManager initialized (cache_ttl=%ss).", self.cache_timeout)

    # ----------------------------- schema helpers ----------------------------
    def _ensure_tables(self) -> None:
        """Create tables required by analytics if they don't exist."""
        with self.data_framework.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""                    CREATE TABLE IF NOT EXISTS recommendation_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recommendation_id TEXT,
                    user_id INTEGER,
                    predicted_price REAL,
                    actual_price REAL,
                    entry_price REAL,
                    prediction_accuracy REAL,
                    actual_return REAL,
                    predicted_return REAL,
                    pathway_used TEXT,
                    bias_detected BOOLEAN,
                    bias_adjusted BOOLEAN,
                    created_at TIMESTAMP
                )
            """)
            cur.execute("""                    CREATE TABLE IF NOT EXISTS recommendations (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    final_action TEXT,
                    final_confidence REAL,
                    detailed_reasoning TEXT,
                    bias_analysis TEXT,
                    timestamp TIMESTAMP,
                    final_target_price REAL,
                    final_stop_loss REAL,
                    final_position_size REAL
                )
            """)
            cur.execute("""                    CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT
                )
            """)
            cur.execute("""                    CREATE TABLE IF NOT EXISTS user_activities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    activity_type TEXT,
                    activity_data TEXT,
                    created_at TIMESTAMP
                )
            """)
            cur.execute("""                    CREATE TABLE IF NOT EXISTS system_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analytics_data TEXT,
                    created_at TIMESTAMP
                )
            """)
            conn.commit()

    # --------------------------- tracking functions --------------------------
    def track_recommendation_performance(
        self,
        recommendation_id: str,
        user_id: int,
        actual_outcome: Dict[str, Any],
        recommendation_output: Dict[str, Any],
    ) -> Optional[float]:
        """Track real-world performance of a generated recommendation.

        Aligns with the new dashboard by supporting:
        - 'final_recommendation': {'action','confidence','target_price','primary_pathway'}
        - 'bias_analysis_report': {'user_biases_detected', 'bias_impact': {'confidence_adjustment': bool}}
        """
        try:
            final_block = recommendation_output.get("final_recommendation", {}) or {}
            bias_block  = recommendation_output.get("bias_analysis_report", {}) or {}

            predicted_action = str(final_block.get("action") or "HOLD").upper()
            predicted_confidence = float(final_block.get("confidence") or 0.0)
            predicted_target_price = final_block.get("target_price")  # may be None

            entry_price = actual_outcome.get("entry_price")
            exit_price  = actual_outcome.get("exit_price")
            actual_action_implied = self._determine_actual_action(entry_price, exit_price)

            # Simple accuracy proxy
            prediction_accuracy = 1.0 if predicted_action == actual_action_implied else 0.0
            if predicted_action == "HOLD" and actual_action_implied == "HOLD":
                prediction_accuracy = 1.0
            elif predicted_action == "HOLD" and actual_action_implied != "HOLD":
                prediction_accuracy = 0.0

            # Returns
            actual_return = 0.0
            if entry_price is not None and exit_price is not None and entry_price not in (0, 0.0):
                actual_return = float(exit_price - entry_price) / float(entry_price)

            predicted_return = 0.0
            if entry_price is not None and predicted_target_price is not None and entry_price not in (0, 0.0):
                predicted_return = float(predicted_target_price - entry_price) / float(entry_price)

            # Insert
            with self.data_framework.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """                        INSERT INTO recommendation_performance (
                        recommendation_id, user_id, predicted_price, actual_price, entry_price,
                        prediction_accuracy, actual_return, predicted_return, pathway_used,
                        bias_detected, bias_adjusted, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """.strip(),
                    (
                        recommendation_id,
                        user_id,
                        predicted_target_price,
                        exit_price,
                        entry_price,
                        float(prediction_accuracy),
                        float(actual_return),
                        float(predicted_return),
                        str(final_block.get("primary_pathway", "UNKNOWN")),
                        bool(bias_block.get("user_biases_detected", False)),
                        bool((bias_block.get("bias_impact") or {}).get("confidence_adjustment", False)),
                    ),
                )
                conn.commit()

            logger.info("Performance tracked for recommendation %s (acc=%.2f)", recommendation_id, prediction_accuracy)
            return float(prediction_accuracy)
        except Exception as e:
            logger.error("Failed to track recommendation performance for %s: %s", recommendation_id, e, exc_info=True)
            return None

    def _determine_actual_action(self, entry_price: Optional[float], exit_price: Optional[float]) -> str:
        """Determine actual market action based on price movement."""
        if entry_price is None or exit_price is None or entry_price in (0, 0.0):
            return "UNKNOWN"
        if exit_price > entry_price * 1.005:  # > +0.5%
            return "BUY"
        if exit_price < entry_price * 0.995:  # < -0.5%
            return "SELL"
        return "HOLD"  # flat

    # ------------------------------ system metrics ---------------------------
    def get_system_metrics(self) -> Dict[str, Any]:
        """Retrieve aggregated system-wide metrics with caching."""
        ts = datetime.now()
        cached = self.metrics_cache.get("system_metrics")
        if cached and (ts - self.metrics_cache.get("timestamp", ts)).total_seconds() < self.cache_timeout:
            logger.debug("Serving system metrics from cache.")
            return cached

        metrics = {
            "total_recommendations_generated": 0,
            "overall_accuracy": 0.0,
            "average_actual_return": 0.0,
            "total_users": 0,
            "active_users_24h": 0,
            "average_recommendation_confidence": 0.0,
            "api_health_score": 0.0,
            "system_uptime": "N/A",
        }

        try:
            with self.data_framework.get_connection() as conn:
                cur = conn.cursor()

                # Totals
                cur.execute("SELECT COUNT(*) FROM recommendations")
                metrics["total_recommendations_generated"] = int(cur.fetchone()[0] or 0)

                cur.execute("SELECT AVG(prediction_accuracy), AVG(actual_return) FROM recommendation_performance")
                avg_acc, avg_ret = cur.fetchone() or (0.0, 0.0)
                metrics["overall_accuracy"] = round(float(avg_acc or 0.0), 4)
                metrics["average_actual_return"] = round(float(avg_ret or 0.0), 4)

                cur.execute("SELECT COUNT(*) FROM users")
                metrics["total_users"] = int(cur.fetchone()[0] or 0)

                cur.execute("SELECT COUNT(DISTINCT user_id) FROM user_activities WHERE created_at >= datetime('now','-1 day')")
                metrics["active_users_24h"] = int(cur.fetchone()[0] or 0)

                cur.execute("SELECT AVG(final_confidence) FROM recommendations")
                metrics["average_recommendation_confidence"] = round(float((cur.fetchone() or [0.0])[0] or 0.0), 4)

                # API health (safe on missing keys)
                api_report = self.data_framework.get_usage_report() or {"apis": {}}
                apis = api_report.get("apis") or {}
                total_calls = sum((v or {}).get("total_usage", 0) for v in apis.values())
                total_errors = sum((v or {}).get("total_errors", 0) for v in apis.values())
                err_rate = (float(total_errors) / float(total_calls)) if total_calls else 0.0
                metrics["api_health_score"] = round((1.0 - err_rate) * 100.0, 2)

                # Uptime placeholder
                metrics["system_uptime"] = str(datetime.now() - datetime.min)

            self.metrics_cache["system_metrics"] = metrics
            self.metrics_cache["timestamp"] = ts
            logger.info("System metrics refreshed and cached.")
            return metrics
        except Exception as e:
            logger.error("Failed to retrieve system metrics: %s", e, exc_info=True)
            return self.metrics_cache.get("system_metrics", metrics)

    # ------------------------------- logging APIs ----------------------------
    def log_user_activity(self, user_id: int, activity_type: str, activity_data: Optional[Dict[str, Any]] = None) -> None:
        """Log a user activity to the database."""
        try:
            with self.data_framework.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """                        INSERT INTO user_activities (user_id, activity_type, activity_data, created_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """.strip(),
                    (user_id, activity_type, json.dumps(activity_data) if activity_data else None),
                )
                conn.commit()
            logger.debug("User %s activity logged: %s", user_id, activity_type)
        except Exception as e:
            logger.error("Failed to log user activity %s for user %s: %s", activity_type, user_id, e, exc_info=True)

    def log_system_event(self, event_type: str, event_data: Optional[Dict[str, Any]] = None) -> None:
        """Log a system-wide event."""
        try:
            with self.data_framework.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """                        INSERT INTO system_analytics (analytics_data, created_at)
                    VALUES (?, CURRENT_TIMESTAMP)
                    """.strip(),
                    (json.dumps({"event_type": event_type, "data": event_data}),),
                )
                conn.commit()
            logger.debug("System event logged: %s", event_type)
        except Exception as e:
            logger.error("Failed to log system event %s: %s", event_type, e, exc_info=True)

    # ------------------------ dashboard mapping helpers ----------------------
    @staticmethod
    def map_bias_for_dashboard(bias_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize various bias payloads into {level, headline, confidence} used by the dashboard."""
        level = (bias_obj.get("biasLevel") or bias_obj.get("level") or bias_obj.get("bias") or "neutral").upper()
        headline = bias_obj.get("headline") or bias_obj.get("message") or f"{level.capitalize()} bias"
        conf = bias_obj.get("confidence") or bias_obj.get("score") or 0.5
        try:
            conf = float(conf)
            if conf > 1.0: conf = max(0.0, min(100.0, conf)) / 100.0
        except Exception:
            conf = 0.5
        return {"level": level, "headline": headline, "confidence": conf}

    @staticmethod
    def map_prediction_for_dashboard(pred_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize prediction dict into {action, confidence, entry, target, stop_loss, risk_reward}."""
        action = (pred_obj.get("action") or pred_obj.get("final_action") or "HOLD").upper()
        conf = pred_obj.get("confidence") or pred_obj.get("final_confidence") or 0.5
        try:
            conf = float(conf)
            if conf > 1.0: conf = max(0.0, min(100.0, conf)) / 100.0
        except Exception:
            conf = 0.5
        return {
            "action": action,
            "confidence": conf,
            "entry": pred_obj.get("entry"),
            "target": pred_obj.get("target") or pred_obj.get("final_target_price"),
            "stop_loss": pred_obj.get("stop_loss") or pred_obj.get("final_stop_loss"),
            "risk_reward": pred_obj.get("risk_reward"),
        }

# ------------------------------- standalone test ---------------------------
if __name__ == "__main__":
    import uuid
    import asyncio
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    class MockFDF(FinancialDataFramework):
        def __init__(self, db_path=":memory:"):
            super().__init__(db_path)
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("CREATE TABLE IF NOT EXISTS recommendations (id TEXT PRIMARY KEY, user_id INTEGER, final_confidence REAL)")
                conn.commit()
        def get_usage_report(self):
            return {'apis': {'alpha_vantage': {'total_usage': 1000, 'total_errors': 50}, 'fred': {'total_usage': 500, 'total_errors': 0}}}

    fdf = MockFDF()
    mgr = AdvancedAnalyticsManager(fdf)

    # Track two sample recommendations
    rec_id_1 = str(uuid.uuid4())
    rec_out_1 = {"final_recommendation": {"action": "BUY", "confidence": 0.8, "target_price": 105.0, "primary_pathway": "LSTM"}, "bias_analysis_report": {"user_biases_detected": False, "bias_impact": {}}}
    actual_1 = {"entry_price": 100.0, "exit_price": 106.0}
    print("Accuracy 1:", mgr.track_recommendation_performance(rec_id_1, 1, actual_1, rec_out_1))

    rec_id_2 = str(uuid.uuid4())
    rec_out_2 = {"final_recommendation": {"action": "SELL", "confidence": 0.7, "target_price": 95.0, "primary_pathway": "ANALYTICAL"}, "bias_analysis_report": {"user_biases_detected": True, "bias_impact": {"confidence_adjustment": True}}}
    actual_2 = {"entry_price": 100.0, "exit_price": 93.0}
    print("Accuracy 2:", mgr.track_recommendation_performance(rec_id_2, 1, actual_2, rec_out_2))

    # Log activity + system event
    mgr.log_user_activity(1, "login", {"device": "mobile"})
    mgr.log_system_event("model_retrain_success", {"model": "LSTM", "duration": "15min"})

    # Metrics
    print("System metrics:", json.dumps(mgr.get_system_metrics(), indent=2))

    # Mapping helpers
    print("Map bias:", mgr.map_bias_for_dashboard({"bias": "neutral", "confidence": 0.63}))
    print("Map pred:", mgr.map_prediction_for_dashboard({"action": "buy", "confidence": 0.71, "entry": 100, "target": 103, "stop_loss": 98, "risk_reward": 1.5}))
