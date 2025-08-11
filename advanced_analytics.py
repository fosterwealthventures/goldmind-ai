# advanced_analytics.py
"""
Advanced Analytics Manager for GoldMIND AI
Provides comprehensive analytics tracking, performance monitoring,
and data aggregation for system insights.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import sqlite3 # Used for db_manager connection
import pandas as pd # Used for data processing in more advanced analytics

logger = logging.getLogger(__name__)

# Import FinancialDataFramework (used by AdvancedAnalyticsManager's __init__)
try:
    from financial_data_framework import FinancialDataFramework
except ImportError:
    logger.critical("❌ Could not import FinancialDataFramework in advanced_analytics.py. Using mock.")
    class FinancialDataFramework: # Mock for parsing
        def __init__(self, *args, **kwargs): pass
        def get_connection(self):
            # Return a simple in-memory connection if FDF is truly missing
            conn = sqlite3.connect(':memory:')
            conn.execute("CREATE TABLE IF NOT EXISTS system_analytics (id INTEGER PRIMARY KEY)")
            conn.execute("CREATE TABLE IF NOT EXISTS recommendation_performance (id INTEGER PRIMARY KEY)")
            conn.commit()
            return conn
        def get_usage_report(self): return {'apis': {}}


class AdvancedAnalyticsManager:
    """
    Manages advanced analytics for GoldMIND AI system.
    Tracks model performance, user engagement, and system-wide metrics.
    """
    def __init__(self, data_framework: FinancialDataFramework):
        self.data_framework = data_framework
        self.metrics_cache = {}
        self.cache_timeout = 3600 # 1 hour cache for system metrics
        logger.info("AdvancedAnalyticsManager initialized.")

    def track_recommendation_performance(self, recommendation_id: str, user_id: int, actual_outcome: Dict, recommendation_output: Dict) -> Optional[float]:
        """
        Tracks the real-world performance of a generated recommendation.
        Calculates accuracy and stores it in the database.
        """
        try:
            # Extract relevant data
            predicted_action = recommendation_output['final_recommendation']['action']
            predicted_confidence = recommendation_output['final_recommendation']['confidence']
            predicted_target_price = recommendation_output['final_recommendation']['target_price']
            
            entry_price = actual_outcome.get('entry_price')
            exit_price = actual_outcome.get('exit_price')
            actual_action_implied = self._determine_actual_action(entry_price, exit_price)

            # Simple accuracy calculation (can be much more sophisticated)
            prediction_accuracy = 1.0 if predicted_action == actual_action_implied else 0.0
            if predicted_action == "HOLD" and actual_action_implied == "HOLD":
                prediction_accuracy = 1.0
            elif predicted_action == "HOLD" and actual_action_implied != "HOLD":
                prediction_accuracy = 0.0 # Holding when movement occurred

            # Calculate actual return
            actual_return = 0.0
            if entry_price and exit_price and entry_price != 0:
                actual_return = (exit_price - entry_price) / entry_price

            predicted_return = 0.0
            if entry_price and predicted_target_price and entry_price != 0:
                predicted_return = (predicted_target_price - entry_price) / entry_price

            # Store in database (recommendation_performance table)
            with self.data_framework.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO recommendation_performance (
                        recommendation_id, user_id, predicted_price, actual_price, entry_price,
                        prediction_accuracy, actual_return, predicted_return, pathway_used,
                        bias_detected, bias_adjusted, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    recommendation_id,
                    user_id,
                    predicted_target_price,
                    exit_price,
                    entry_price,
                    prediction_accuracy,
                    actual_return,
                    predicted_return,
                    recommendation_output['final_recommendation'].get('primary_pathway', 'UNKNOWN'),
                    recommendation_output['bias_analysis_report'].get('user_biases_detected', False),
                    recommendation_output['bias_analysis_report'].get('bias_impact', {}).get('confidence_adjustment', False),
                ))
                conn.commit()
            logger.info(f"Performance tracked for recommendation {recommendation_id}. Accuracy: {prediction_accuracy:.2f}")
            return prediction_accuracy

        except Exception as e:
            logger.error(f"Failed to track recommendation performance for {recommendation_id}: {e}", exc_info=True)
            return None

    def _determine_actual_action(self, entry_price: Optional[float], exit_price: Optional[float]) -> str:
        """Helper to determine actual market action based on price movement."""
        if entry_price is None or exit_price is None or entry_price == 0:
            return "UNKNOWN"
        if exit_price > entry_price * 1.005: # Price increased by >0.5%
            return "BUY"
        if exit_price < entry_price * 0.995: # Price decreased by >0.5%
            return "SELL"
        return "HOLD" # Price stayed relatively flat


    def get_system_metrics(self) -> Dict:
        """
        Retrieves aggregated system-wide analytics metrics.
        Caches results to reduce database load.
        """
        if 'system_metrics' in self.metrics_cache and \
           (datetime.now() - self.metrics_cache['timestamp']).total_seconds() < self.cache_timeout:
            logger.debug("Serving system metrics from cache.")
            return self.metrics_cache['system_metrics']

        metrics = {
            "total_recommendations_generated": 0,
            "overall_accuracy": 0.0,
            "average_actual_return": 0.0,
            "total_users": 0,
            "active_users_24h": 0,
            "average_recommendation_confidence": 0.0,
            "api_health_score": 0.0,
            "system_uptime": "N/A"
        }

        try:
            with self.data_framework.get_connection() as conn:
                cursor = conn.cursor()

                # Total recommendations
                cursor.execute("SELECT COUNT(*) FROM recommendations")
                metrics["total_recommendations_generated"] = cursor.fetchone()[0]

                # Overall accuracy and return
                cursor.execute("SELECT AVG(prediction_accuracy), AVG(actual_return) FROM recommendation_performance")
                acc_ret = cursor.fetchone()
                metrics["overall_accuracy"] = round(acc_ret[0] or 0.0, 4)
                metrics["average_actual_return"] = round(acc_ret[1] or 0.0, 4)

                # Total users
                cursor.execute("SELECT COUNT(*) FROM users")
                metrics["total_users"] = cursor.fetchone()[0]

                # Active users (e.g., logged in or generated recs in last 24h)
                cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_activities WHERE created_at >= datetime('now', '-1 day')")
                metrics["active_users_24h"] = cursor.fetchone()[0]

                # Average recommendation confidence
                cursor.execute("SELECT AVG(final_confidence) FROM recommendations")
                metrics["average_recommendation_confidence"] = round(cursor.fetchone()[0] or 0.0, 4)
                
                # API health score (from FinancialDataFramework's report)
                api_report = self.data_framework.get_usage_report()
                total_api_calls = sum(api['total_usage'] for api in api_report['apis'].values())
                total_api_errors = sum(api['total_errors'] for api in api_report['apis'].values())
                api_error_rate = (total_api_errors / total_api_calls) if total_api_calls > 0 else 0
                metrics["api_health_score"] = round((1 - api_error_rate) * 100, 2) # 100 - (error_rate * 100)

                # System uptime (assuming start_time is available, perhaps from a global config)
                # This would typically come from the main Flask app or a system-level monitor
                metrics["system_uptime"] = str(datetime.now() - datetime.min) # Placeholder, adjust as needed

            self.metrics_cache['system_metrics'] = metrics
            self.metrics_cache['timestamp'] = datetime.now()
            logger.info("System metrics refreshed and cached.")
            return metrics

        except Exception as e:
            logger.error(f"Failed to retrieve system metrics: {e}", exc_info=True)
            # Return cached data if available, or basic default if not
            if 'system_metrics' in self.metrics_cache:
                return self.metrics_cache['system_metrics']
            return metrics # Return basic metrics on error

    def log_user_activity(self, user_id: int, activity_type: str, activity_data: Optional[Dict] = None):
        """Logs user activities to the database."""
        try:
            with self.data_framework.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_activities (user_id, activity_type, activity_data, created_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    user_id,
                    activity_type,
                    json.dumps(activity_data) if activity_data else None
                ))
                conn.commit()
            logger.debug(f"User {user_id} activity logged: {activity_type}")
        except Exception as e:
            logger.error(f"Failed to log user activity {activity_type} for user {user_id}: {e}", exc_info=True)

    def log_system_event(self, event_type: str, event_data: Optional[Dict] = None):
        """Logs system-wide events to the database."""
        try:
            with self.data_framework.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_analytics (analytics_data, created_at)
                    VALUES (?, CURRENT_TIMESTAMP)
                ''', (
                    json.dumps({"event_type": event_type, "data": event_data})
                ))
                conn.commit()
            logger.debug(f"System event logged: {event_type}")
        except Exception as e:
            logger.error(f"Failed to log system event {event_type}: {e}", exc_info=True)

# --- Standalone Testing Example ---
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('AdvancedAnalyticsManager_Test')

    # Mock FinancialDataFramework for standalone testing
    class MockDataFrameworkForAnalytics(FinancialDataFramework): # Inherit from placeholder/real FDF
        def __init__(self, db_path=":memory:"):
            super().__init__(db_path)
            self.db_path = db_path
            # Ensure tables needed by analytics are created in the mock DB
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS recommendations (
                        id TEXT PRIMARY KEY, user_id INTEGER, final_action TEXT, final_confidence REAL, 
                        detailed_reasoning TEXT, bias_analysis TEXT, timestamp TIMESTAMP,
                        final_target_price REAL, final_stop_loss REAL, final_position_size REAL
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS recommendation_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, recommendation_id TEXT, user_id INTEGER, 
                        predicted_price REAL, actual_price REAL, entry_price REAL, prediction_accuracy REAL,
                        actual_return REAL, predicted_return REAL, pathway_used TEXT, bias_detected BOOLEAN,
                        bias_adjusted BOOLEAN, created_at TIMESTAMP
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT)
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_activities (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, activity_type TEXT, activity_data TEXT, created_at TIMESTAMP)
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_analytics (id INTEGER PRIMARY KEY AUTOINCREMENT, analytics_data TEXT, created_at TIMESTAMP)
                ''')
                conn.commit()
            logger.info("MockDataFrameworkForAnalytics: Specific tables for analytics created.")
        
        def get_usage_report(self):
            # Simulate API usage for get_system_metrics
            return {'apis': {'alpha_vantage': {'total_usage': 1000, 'total_errors': 50}, 'fred': {'total_usage': 500, 'total_errors': 0}}}

    mock_data_framework = MockDataFrameworkForAnalytics()
    analytics_manager = AdvancedAnalyticsManager(mock_data_framework)

    print("\n--- Testing Tracking Recommendation Performance ---")
    rec_id_1 = str(uuid.uuid4())
    user_id_1 = 1
    rec_output_1 = {'final_recommendation': {'action': 'BUY', 'confidence': 0.8, 'target_price': 105.0, 'primary_pathway': 'LSTM'}, 'bias_analysis_report': {'user_biases_detected': False, 'bias_impact': {}}}
    actual_outcome_1 = {'entry_price': 100.0, 'exit_price': 106.0}
    accuracy = analytics_manager.track_recommendation_performance(rec_id_1, user_id_1, actual_outcome_1, rec_output_1)
    print(f"Tracked accuracy for rec {rec_id_1}: {accuracy}")

    rec_id_2 = str(uuid.uuid4())
    user_id_2 = 1
    rec_output_2 = {'final_recommendation': {'action': 'SELL', 'confidence': 0.7, 'target_price': 95.0, 'primary_pathway': 'ANALYTICAL'}, 'bias_analysis_report': {'user_biases_detected': True, 'bias_impact': {'confidence_adjustment': True}}}
    actual_outcome_2 = {'entry_price': 100.0, 'exit_price': 93.0}
    accuracy = analytics_manager.track_recommendation_performance(rec_id_2, user_id_2, actual_outcome_2, rec_output_2)
    print(f"Tracked accuracy for rec {rec_id_2}: {accuracy}")

    print("\n--- Testing Logging User Activity ---")
    analytics_manager.log_user_activity(user_id=1, activity_type="login", activity_data={"device": "mobile"})
    analytics_manager.log_user_activity(user_id=1, activity_type="recommendation_view", activity_data={"rec_id": rec_id_1})
    analytics_manager.log_user_activity(user_id=2, activity_type="login")

    print("\n--- Testing Logging System Event ---")
    analytics_manager.log_system_event(event_type="model_retrain_success", event_data={"model": "LSTM", "duration": "15min"})

    print("\n--- Testing Getting System Metrics ---")
    # Insert some dummy data for users to make counts non-zero
    with mock_data_framework.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO users (id, username) VALUES (?, ?)", (1, 'testuser1'))
        cursor.execute("INSERT OR IGNORE INTO users (id, username) VALUES (?, ?)", (2, 'testuser2'))
        conn.commit()

    system_metrics = analytics_manager.get_system_metrics()
    print(json.dumps(system_metrics, indent=2))
    assert system_metrics['total_recommendations_generated'] >= 2 # At least the two we inserted

    print("\nAdvanced Analytics Manager testing complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Advanced Analytics Manager example interrupted by user.")
    except Exception as e:
        logger.critical(f"Advanced Analytics Manager example failed: {e}", exc_info=True)
        exit(1)