"""
GoldMIND AI User Analytics System
Advanced user behavior analytics and engagement tracking
Provides insights into user patterns, performance, and system optimization.
Integrates with FinancialDataFramework for database persistence.
"""

import sqlite3
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import time
from collections import defaultdict
import math
import sys
import contextlib

# Import FinancialDataFramework for database interaction
try:
    from financial_data_framework import FinancialDataFramework
except ImportError:
    logging.critical("❌ Could not import FinancialDataFramework. Please ensure 'financial_data_framework.py' is accessible.")
    class FinancialDataFramework: # Mock for parsing
        def __init__(self, *args, **kwargs): pass
        def get_connection(self): 
            # Return a simple in-memory connection if FDF is truly missing
            conn = sqlite3.connect(':memory:')
            conn.row_factory = sqlite3.Row
            # Create a mock users table with 'status' column for testing
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    status TEXT DEFAULT 'active'
                )
            ''')
            conn.commit()
            return conn
        def init_database(self):
            logging.info("Mock FinancialDataFramework: init_database called.")
            # In a real FDF, this would create all necessary tables including 'users'
            pass
        def get_user_profile(self, user_id: int) -> Optional[Dict]:
            return None # Mock
        def save_user_profile(self, user_id: int, profile_data: Dict):
            pass # Mock


logger = logging.getLogger(__name__)

class BehaviorTracker:
    """Tracks and logs user actions and interactions within the system."""
    def __init__(self):
        self.action_log: List[Dict] = []
        logger.info("BehaviorTracker initialized.")

    def track_user_action(self, user_id: int, action_type: str, details: Dict):
        """Logs a user action."""
        action = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action_type": action_type,
            "details": details
        }
        self.action_log.append(action)
        logger.debug(f"User {user_id} performed action: {action_type} with details: {details}")

    def get_recent_actions(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Retrieves recent actions for a specific user."""
        return [a for a in self.action_log if a['user_id'] == user_id][-limit:]

class EngagementScorer:
    """Calculates user engagement scores based on various metrics."""
    def __init__(self):
        logger.info("EngagementScorer initialized.")

    def calculate_score(self, user_activity_data: List[Dict]) -> float:
        """
        Calculates an engagement score.
        Simple example: score based on number of actions and recency.
        """
        if not user_activity_data:
            return 0.0

        score = 0.0
        now = datetime.utcnow()
        for activity in user_activity_data:
            activity_time = datetime.fromisoformat(activity['timestamp'])
            time_diff_hours = (now - activity_time).total_seconds() / 3600

            # Weight by recency (more recent actions get higher weight)
            recency_weight = max(0, 1 - (time_diff_hours / (24 * 7))) # Max 1 week recency
            
            # Action type weights
            action_weight = 1.0
            if activity['action_type'] == 'recommendation_request':
                action_weight = 2.0
            elif activity['action_type'] == 'feedback_provided':
                action_weight = 3.0
            elif activity['action_type'] == 'login':
                action_weight = 0.5
            
            score += (action_weight * recency_weight)
        
        return min(100.0, score * 10) # Max score 100

class PerformanceAnalyzer:
    """Analyzes user investment performance and decision-making."""
    def __init__(self):
        logger.info("PerformanceAnalyzer initialized.")

    def analyze_performance(self, user_recommendations: List[Dict], user_feedback: List[Dict]) -> Dict:
        """
        Analyzes user performance based on recommendations and feedback.
        (Simplified for demonstration)
        """
        total_recommendations = len(user_recommendations)
        followed_recommendations = sum(1 for fb in user_feedback if fb.get('followed_recommendation'))
        
        # Mock accuracy and return for demonstration
        mock_accuracy = np.random.uniform(0.6, 0.9)
        mock_return = np.random.uniform(-0.05, 0.15) # -5% to +15%

        return {
            "total_recommendations": total_recommendations,
            "followed_recommendations": followed_recommendations,
            "mock_accuracy_score": mock_accuracy,
            "mock_average_return": mock_return
        }

class ChurnPredictor:
    """Predicts user churn likelihood."""
    def __init__(self):
        logger.info("ChurnPredictor initialized.")

    def predict_churn_likelihood(self, engagement_score: float, last_login_days: int) -> float:
        """
        Predicts churn likelihood (0-1, higher means more likely to churn).
        Simplified model: inverse of engagement, proportional to days since last login.
        """
        # Lower engagement means higher churn likelihood
        churn_from_engagement = max(0, 1 - (engagement_score / 100))

        # Longer time since last login means higher churn likelihood
        # Max churn from inactivity after 30 days
        churn_from_inactivity = min(1.0, last_login_days / 30.0)

        # Combine (simple average for now)
        likelihood = (churn_from_engagement + churn_from_inactivity) / 2.0
        return likelihood

class UserAnalytics:
    """
    Main User Analytics System.
    Orchestrates behavior tracking, engagement scoring, performance analysis, and churn prediction.
    Processes data in a background thread for real-time insights.
    """
    def __init__(self, config: Dict, db_manager: Optional[FinancialDataFramework] = None):
        self.config = config
        self.analytics_config = config.get('analytics', {})
        self.db_manager: Optional[FinancialDataFramework] = db_manager
        self.is_running = False
        self.shutdown_event = threading.Event() # For graceful shutdown
        
        # Analytics components
        self.behavior_tracker = BehaviorTracker()
        self.engagement_scorer = EngagementScorer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.churn_predictor = ChurnPredictor()

        self._processing_thread: Optional[threading.Thread] = None
        self.last_processed_user_id: Optional[int] = None

        logger.info("User Analytics System initialized.")
    
    def set_database_manager(self, db_manager: FinancialDataFramework):
        """Sets the FinancialDataFramework instance for database interaction."""
        self.db_manager = db_manager
        logger.info("User Analytics: Database manager set.")

    def start_analytics(self):
        """Starts the background thread for processing user analytics."""
        if self.is_running:
            logger.warning("User analytics processing is already running.")
            return
        if not self.db_manager:
            logger.error("Database manager not set. Cannot start user analytics processing.")
            return

        self.is_running = True
        self.shutdown_event.clear()
        self._processing_thread = threading.Thread(target=self._analytics_processing_loop, daemon=True)
        self._processing_thread.start()
        logger.info("Started user analytics processing thread.")

    def stop_analytics(self):
        """Stops the background thread for processing user analytics gracefully."""
        if not self.is_running:
            logger.warning("User analytics processing is not running.")
            return
        self.is_running = False
        self.shutdown_event.set() # Signal the thread to shut down
        if self._processing_thread:
            self._processing_thread.join(timeout=self.analytics_config.get('update_interval', 3600) + 5) # Give it time to finish
        logger.info("User analytics processing thread stopped.")

    def _analytics_processing_loop(self):
        """The main loop for processing user analytics in a background thread."""
        while not self.shutdown_event.is_set():
            try:
                self.process_user_analytics()
                self.process_system_analytics()
            except Exception as e:
                logger.error(f"Error in analytics processing loop: {e}", exc_info=True)
            
            # Wait for the next cycle or until shutdown is signaled
            self.shutdown_event.wait(self.analytics_config.get('update_interval', 3600)) # Default to hourly

    def process_user_analytics(self):
        """
        Processes and updates analytics for all active users.
        This method will be called periodically by the background thread.
        """
        if not self.db_manager:
            logger.error("Database manager not available for user analytics processing.")
            return

        logger.info("Processing user analytics...")
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                # Ensure the 'status' column exists in the 'users' table
                # This check is a safeguard; init_database should create it.
                cursor.execute("PRAGMA table_info(users)")
                columns = [col[1] for col in cursor.fetchall()]
                if 'status' not in columns:
                    logger.warning("Adding 'status' column to 'users' table. This should ideally be handled by init_database.")
                    cursor.execute("ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'active'")
                    conn.commit()

                cursor.execute("SELECT user_id, last_login FROM users WHERE status = 'active'")
                active_users = cursor.fetchall()

            for user in active_users:
                user_id = user['user_id']
                last_login = datetime.fromisoformat(user['last_login']) if user['last_login'] else datetime.utcnow()
                days_since_last_login = (datetime.utcnow() - last_login).days

                # Retrieve user activity and feedback (mock for now)
                user_activity = self.behavior_tracker.get_recent_actions(user_id, limit=100)
                user_feedback = [] # Assuming feedback is stored elsewhere or mocked

                engagement_score = self.engagement_scorer.calculate_score(user_activity)
                performance_summary = self.performance_analyzer.analyze_performance(user_activity, user_feedback)
                churn_likelihood = self.churn_predictor.predict_churn_likelihood(engagement_score, days_since_last_login)

                analytics_data = {
                    "last_processed": datetime.utcnow().isoformat(),
                    "engagement_score": engagement_score,
                    "performance_summary": performance_summary,
                    "churn_likelihood": churn_likelihood,
                    "recent_activity": user_activity # Store recent actions directly
                }
                
                # Save or update analytics data in the database
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO user_analytics (user_id, analytics_data, last_updated)
                        VALUES (?, ?, ?)
                    ''', (user_id, json.dumps(analytics_data), datetime.utcnow().isoformat()))
                    conn.commit()
                logger.debug(f"User analytics updated for user {user_id}.")
            logger.info("User analytics processing cycle completed.")
        except Exception as e:
            logger.error(f"Error in process_user_analytics: {e}", exc_info=True)


    def process_system_analytics(self):
        """
        Processes and updates system-wide analytics.
        This method will be called periodically by the background thread.
        """
        if not self.db_manager:
            logger.error("Database manager not available for system analytics processing.")
            return

        logger.info("Processing system analytics...")
        try:
            # Aggregate data from all users or system-wide metrics
            # For demonstration, let's use some mock aggregated data
            total_users = 10 # Mock
            active_users = 5 # Mock
            avg_engagement = 75.0 # Mock
            total_recommendations_served = 1000 # Mock
            total_errors = 10 # Mock

            system_analytics_data = {
                "last_processed": datetime.utcnow().isoformat(),
                "total_users": total_users,
                "active_users": active_users,
                "average_engagement_score": avg_engagement,
                "total_recommendations_served": total_recommendations_served,
                "total_system_errors": total_errors,
                "system_health_status": "Operational" # Mock
            }

            # Save or update system analytics data in the database
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                # Always update the single system analytics entry (assuming ID 1)
                cursor.execute('''
                    INSERT OR REPLACE INTO system_analytics (id, metric_name, analytics_data, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (1, 'system_health_metrics', json.dumps(system_analytics_data), datetime.utcnow().isoformat()))
                conn.commit()
            logger.debug("System analytics updated.")
        except Exception as e:
            logger.error(f"Error in process_system_analytics: {e}", exc_info=True)

    def get_user_analytics(self, user_id: int) -> Optional[Dict]:
        """Retrieves the latest analytics data for a specific user."""
        if not self.db_manager:
            logger.error("Database manager not available to get user analytics.")
            return {"error": "Database not available."}
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT analytics_data FROM user_analytics WHERE user_id = ?", (user_id,))
                result = cursor.fetchone()
                if result:
                    return json.loads(result['analytics_data'])
                return {"message": "No analytics data found for this user."}
        except Exception as e:
            logger.error(f"Error retrieving user analytics for {user_id}: {e}", exc_info=True)
            return {"error": f"Failed to retrieve user analytics: {str(e)}"}

    def get_system_analytics(self) -> Optional[Dict]:
        """Retrieves the latest system-wide analytics data."""
        if not self.db_manager:
            logger.error("Database manager not available to get system analytics.")
            return {"error": "Database not available."}
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT analytics_data FROM system_analytics WHERE id = 1") # Assuming single entry
                result = cursor.fetchone()
                if result:
                    return json.loads(result['analytics_data'])
                return {"message": "No system analytics data found."}
        except Exception as e:
            logger.error(f"Error retrieving system analytics: {e}", exc_info=True)
            return {"error": f"Failed to retrieve system analytics: {str(e)}"}

# --- Standalone Testing Example ---
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    class MockConfig:
        def __init__(self):
            self.config_data = {
                "analytics": {
                    "enable_realtime": True,
                    "retention_days": 90,
                    "update_interval": 5, # Process every 5 seconds for testing
                    "cache_timeout": 300
                },
                "database": {
                    "path": ":memory:" # Use in-memory for testing
                }
            }
        def get(self, key, default=None):
            return self.config_data.get(key, default)

    mock_config = MockConfig()

    # Create a dummy database file if not in-memory, to ensure path exists
    db_path = mock_config.get('database', {}).get('path')
    if db_path != ':memory:' and not os.path.exists(os.path.dirname(db_path)):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    class MockDataFrameworkForAnalytics(FinancialDataFramework):
        def __init__(self, db_path: str = ':memory:'):
            super().__init__({'database': {'path': db_path}})
            self.init_database() # Ensure tables are created for mock

        def init_database(self):
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        user_id INTEGER PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        email TEXT UNIQUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP,
                        status TEXT DEFAULT 'active'
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER UNIQUE,
                        analytics_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_analytics (
                        id INTEGER PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metric_name TEXT NOT NULL,
                        value REAL,
                        analytics_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_activities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        activity_type TEXT,
                        activity_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
                # Insert some mock users for testing
                cursor.execute("INSERT OR IGNORE INTO users (user_id, username, password_hash, last_login, status) VALUES (?, ?, ?, ?, ?)",
                               (1, 'user1', 'hashedpass1', datetime.now().isoformat(), 'active'))
                cursor.execute("INSERT OR IGNORE INTO users (user_id, username, password_hash, last_login, status) VALUES (?, ?, ?, ?, ?)",
                               (2, 'user2', 'hashedpass2', (datetime.now() - timedelta(days=10)).isoformat(), 'active'))
                conn.commit()

        @contextlib.contextmanager
        def get_connection(self) -> sqlite3.Connection:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    mock_db_manager = MockDataFrameworkForAnalytics()
    analytics = UserAnalytics(mock_config)
    analytics.set_database_manager(mock_db_manager)
    
    # Simulate some user activity
    analytics.behavior_tracker.track_user_action(1, 'recommendation_request', {'item': 'GLD'})
    analytics.behavior_tracker.track_user_action(1, 'feedback_provided', {'rec_id': 'abc', 'score': 5})
    analytics.behavior_tracker.track_user_action(1, 'login', {})
    analytics.behavior_tracker.track_user_action(2, 'view_report', {'report_type': 'daily'})

    # Start analytics processing in background
    analytics.start_analytics()
    
    # Give some time for the background processing to run
    print("Waiting for initial analytics processing (7 seconds)...")
    await asyncio.sleep(7) # Wait a bit longer than update_interval

    # Test getting user analytics
    user_analytics = analytics.get_user_analytics(1)
    print("\nUser Analytics for User 1:")
    print(json.dumps(user_analytics, indent=2))
    
    # Test getting system analytics
    system_analytics = analytics.get_system_analytics()
    print("\nSystem Analytics:")
    print(json.dumps(system_analytics, indent=2))

    # Test user 2 (should get default/newly generated)
    user_analytics_2 = analytics.get_user_analytics(2)
    print("\nUser Analytics for User 2 (new user or less active):")
    print(json.dumps(user_analytics_2, indent=2))

    # Stop analytics gracefully
    analytics.stop_analytics()
    print("\nUser Analytics System testing complete.")

if __name__ == "__main__":
    try:
        import asyncio
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("User Analytics System example interrupted by user.")
    except Exception as e:
        logger.critical(f"User Analytics System example failed: {e}", exc_info=True)
        sys.exit(1)