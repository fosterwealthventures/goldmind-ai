# ADVANCED FEATURES & PRODUCTION OPTIMIZATION FOR GOLDMIND AI
# Integrates real-time analytics, adaptive learning, market regime detection,
# and enhanced user personalization into the ultimate recommendation engine.

import asyncio
import redis
from sqlalchemy import create_engine, text # Keep if you plan to use SQLAlchemy, otherwise can remove
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Callable
import logging
from concurrent.futures import ThreadPoolExecutor
import json
import uuid
import sqlite3 # Used for Optional[sqlite3.Row] type hint in UserPersonalizationEngine
import os # For os.getenv and path ops in mock FDF
import sys # For sys.exit in main block
import shutil # For cleanup in main block

logger = logging.getLogger(__name__)

# --- CORE EXTERNAL MODULE IMPORTS with PLACEHOLDER FALLBACK ---
# This structure ensures that if a module fails to import, a basic placeholder
# is available, preventing NameErrors and allowing the application to start
# in a degraded (but runnable) state.

# FinancialDataFramework is critical, its mock is defined before import attempts
class FinancialDataFramework:
    def __init__(self, db_path: Optional[str] = None):
        logging.warning("Using FinancialDataFramework placeholder. Functionality will be limited.")
        self.db_path = db_path or ':memory:' # Ensure a dummy path for init_database
        self._init_mock_database()

    def _init_mock_database(self):
        """Initializes tables for the mock FinancialDataFramework."""
        try:
            # Create directory if path is not :memory: and has one
            if self.db_path != ':memory__':
                db_dir = os.path.dirname(self.db_path)
                if db_dir:
                    os.makedirs(db_dir, exist_ok=True)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Ensure tables needed by UserPersonalizationEngine and _store_recommendation exist in mock
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recommendations (
                    id TEXT PRIMARY KEY, user_id INTEGER, final_action TEXT, final_confidence REAL,
                    detailed_reasoning TEXT, bias_analysis TEXT, timestamp TIMESTAMP,
                    predicted_price REAL, actual_price REAL, entry_price REAL, prediction_accuracy REAL,
                    actual_return REAL, predicted_return REAL, pathway_used TEXT, bias_detected BOOLEAN, bias_adjusted BOOLEAN,
                    conflict_severity TEXT, resolution_method TEXT, pathway_weights TEXT,
                    alternative_scenarios TEXT, market_regime TEXT, personalization_applied BOOLEAN,
                    final_target_price REAL, final_stop_loss REAL, final_position_size REAL, followed_recommendation BOOLEAN
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id INTEGER PRIMARY KEY,
                    risk_tolerance REAL,
                    time_horizon TEXT,
                    preferred_pathway TEXT,
                    profile_data TEXT
                )
            ''')
            cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, email TEXT UNIQUE, password_hash TEXT)") # Basic users table for foreign key if needed
            conn.commit()
            conn.close()
            logging.info(f"Mock DB for FinancialDataFramework placeholder setup at {self.db_path}.")
        except Exception as e:
            logging.warning(f"Failed to setup mock DB for FinancialDataFramework placeholder: {e}")

    def get_connection(self):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logging.error(f"Failed to get mock DB connection: {e}")
            return None
    async def get_stock_data(self, *args, **kwargs): return pd.DataFrame()
    async def get_economic_data(self, *args, **kwargs): return pd.DataFrame()
    async def get_gold_ohlcv_data(self, *args, **kwargs): return pd.DataFrame() # Add mock for gold OHLCV
    def get_usage_report(self): return {'apis': {}} # Mock for monitoring

# Overwrite placeholder with real import if successful
try:
    from financial_data_framework import FinancialDataFramework as RealFinancialDataFramework
    FinancialDataFramework = RealFinancialDataFramework
    logging.info("✅ FinancialDataFramework loaded successfully.")
except ImportError as e:
    logging.critical(f"❌ Failed to import FinancialDataFramework. Using placeholder. Error: {e}")

# Define PathwayType here as it's used by multiple modules/placeholders
class PathwayType:
    ANALYTICAL = "ANALYTICAL"
    MACHINE_LEARNING = "MACHINE_LEARNING"
    LSTM_TEMPORAL = "LSTM_TEMPORAL"
    HYBRID = "HYBRID"

# Try importing core AI system dependencies, falling back to mocks if needed
# Define placeholder classes first to ensure they are available even if the try block fails
class BiasAwareDualSystemManager:
    def __init__(self, config: Dict, goldmind_client: Any = None):
        logging.warning("Using BiasAwareDualSystemManager placeholder.")
        self.config = config
        self.goldmind_client = goldmind_client
        # Initialize placeholder resolver - ensure it receives a config
        self.conflict_resolver = DualSystemConflictResolver(config=self.config.get('dual_system', {}))
    async def generate_bias_aware_recommendation(self, market_data: pd.DataFrame, user_context: Optional[Dict] = None, user_input: Optional[str] = None, model: Optional[Any] = None, data_framework: Optional[Any] = None) -> Dict:
        logging.warning("Mock BiasAwareDualSystemManager: Generating fallback recommendation.")
        return {'success': False, 'error': 'BiasAwareDualSystemManager not available', 'recommendation': {'action': 'HOLD', 'confidence': 0.5}}
    def get_bias_trends(self, *args, **kwargs): return {}
    def _default_config(self): return {}
    def _get_default_user_context(self): return {}
    def _format_recommendation_response(self, *args, **kwargs): return {}
    def _analyze_user_input_biases(self, *args, **kwargs): return []
    def _analyze_pathway_biases(self, *args, **kwargs): return {}
    def _enhance_context_with_bias_analysis(self, *args, **kwargs): return {}
    def _apply_bias_adjustments(self, *args, **kwargs): return None
    def _generate_bias_recommendations(self, *args, **kwargs): return []
    def _update_bias_history(self, *args, **kwargs): pass
    def generate_bias_report(self, *args, **kwargs): return "Mock Bias Report"

@dataclass
class PathwayRecommendation:
    pathway_type: str
    action: str
    confidence: float
    target_price: float
    stop_loss: float
    position_size: float
    reasoning: str
    technical_indicators: Dict[str, Any]
    risk_score: float
    timestamp: datetime

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
    _original_confidence: Optional[float] = None
    _original_position_size: Optional[float] = None

class DualSystemConflictResolver:
    def __init__(self, config: Dict, db_manager: Optional[FinancialDataFramework] = None):
        logging.warning("Using DualSystemConflictResolver placeholder.")
        self.config = config
        self.pathway_weights = {
            PathwayType.ANALYTICAL: 0.3,
            PathwayType.MACHINE_LEARNING: 0.3,
            PathwayType.LSTM_TEMPORAL: 0.4
        }
    def resolve_recommendations(self, analytical_rec, ml_rec, lstm_rec=None, user_context=None):
        logging.warning("Mock DualSystemConflictResolver: Returning default HOLD recommendation.")
        return ConflictResolution(
            final_action="HOLD", final_confidence=0.5, final_target_price=0.0,
            final_stop_loss=0.0, final_position_size=0.0, conflict_severity="MINOR",
            resolution_method="Mock Resolution", pathway_weights={},
            consensus_score=0.5, uncertainty_factor=0.5, detailed_reasoning="Mocked due to import failure.",
            alternative_scenarios=[]
        )

class LSTMTemporalAnalysis:
    def __init__(self, financial_data_framework: Any):
        logging.warning("Using LSTMTemporalAnalysis placeholder.")
        self.data_framework = financial_data_framework
        self.models = {} # Models dictionary (empty in placeholder)
    def _generate_synthetic_data(self, days):
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        base_price = np.random.rand(days) * 100 + 100
        df = pd.DataFrame(
            {
                'open': base_price * (1 + np.random.uniform(-0.005, 0.005, days)),
                'high': base_price * (1 + np.random.uniform(0.005, 0.01, days)),
                'low': base_price * (1 - np.random.uniform(0.005, 0.01, days)),
                'close': base_price,
                'volume': np.random.randint(1000, 10000, days)
            },
            index=dates
        )
        df['price'] = df['close'] # Ensure 'price' column is always present
        return df
    async def predict_stock_price(self, ticker: str) -> Dict:
        logging.warning("Mock LSTMTemporalAnalysis: Returning dummy prediction.")
        recent_data = self._generate_synthetic_data(days=60)
        return 105.0, recent_data
    async def train_models(self, ticker, start_date, end_date, epochs: int = 1): # Match real signature
        logging.warning(f"Mock LSTMTemporalAnalysis: Simulating training for {epochs} epochs.")
        self.models[ticker] = True # Simulate model being "trained"
        return True # Return True on success for placeholder consistency
    def save_models(self):
        logging.warning("Mock LSTMTemporalAnalysis: Simulating model saving.")
    def load_models(self):
        logging.warning("Mock LSTMTemporalAnalysis: Simulating model loading (no actual loading in mock).")

try:
    # Attempt real imports, these will overwrite the placeholder classes if successful
    from BIAS_AWARE_DUAL_SYSTEM_INTEGRATION import BiasAwareDualSystemManager as RealBiasAwareDualSystemManager
    from dual_system_conflict_resolution import (
        DualSystemConflictResolver as RealDualSystemConflictResolver,
        PathwayRecommendation,
        ConflictResolution
    )
    from lstm_temporal_analysis import LSTMTemporalAnalysis as RealLSTMTemporalAnalysis

    BiasAwareDualSystemManager = RealBiasAwareDualSystemManager
    DualSystemConflictResolver = RealDualSystemConflictResolver
    LSTMTemporalAnalysis = RealLSTMTemporalAnalysis

    logging.info("✅ Core AI System dependencies (BiasAwareDualSystemManager, DualSystemConflictResolver, LSTMTemporalAnalysis) loaded successfully.")

except ImportError as e:
    logging.critical(f"❌ Failed to import a core AI System dependency (e.g., BIAS_AWARE_DUAL_SYSTEM_INTEGRATION, dual_system_conflict_resolution, lstm_temporal_analysis). Using placeholders. Error: {e}")


# AdvancedAnalyticsManager Placeholder
class AdvancedAnalyticsManager:
    def __init__(self, data_framework: FinancialDataFramework):
        self.data_framework = data_framework
        logging.warning("Using AdvancedAnalyticsManager placeholder. Analytics functionality will be limited.")
    def track_recommendation_performance(self, recommendation_id: str, user_id: int, actual_outcome: Dict, recommendation_output: Dict) -> Optional[float]:
        logging.info(f"Mock Analytics: Tracking performance for rec_id={recommendation_id}")
        return 0.9 # Placeholder accuracy
    def get_system_metrics(self) -> Dict:
        return {"total_recommendations": 100, "average_accuracy": 0.85} # Mock metrics

try:
    from advanced_analytics import AdvancedAnalyticsManager as RealAdvancedAnalyticsManager
    AdvancedAnalyticsManager = RealAdvancedAnalyticsManager
    logging.info("✅ AdvancedAnalyticsManager loaded successfully.")
except ImportError as e:
    logging.critical(f"❌ Failed to import AdvancedAnalyticsManager. Using placeholder. Error: {e}")


# ============================================================================
# USER PERSONALIZATION ENGINE
# ============================================================================
class UserPersonalizationEngine:
    """
    Learn user preferences and trading patterns for personalized recommendations.
    Builds comprehensive user profiles from trading history and preferences stored in DB.
    """

    def __init__(self, db_manager: FinancialDataFramework):
        self.db = db_manager
        self.user_profiles = {} # In-memory cache for user profiles
        self.cache_timeout = 300 # 5 minutes

        logger.info("UserPersonalizationEngine initialized.")

    async def build_user_profile(self, user_id: int) -> Dict:
        """Build comprehensive user profile from trading history and preferences."""
        try:
            # Check in-memory cache first
            if user_id in self.user_profiles:
                if (datetime.utcnow() - self.user_profiles[user_id]['_cached_at']).total_seconds() < self.cache_timeout:
                    logger.debug(f"Serving user profile for {user_id} from cache.")
                    return self.user_profiles[user_id]['profile']

            # Fetch data from database
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Get user trading history (last 100 trades)
                cursor.execute('''
                    SELECT
                        id AS recommendation_id, user_id, final_action, final_confidence,
                        detailed_reasoning, bias_analysis, timestamp,
                        final_target_price,
                        actual_price,
                        entry_price,
                        prediction_accuracy,
                        actual_return,
                        predicted_return,
                        pathway_used,
                        bias_detected,
                        bias_adjusted,
                        followed_recommendation
                    FROM recommendations
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 100
                ''', (user_id,))

                history_rows = cursor.fetchall()

                # Get user_profile_data from user_profiles table
                cursor.execute('''
                    SELECT profile_data, risk_tolerance, time_horizon, preferred_pathway
                    FROM user_profiles
                    WHERE user_id = ?
                ''', (user_id,))
                user_profile_db = cursor.fetchone()

                if not history_rows and not user_profile_db:
                    logger.info(f"No history or profile data for user {user_id}. Returning default profile.")
                    return self._default_user_profile(user_id)

                # Process history data
                history = []
                for row in history_rows:
                    # Convert bias_analysis JSON string back to dict
                    bias_analysis_dict = json.loads(row['bias_analysis']) if row['bias_analysis'] else {}
                    # Ensure all fields expected by analysis methods are present
                    history.append({
                        'recommendation_id': row['recommendation_id'],
                        'user_id': row['user_id'],
                        'final_action': row['final_action'],
                        'final_confidence': row['final_confidence'],
                        'detailed_reasoning': row['detailed_reasoning'],
                        'bias_detected': bool(row['bias_detected']),
                        'bias_adjusted': bool(row['bias_adjusted']),
                        'timestamp': datetime.fromisoformat(row['timestamp']) if isinstance(row['timestamp'], str) else row['timestamp'],
                        'predicted_price': row['predicted_price'],
                        'actual_price': row['actual_price'],
                        'entry_price': row['entry_price'],
                        'prediction_accuracy': row['prediction_accuracy'],
                        'actual_return': row['actual_return'],
                        'predicted_return': row['predicted_return'],
                        'primary_pathway': row['pathway_used'],
                        'followed_recommendation': row['followed_recommendation']
                    })

                # Analyze patterns
                profile = {
                    'user_id': user_id,
                    'trading_patterns': self._analyze_trading_patterns(history),
                    'risk_preferences': self._analyze_risk_preferences(history, user_profile_db),
                    'bias_susceptibility': self._analyze_bias_patterns(history),
                    'pathway_preferences': self._analyze_pathway_preferences(history, user_profile_db),
                    'performance_metrics': self._calculate_user_performance(history),
                    'last_updated': datetime.utcnow().isoformat()
                }

                # Merge with DB stored profile data if exists
                if user_profile_db:
                    db_profile_data = json.loads(user_profile_db['profile_data']) if user_profile_db['profile_data'] else {}
                    profile.update(db_profile_data)
                    profile['risk_preferences']['avg_risk_tolerance'] = user_profile_db['risk_tolerance']
                    profile['time_horizon'] = user_profile_db['time_horizon']
                    profile['preferred_pathway'] = user_profile_db['preferred_pathway']

                # Store in-memory cache
                self.user_profiles[user_id] = {'profile': profile, '_cached_at': datetime.utcnow()}

                return profile

        except Exception as e:
            logger.error(f"Failed to build user profile for {user_id}: {str(e)}", exc_info=True)
            return self._default_user_profile(user_id)

    def _analyze_trading_patterns(self, history: List[Dict]) -> Dict:
        """Analyze trading patterns from user history."""
        if not history:
            return {'average_confidence': 0.0, 'preferred_actions': {}, 'volatility_preference': 'medium'}

        total_confidence = sum(h['final_confidence'] for h in history if h['final_confidence'] is not None)
        avg_confidence = total_confidence / len(history) if history else 0

        action_counts = pd.Series([h['final_action'] for h in history if h['final_action']]).value_counts().to_dict()

        confidence_values = [h['final_confidence'] for h in history if h['final_confidence'] is not None]
        confidence_std = np.std(confidence_values) if len(confidence_values) > 1 else 0
        volatility_preference = 'low'
        if confidence_std > 0.1:
            volatility_preference = 'high'
        elif confidence_std > 0.05:
            volatility_preference = 'medium'

        return {
            'average_confidence': float(avg_confidence),
            'preferred_actions': action_counts,
            'volatility_preference': volatility_preference
        }

    def _analyze_risk_preferences(self, history: List[Dict], user_profile_db: Optional[sqlite3.Row]) -> Dict:
        """Analyze user's risk preferences."""
        risk_tolerance = user_profile_db['risk_tolerance'] if user_profile_db and 'risk_tolerance' in user_profile_db.keys() and user_profile_db['risk_tolerance'] is not None else 0.5
        time_horizon = user_profile_db['time_horizon'] if user_profile_db and 'time_horizon' in user_profile_db.keys() and user_profile_db['time_horizon'] else 'medium'

        followed_risky_recs = 0
        total_recs = 0
        for h in history:
            if h.get('final_confidence', 0) < 0.7:
                total_recs += 1
            if h.get('followed_recommendation'):
                followed_risky_recs += 1

        if total_recs > 0:
            risk_tolerance_adjustment = (followed_risky_recs / total_recs - 0.5) * 0.2
            risk_tolerance = max(0.0, min(1.0, risk_tolerance + risk_tolerance_adjustment))

        return {
            'avg_risk_tolerance': float(risk_tolerance),
            'inferred_time_horizon': time_horizon,
            'risk_aversion_score': 1.0 - float(risk_tolerance)
        }

    def _analyze_bias_patterns(self, history: List[Dict]) -> Dict:
        """Analyze user's susceptibility to cognitive biases."""
        bias_counts = {}
        total_biased_recs = 0
        total_adjusted_recs = 0

        for h in history:
            if h.get('bias_detected'):
                total_biased_recs += 1
            if h.get('bias_adjusted'):
                total_adjusted_recs += 1

        return {
            'total_biased_recommendations': total_biased_recs,
            'total_adjusted_recommendations': total_adjusted_recs,
            'adjustment_rate': float(total_adjusted_recs / max(1, total_biased_recs)),
            'common_biases': {'confirmation_bias_risk': 0.6, 'anchoring_bias_risk': 0.4}
        }

    def _analyze_pathway_preferences(self, history: List[Dict], user_profile_db: Optional[sqlite3.Row]) -> Dict:
        """Analyze user's preference for specific recommendation pathways."""
        preferred_pathway_from_db = user_profile_db['preferred_pathway'] if user_profile_db and 'preferred_pathway' in user_profile_db.keys() else None

        pathway_counts = pd.Series([
            h['primary_pathway'] for h in history if 'primary_pathway' in h
        ]).value_counts().to_dict()

        if pathway_counts:
            most_frequent_pathway = max(pathway_counts, key=pathway_counts.get)
            if preferred_pathway_from_db and preferred_pathway_from_db != most_frequent_pathway:
                logger.debug(f"User {history[0]['user_id']} historical pathway preference ({most_frequent_pathway}) differs from stored ({preferred_pathway_from_db}).")

            return {
                'historical_preferences': pathway_counts,
                'inferred_preferred_pathway': most_frequent_pathway,
                'explicit_preferred_pathway': preferred_pathway_from_db
            }
        else:
            return {
                'historical_preferences': {},
                'inferred_preferred_pathway': None,
                'explicit_preferred_pathway': preferred_pathway_from_db
            }

    def _calculate_user_performance(self, history: List[Dict]) -> Dict:
        """Calculate user's average performance and success rates."""
        if not history:
            return {'avg_prediction_accuracy': 0.0, 'avg_actual_return': 0.0, 'followed_recommendation_rate': 0.0}

        prediction_accuracies = [h['prediction_accuracy'] for h in history if 'prediction_accuracy' in h and h['prediction_accuracy'] is not None]
        actual_returns = [h['actual_return'] for h in history if 'actual_return' in h and h['actual_return'] is not None]
        followed_count = sum(1 for h in history if h.get('followed_recommendation'))

        avg_accuracy = sum(prediction_accuracies) / len(prediction_accuracies) if prediction_accuracies else 0.0
        avg_return = sum(actual_returns) / len(actual_returns) if actual_returns else 0.0
        followed_rate = followed_count / len(history) if history else 0.0

        return {
            'avg_prediction_accuracy': float(avg_accuracy),
            'avg_actual_return': float(avg_return),
            'followed_recommendation_rate': float(followed_rate)
        }

    def _default_user_profile(self, user_id: int) -> Dict:
        """Generate a default user profile."""
        logger.info(f"Generating default user profile for user {user_id}.")
        return {
            'user_id': user_id,
            'trading_patterns': {'average_confidence': 0.7, 'preferred_actions': {'BUY': 0.6, 'HOLD': 0.3, 'SELL': 0.1}, 'volatility_preference': 'medium'},
            'risk_preferences': {'avg_risk_tolerance': 0.5, 'inferred_time_horizon': 'medium', 'risk_aversion_score': 0.5},
            'bias_susceptibility': {'total_biased_recommendations': 0, 'total_adjusted_recommendations': 0, 'adjustment_rate': 0.0, 'common_biases': {}},
            'pathway_preferences': {'historical_preferences': {}, 'inferred_preferred_pathway': None, 'explicit_preferred_pathway': None},
            'performance_metrics': {'avg_prediction_accuracy': 0.0, 'avg_actual_return': 0.0, 'followed_recommendation_rate': 0.0},
            'last_updated': datetime.utcnow().isoformat()
        }


# ============================================================================
# MARKET REGIME DETECTOR
# ============================================================================
class MarketRegimeDetector:
    def __init__(self, data_framework: FinancialDataFramework, lstm_temporal_analyzer: LSTMTemporalAnalysis):
        self.data_framework = data_framework
        self.lstm = lstm_temporal_analyzer
        logging.warning("Using MarketRegimeDetector placeholder.")

    async def detect_current_regime(self, market_data: pd.DataFrame) -> Dict:
        logging.info("Mock MarketRegimeDetector: Detecting market regime.")
        try:
            data_to_use = market_data if not market_data.empty else self.lstm._generate_synthetic_data(days=100)

            latest_close = data_to_use['close'].iloc[-1]
            prev_close = data_to_use['close'].iloc[-20]

            regime = "sideways"
            sentiment = "neutral"

            if latest_close > prev_close * 1.05:
                regime = "trending"
                sentiment = "bullish"
            elif latest_close < prev_close * 0.95:
                regime = "trending"
                sentiment = "bearish"
            elif data_to_use['close'].tail(20).std() / data_to_use['close'].tail(20).mean() > 0.03:
                regime = "volatile"
                sentiment = "uncertain"

            return {
                "regime": regime,
                "sentiment": sentiment,
                "confidence": 0.8,
                "recommended_adjustments": {
                    "position_size_multiplier": 0.7 if regime == "volatile" else 1.1 if regime == "trending" else 1.0,
                    "preferred_pathways": ["ANALYTICAL"] if regime == "trending" else ["MACHINE_LEARNING"]
                }
            }
        except Exception as e:
            logging.error(f"Error in mock MarketRegimeDetector: {e}", exc_info=True)
            return {"regime": "unknown", "sentiment": "unknown", "confidence": 0.0, "recommended_adjustments": {}}


# ============================================================================
# ADAPTIVE LEARNING SYSTEM
# ============================================================================
class AdaptiveLearningSystem:
    def __init__(self, bias_manager: BiasAwareDualSystemManager, analytics_manager: AdvancedAnalyticsManager):
        self.bias_manager = bias_manager
        self.analytics_manager = analytics_manager
        self.adaptation_threshold = 0.05
        logging.warning("Using AdaptiveLearningSystem placeholder.")

    def adapt_pathway_weights(self) -> Dict:
        logging.info("Mock AdaptiveLearningSystem: Adapting pathway weights.")
        weights = self.bias_manager.conflict_resolver.pathway_weights.copy()
        if PathwayType.ANALYTICAL in weights:
            weights[PathwayType.ANALYTICAL] = min(0.5, weights[PathwayType.ANALYTICAL] + 0.01)
        if PathwayType.LSTM_TEMPORAL in weights:
            weights[PathwayType.LSTM_TEMPORAL] = max(0.2, weights[PathwayType.LSTM_TEMPORAL] - 0.01)
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def learn_from_user_feedback(self, recommendation_id: str, feedback: Dict):
        logging.info(f"Mock AdaptiveLearningSystem: Learning from feedback for {recommendation_id}: {feedback}")

    def _calculate_weight_improvement(self, initial_weights: Dict, adapted_weights: Dict) -> float:
        return 0.06


# ============================================================================
# ULTIMATE BIAS-AWARE RECOMMENDATION MANAGER (ORCHESTRATOR)
# ============================================================================

class UltimateBiasAwareManager:
    """
    Orchestrates advanced features to generate highly personalized, bias-aware,
    and market-adaptive financial recommendations.
    """
    def __init__(self, config: Dict, financial_data_framework: Any, redis_client: Any = None, config_manager_instance: Any = None, lstm_temporal_analyzer: Any = None): # NEW: Accept pre-initialized LSTM analyzer
        self.config = config
        self.financial_data_framework = financial_data_framework
        self.redis_client = redis_client
        self.config_manager = config_manager_instance # Use this for secrets

        # Goldmind client initialization
        goldmind_api_key_val = None
        if self.config_manager and hasattr(self.config_manager, 'secrets') and \
           'api_keys' in self.config_manager.secrets and 'goldmind' in self.config_manager.secrets['api_keys']:
            goldmind_api_key_val = self.config_manager.secrets['api_keys']['goldmind']
        else:
            goldmind_api_key_val = os.getenv('GOLDMIND_API_KEY')
            logger.warning("GoldMIND API key not found in config_manager_instance. Falling back to GOLDMIND_API_KEY environment variable.")


        try:
            from goldmind_integration import Goldmind
            goldmind_client = Goldmind(api_key=goldmind_api_key_val)
            logger.info("✅ Goldmind client initialized successfully.")
        except ImportError:
            logging.critical("❌ goldmind_integration not found. Goldmind client will be a mock.")
            class Goldmind: # Mock Goldmind client
                def __init__(self, *args, **kwargs): pass
                async def search(self, query: str, *args, **kwargs) -> List[Dict]:
                    logging.warning("Mock Goldmind Client: Returning dummy bias analysis.")
                    return [{"content": "Mock bias analysis: confirmation", "bias_type": "confirmation"}]
            goldmind_client = Goldmind()
        except Exception as e:
            logging.critical(f"❌ Error initializing Goldmind client: {e}. Goldmind client will be a mock.", exc_info=True)
            class Goldmind: # Mock Goldmind client
                def __init__(self, *args, **kwargs): pass
                async def search(self, query: str, *args, **kwargs) -> List[Dict]:
                    logging.warning("Mock Goldmind Client: Returning dummy bias analysis.")
                    return [{"content": "Mock bias analysis: confirmation", "bias_type": "confirmation"}]
            goldmind_client = Goldmind()

        self.bias_aware_manager = BiasAwareDualSystemManager(
            config=self.config.get('dual_system', {}),
            goldmind_client=goldmind_client
        )
        
        # CORRECTED: This line now correctly instantiates the LSTMTemporalAnalysis class
        # with the expected `financial_data_framework` argument, avoiding the TypeError.
        self.lstm_temporal_analyzer = LSTMTemporalAnalysis(financial_data_framework=self.financial_data_framework)
        
        # Handle the case where lstm_temporal_analyzer might still be None (e.g., in a test mock)
        if self.lstm_temporal_analyzer:
            logger.info("LSTMTemporalAnalysis instance successfully created.")

        self.market_regime_detector = MarketRegimeDetector(
            data_framework=self.financial_data_framework,
            lstm_temporal_analyzer=self.lstm_temporal_analyzer
        )
        self.user_personalization_engine = UserPersonalizationEngine(db_manager=self.financial_data_framework)
        self.adaptive_learning_system = AdaptiveLearningSystem(
            bias_manager=self.bias_aware_manager,
            analytics_manager=AdvancedAnalyticsManager(data_framework=self.financial_data_framework)
        )
        self.thread_executor = ThreadPoolExecutor(max_workers=5)
        self.recommendation_store_lock = asyncio.Lock()
        
        logger.info("UltimateBiasAwareManager initialized and components ready.")

    async def generate_recommendation(self, ticker: str, user_id: int, user_input: str) -> Dict:
        """
        Main orchestration method to generate a final recommendation.
        """
        try:
            market_data = await self.financial_data_framework.get_stock_data(ticker)
            if market_data.empty:
                return {'success': False, 'error': f'No market data found for {ticker}'}

            user_profile = await self.user_personalization_engine.build_user_profile(user_id)
            market_regime = await self.market_regime_detector.detect_current_regime(market_data)

            # Generate recommendations from different pathways
            # For simplicity, we'll use a single unified call to a mock bias manager.
            # In a real implementation, you'd call individual models (analytical, ML, LSTM)
            # and pass their outputs to the ConflictResolver.
            
            # The BIAS_AWARE_DUAL_SYSTEM_INTEGRATION module is meant to be the orchestrator for this.
            recommendation = await self.bias_aware_manager.generate_bias_aware_recommendation(
                market_data=market_data,
                user_context=user_profile,
                user_input=user_input,
                model=self.lstm_temporal_analyzer, # Pass the LSTM model as an argument
                data_framework=self.financial_data_framework
            )

            # Store the recommendation asynchronously
            asyncio.create_task(self._store_recommendation(recommendation, user_id, market_regime, user_profile))
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error in generate_recommendation for {ticker}: {e}", exc_info=True)
            return {'success': False, 'error': f'Internal error during recommendation generation: {str(e)}'}

    async def _store_recommendation(self, recommendation: Dict, user_id: int, market_regime: Dict, user_profile: Dict):
        """Asynchronously stores the final recommendation and its context in the database."""
        async with self.recommendation_store_lock:
            try:
                with self.financial_data_framework.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    rec_data = recommendation['recommendation']
                    final_resolution = recommendation.get('final_resolution', {})
                    bias_analysis = recommendation.get('bias_analysis', {})
                    
                    rec_id = str(uuid.uuid4())
                    
                    cursor.execute('''
                        INSERT INTO recommendations (
                            id, user_id, final_action, final_confidence, detailed_reasoning,
                            bias_analysis, timestamp, predicted_price, actual_price, entry_price,
                            prediction_accuracy, actual_return, predicted_return, pathway_used,
                            bias_detected, bias_adjusted, conflict_severity, resolution_method,
                            pathway_weights, alternative_scenarios, market_regime, personalization_applied,
                            final_target_price, final_stop_loss, final_position_size
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        rec_id,
                        user_id,
                        rec_data.get('action', 'HOLD'),
                        rec_data.get('confidence', 0.0),
                        final_resolution.get('detailed_reasoning', ''),
                        json.dumps(bias_analysis),
                        datetime.now().isoformat(),
                        rec_data.get('target_price', 0.0), # Assuming 'predicted_price' is 'target_price' from the recommendation
                        0.0, # Actual price, to be filled later
                        0.0, # Entry price, to be filled later
                        0.0, # Accuracy, to be filled later
                        0.0, # Actual return, to be filled later
                        0.0, # Predicted return, to be filled later
                        final_resolution.get('pathway_used', 'N/A'),
                        bias_analysis.get('bias_detected', False),
                        bias_analysis.get('bias_adjusted', False),
                        final_resolution.get('conflict_severity', 'N/A'),
                        final_resolution.get('resolution_method', 'N/A'),
                        json.dumps(final_resolution.get('pathway_weights', {})),
                        json.dumps(final_resolution.get('alternative_scenarios', [])),
                        market_regime.get('regime', 'N/A'),
                        True if user_profile.get('user_id') else False,
                        rec_data.get('target_price', 0.0),
                        rec_data.get('stop_loss', 0.0),
                        rec_data.get('position_size', 0.0)
                    ))
                    conn.commit()
                    logger.info(f"Recommendation for user {user_id} stored with ID: {rec_id}")

            except Exception as e:
                logger.error(f"Failed to store recommendation for user {user_id}: {e}", exc_info=True)


    def _run_in_executor(self, func: Callable, *args, **kwargs):
        """Helper to run a synchronous function in a thread pool."""
        return self.thread_executor.submit(func, *args, **kwargs)

    async def track_and_learn_from_feedback(self, recommendation_id: str, user_id: int, feedback: Dict):
        """
        Processes user feedback and updates the adaptive learning system.
        """
        try:
            # Update user profile based on feedback
            await self._update_user_profile_with_feedback(user_id, recommendation_id, feedback)
            
            # Use analytics manager to track performance
            # This would likely involve fetching the recommendation and actual data
            # for performance calculation.
            self.adaptive_learning_system.learn_from_user_feedback(recommendation_id, feedback)
            
            logger.info(f"Processed feedback for recommendation {recommendation_id} from user {user_id}.")

        except Exception as e:
            logger.error(f"Error processing feedback for {recommendation_id}: {e}", exc_info=True)
    
    async def _update_user_profile_with_feedback(self, user_id: int, recommendation_id: str, feedback: Dict):
        """Updates the user profile and recommendation history based on user feedback."""
        with self.financial_data_framework.get_connection() as conn:
            cursor = conn.cursor()
            
            followed_recommendation = feedback.get('followed', False)
            actual_return = feedback.get('actual_return', 0.0)
            
            cursor.execute('''
                UPDATE recommendations
                SET followed_recommendation = ?, actual_return = ?
                WHERE id = ? AND user_id = ?
            ''', (followed_recommendation, actual_return, recommendation_id, user_id))
            
            # In a real system, you would update the user_profiles table as well,
            # perhaps re-running the build_user_profile logic periodically to reflect
            # new data.
            conn.commit()
            logger.info(f"Updated recommendation {recommendation_id} with user feedback.")
            
# If this file is run directly, it can act as a standalone test script
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Simple setup for demonstration
    test_config = {
        "bias_detection": {
            "is_active": True,
            "bias_threshold": 0.05,
            "detection_interval": 60
        },
        "dual_system": {}, # Add a dummy dual system config
        "database": {"path": "test_db.db"}
    }
    
    # Initialize the framework, which will create the mock db
    test_framework = FinancialDataFramework(db_path=test_config.get("database", {}).get("path"))
    
    # Initialize the manager with mock components
    try:
        # We pass a mock LSTMTemporalAnalysis instance to avoid the init error in a test context
        lstm_mock_instance = LSTMTemporalAnalysis(financial_data_framework=test_framework)
        manager = UltimateBiasAwareManager(
            config=test_config,
            financial_data_framework=test_framework,
            lstm_temporal_analyzer=lstm_mock_instance
        )
        
        # Simple test run (you'd need to run this in an async loop for a full test)
        print("Testing UltimateBiasAwareManager...")
        # Simulate generating a recommendation
        # result = asyncio.run(manager.generate_recommendation("MSFT", 1, "Should I buy MSFT?"))
        # print(f"Test recommendation result: {result}")
        
    except Exception as e:
        logger.error(f"Standalone test failed: {e}", exc_info=True)
    
    print("Cleaning up test database...")
    if os.path.exists("test_db.db"):
        os.remove("test_db.db")
    print("Cleanup complete.")