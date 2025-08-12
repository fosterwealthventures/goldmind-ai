"""
config.py

GoldMIND AI System Configuration

This file defines the default configuration settings for the GoldMIND AI system.
These settings can be overridden by a local 'config.json' file and by environment
variables in a '.env' file using '__' as a nested-key separator.
"""

import os
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def _deep_merge_dicts(source: dict, destination: dict) -> dict:
    """
    Recursively merges source dict into destination dict.
    Values from source overwrite values in destination.
    """
    for key, value in source.items():
        if (
            key in destination
            and isinstance(destination[key], dict)
            and isinstance(value, dict)
        ):
            destination[key] = _deep_merge_dicts(value, destination[key])
        else:
            destination[key] = value
    return destination


class ProductionConfigManager:
    def __init__(self):
        # 1️⃣ Load built-in defaults (with os.getenv overrides baked in)
        self._config: dict = {
            "system": {
                "environment": os.getenv("SYSTEM_ENVIRONMENT", "development"),
                "live_mode": os.getenv("SYSTEM_LIVE_MODE", "False").lower() == "true",
                "auto_update": os.getenv("SYSTEM_AUTO_UPDATE", "True").lower() == "true",
                "log_level": os.getenv("SYSTEM_LOG_LEVEL", "INFO").upper(),
            },
            "server": {
                "host": os.getenv("SERVER_HOST", "0.0.0.0"),
                "port": int(os.getenv("SERVER_PORT", 5000)),
                "debug": os.getenv("SERVER_DEBUG", "False").lower() == "true",
                "workers": int(os.getenv("SERVER_WORKERS", 4)),
                "timeout": int(os.getenv("SERVER_TIMEOUT", 30)),
            },
            "database": {
                "path": os.getenv("DATABASE_PATH", "./data/goldmind_ai.db"),
                "backup_interval": int(os.getenv("DATABASE_BACKUP_INTERVAL", 3600)),
                "create_samples": os.getenv("DATABASE_CREATE_SAMPLES", "False").lower() == "true",
            },
            "market_data": {
                "update_interval": int(os.getenv("MARKET_DATA_UPDATE_INTERVAL", 30)),
                "demo_mode": os.getenv("MARKET_DATA_DEMO_MODE", "True").lower() == "true",
                "provider": os.getenv("MARKET_DATA_PROVIDER", "auto"),
                "max_history_size": int(os.getenv("MARKET_DATA_MAX_HISTORY_SIZE", 5000)),
                "retry_attempts": int(os.getenv("MARKET_DATA_RETRY_ATTEMPTS", 5)),
                "retry_delay": int(os.getenv("MARKET_DATA_RETRY_DELAY", 10)),
            },
            "ml_models": {
                "retrain_interval": int(os.getenv("ML_MODELS_RETRAIN_INTERVAL", 86400)),
                "confidence_threshold": float(os.getenv("ML_MODELS_CONFIDENCE_THRESHOLD", 0.6)),
                "auto_update": os.getenv("ML_MODELS_AUTO_UPDATE", "True").lower() == "true",
                "gpu_enabled": os.getenv("ML_MODELS_GPU_ENABLED", "False").lower() == "true",
                "sequence_length": int(os.getenv("ML_MODELS_SEQUENCE_LENGTH", 60)),
                "lstm_units": int(os.getenv("ML_MODELS_LSTM_UNITS", 64)),
                "dropout_rate": float(os.getenv("ML_MODELS_DROPOUT_RATE", 0.2)),
                "learning_rate": float(os.getenv("ML_MODELS_LEARNING_RATE", 0.001)),
                "batch_size": int(os.getenv("ML_MODELS_BATCH_SIZE", 32)),
            },
            "analytics": {
                "enable_realtime": os.getenv("ANALYTICS_ENABLE_REALTIME", "True").lower() == "true",
                "retention_days": int(os.getenv("ANALYTICS_RETENTION_DAYS", 90)),
                "update_interval": int(os.getenv("ANALYTICS_UPDATE_INTERVAL", 3600)),
                "cache_timeout": int(os.getenv("ANALYTICS_CACHE_TIMEOUT", 300)),
            },
            "security": {
                "enable_audit_log": os.getenv("SECURITY_ENABLE_AUDIT_LOG", "True").lower() == "true",
                "session_timeout": int(os.getenv("SECURITY_SESSION_TIMEOUT", 7200)),
                "max_login_attempts": int(os.getenv("SECURITY_MAX_LOGIN_ATTEMPTS", 5)),
                "block_duration_minutes": int(os.getenv("SECURITY_BLOCK_DURATION_MINUTES", 15)),
            },
            "auto_hedging": {
                "risk_trigger": float(os.getenv("AUTO_HEDGING_RISK_TRIGGER", 7.0)),
                "max_exposure": float(os.getenv("AUTO_HEDGING_MAX_EXPOSURE", 0.15)),
                "cost_limit": int(os.getenv("AUTO_HEDGING_COST_LIMIT", 50)),
                "auto_execute": os.getenv("AUTO_HEDGING_AUTO_EXECUTE", "False").lower() == "true",
                "monitoring_interval": int(os.getenv("AUTO_HEDGING_MONITORING_INTERVAL", 300)),
                "circuit_breaker": float(os.getenv("AUTO_HEDGING_CIRCUIT_BREAKER", 9.5)),
                "max_hedge_duration": int(os.getenv("AUTO_HEDGING_MAX_HEDGE_DURATION", 45)),
                "min_hedge_size": float(os.getenv("AUTO_HEDGING_MIN_HEDGE_SIZE", 0.02)),
                "max_hedge_size": float(os.getenv("AUTO_HEDGING_MAX_HEDGE_SIZE", 0.20)),
                "hedge_effectiveness_threshold": float(os.getenv("AUTO_HEDGING_EFFECTIVENESS_THRESHOLD", 0.6)),
            },
            "model_performance": {
                "accuracy_threshold": float(os.getenv("MODEL_PERFORMANCE_ACCURACY_THRESHOLD", 0.75)),
                "response_time_threshold": int(os.getenv("MODEL_PERFORMANCE_RESPONSE_TIME_THRESHOLD", 1000)),
                "error_rate_threshold": float(os.getenv("MODEL_PERFORMANCE_ERROR_RATE_THRESHOLD", 0.05)),
                "drift_threshold": float(os.getenv("MODEL_PERFORMANCE_DRIFT_THRESHOLD", 0.3)),
                "resource_threshold": int(os.getenv("MODEL_PERFORMANCE_RESOURCE_THRESHOLD", 80)),
                "throughput_threshold": int(os.getenv("MODEL_PERFORMANCE_THROUGHPUT_THRESHOLD", 5)),
                "auto_fallback": os.getenv("MODEL_PERFORMANCE_AUTO_FALLBACK", "True").lower() == "true",
                "fallback_threshold": float(os.getenv("MODEL_PERFORMANCE_FALLBACK_THRESHOLD", 0.60)),
                "fallback_delay_minutes": int(os.getenv("MODEL_PERFORMANCE_FALLBACK_DELAY_MINUTES", 5)),
                "recovery_threshold": float(os.getenv("MODEL_PERFORMANCE_RECOVERY_THRESHOLD", 0.80)),
                "max_fallback_duration": int(os.getenv("MODEL_PERFORMANCE_MAX_FALLBACK_DURATION", 1440)),
                "strategy": os.getenv("MODEL_PERFORMANCE_STRATEGY", "gradual"),
                "canary_percentage": int(os.getenv("MODEL_PERFORMANCE_CANARY_PERCENTAGE", 10)),
                "monitoring_interval": int(os.getenv("MODEL_PERFORMANCE_MONITORING_INTERVAL", 300)),
                "prediction_buffer_size": int(os.getenv("MODEL_PERFORMANCE_PREDICTION_BUFFER_SIZE", 200)),
                "ground_truth_buffer_size": int(os.getenv("MODEL_PERFORMANCE_GROUND_TRUTH_BUFFER_SIZE", 100)),
                "response_time_buffer_size": int(os.getenv("MODEL_PERFORMANCE_RESPONSE_TIME_BUFFER_SIZE", 50)),
                "feature_buffer_size": int(os.getenv("MODEL_PERFORMANCE_FEATURE_BUFFER_SIZE", 1000)),
                "history_size": int(os.getenv("MODEL_PERFORMANCE_HISTORY_SIZE", 1000)),
            },
            "dual_system": {
                "analytical_weight": float(os.getenv("DUAL_SYSTEM_ANALYTICAL_WEIGHT", 0.4)),
                "ml_weight": float(os.getenv("DUAL_SYSTEM_ML_WEIGHT", 0.3)),
                "lstm_weight": float(os.getenv("DUAL_SYSTEM_LSTM_WEIGHT", 0.3)),
                "hybrid_weight": float(os.getenv("DUAL_SYSTEM_HYBRID_WEIGHT", 0.5)),
                "bias_threshold": float(os.getenv("DUAL_SYSTEM_BIAS_THRESHOLD", 0.7)),
                "consensus_threshold": float(os.getenv("DUAL_SYSTEM_CONSENSUS_THRESHOLD", 0.8)),
                "uncertainty_penalty": float(os.getenv("DUAL_SYSTEM_UNCERTAINTY_PENALTY", 0.15)),
                "bias_penalty_factor": float(os.getenv("DUAL_SYSTEM_BIAS_PENALTY_FACTOR", 0.2)),
                "bias_detection_enabled": os.getenv("DUAL_SYSTEM_BIAS_DETECTION_ENABLED", "True").lower() == "true",
                "bias_report_threshold": int(os.getenv("DUAL_SYSTEM_BIAS_REPORT_THRESHOLD", 60)),
            },
            "redis": {
                "host": os.getenv("REDIS_HOST", "localhost"),
                "port": int(os.getenv("REDIS_PORT", 6379)),
                "db": int(os.getenv("REDIS_DB", 0)),
                "password": os.getenv("REDIS_PASSWORD", None),
            },
            "goldmind_api": {
                "api_key": os.getenv("GOLDMIND_API_KEY", ""),
            }
        }
        logger.info("Loaded built-in default config (with env var overrides).")

    def load_config_from_file(self, config_file_path: str = "config.json"):
        """Loads configuration from a JSON file and merges it over built-in defaults."""
        file = Path(config_file_path)
        if not file.is_file():
            logger.warning(f"Config file not found at {config_file_path}; skipping.")
            return

        try:
            data = json.loads(file.read_text())
            _deep_merge_dicts(data, self._config)
            logger.info(f"Successfully merged config from {config_file_path}.")
        except Exception as e:
            logger.error(f"Failed to load/merge {config_file_path}: {e}", exc_info=True)

    def load_env_file(self, env_path: str = ".env"):
        """
        Loads and overrides any variables in a .env file.
        Keys must use '__' to indicate nesting, e.g. 'SERVER__PORT=8080'.
        """
        load_dotenv(env_path, override=True)
        for key, val in os.environ.items():
            if "__" not in key:
                continue
            parts = [p.lower() for p in key.split("__")]
            self._set_by_path(parts, val)
        logger.info(f"Loaded overrides from {env_path}.")

    def _set_by_path(self, parts: list[str], value):
        """Helper to set a nested key given a list of parts."""
        current = self._config
        for p in parts[:-1]:
            current = current.setdefault(p, {})
        current[parts[-1]] = value

    def get(self, key_path: str, default=None):
        """Retrieve nested config via dot-notation, e.g. get("server.port")."""
        parts = key_path.split(".")
        current = self._config
        for p in parts:
            if not isinstance(current, dict) or p not in current:
                return default
            current = current[p]
        return current

    @property
    def config_data(self) -> dict:
        """Expose the entire merged config as a plain dict."""
        return self._config


# If you ever run this file directly, you’ll see your merged config printed.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = ProductionConfigManager()
    cfg.load_config_from_file("config.json")
    cfg.load_env_file(".env")
    print(json.dumps(cfg.config_data, indent=2))
