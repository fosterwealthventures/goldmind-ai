"""
Enhanced Model Performance Monitor & Auto-Fallback System for GoldMIND AI
Continuous monitoring with advanced drift detection, multi-stage fallback, and real-time diagnostics.
Integrates with FinancialDataFramework for persistence and NotificationManager for alerts.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import json
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
import statistics
import warnings
from collections import deque, defaultdict
import joblib
import sqlite3
import hashlib
import psutil
from scipy.stats import ks_2samp # Kolmogorov-Smirnov test for drift detection

# Suppress TensorFlow warnings if it's imported elsewhere and causing issues
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
logger = logging.getLogger(__name__)

# Import FinancialDataFramework for database interaction
try:
    from financial_data_framework import FinancialDataFramework
except ImportError:
    logging.critical("❌ Could not import FinancialDataFramework. Please ensure 'financial_data_framework.py' is accessible.")
    class FinancialDataFramework: # Mock for parsing
        def __init__(self, *args, **kwargs): pass
        def get_connection(self): return None
        def init_database(self): pass # Add mock for init_database
        def get_usage_report(self): return {'apis': {}} # Add mock for get_usage_report

# Import NotificationManager (placeholder if not yet created)
try:
    from notification_system import NotificationManager
except ImportError:
    logging.critical("❌ Could not import NotificationManager. Please ensure 'notification_system.py' is accessible. Using a mock.")
    class NotificationManager: # Mock for parsing
        def __init__(self, config=None):
            logging.warning("Mock NotificationManager initialized.")
        def send_alert(self, alert_type: str, message: str, severity: str = "INFO"):
            logging.info(f"Mock Notification: [{severity}] {alert_type}: {message}")
        def send_report(self, report_type: str, data: Dict):
            logging.info(f"Mock Notification Report: {report_type} - {json.dumps(data)}")


# Enums for clarity and consistency
class ModelStatus(Enum):
    """Current operational status of a model."""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    FALLBACK = "fallback"
    TRAINING = "training"
    ERROR = "error"

class MetricType(Enum):
    """Types of metrics monitored."""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    DATA_DRIFT = "data_drift"
    MODEL_DRIFT = "model_drift"
    RESOURCE_UTILIZATION = "resource_utilization"
    ERROR_RATE = "error_rate"
    CONFIDENCE_SCORE = "confidence_score"
    STATUS = "status" # Added for general status messages

class AlertSeverity(Enum):
    """Severity levels for alerts."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical" # Changed from EMERGENCY to CRITICAL

@dataclass
class Alert:
    """Represents an active alert in the system."""
    alert_id: str
    model_name: str
    metric: MetricType
    severity: AlertSeverity
    message: str
    timestamp: datetime = datetime.utcnow()
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None

@dataclass
class ModelMetric:
    """Stores a single metric observation for a model."""
    timestamp: datetime
    value: float
    metric_type: MetricType

@dataclass
class ModelPerformanceSnapshot:
    """Aggregated performance metrics for a model at a given time."""
    timestamp: datetime
    model_name: str
    accuracy: Optional[float] = None
    latency_ms: Optional[float] = None
    throughput_per_sec: Optional[float] = None
    data_drift_score: Optional[float] = None
    model_drift_score: Optional[float] = None
    error_rate: Optional[float] = None
    avg_confidence: Optional[float] = None
    cpu_utilization_percent: Optional[float] = None
    memory_utilization_percent: Optional[float] = None


class ModelPerformanceMonitor:
    """
    Monitors the performance and health of deployed ML models.
    Provides real-time diagnostics, drift detection, and automated fallback mechanisms.
    """
    def __init__(self, config: Dict, db_manager: FinancialDataFramework, notification_manager: NotificationManager):
        self.config = config.get('model_performance', {})
        self.db_manager = db_manager
        self.notification_manager = notification_manager
        
        self.monitoring_interval = self.config.get('monitoring_interval', 60) # seconds
        self.alert_thresholds = self.config.get('alert_thresholds', {})
        self.fallback_strategy = self.config.get('fallback_strategy', 'simple_average')
        self.drift_detection_enabled = self.config.get('drift_detection_enabled', True)
        self.drift_window_size = self.config.get('drift_window_size', 1000) # Number of samples for drift detection
        self.ks_alpha = self.config.get('ks_alpha', 0.05) # Significance level for KS test

        self.models: Dict[str, Dict[str, Any]] = {} # Registered models and their metadata
        self.model_status: Dict[str, ModelStatus] = defaultdict(lambda: ModelStatus.OFFLINE) # Current status of each model
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.get('history_size', 1000))) # Raw metric history
        self.active_alerts: Dict[str, Alert] = {} # Active alerts by alert_id
        self.baseline_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.drift_window_size * 2)) # Data for drift baselines

        self._monitoring_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock() # For thread-safe access to shared data

        self.last_report_time = datetime.utcnow()
        self.report_interval = self.config.get('report_interval', 3600) # Hourly reports by default

        logger.info("ModelPerformanceMonitor initialized.")

    def register_model(self, model_name: str, model_instance: Any, baseline_model_name: Optional[str] = None):
        """
        Registers a model for monitoring.
        :param model_name: Unique name for the model.
        :param model_instance: The actual model object (e.g., Keras model, sklearn model).
        :param baseline_model_name: Optional name of a simpler, more robust model to fall back to.
        """
        with self._lock:
            self.models[model_name] = {
                "instance": model_instance,
                "baseline_model_name": baseline_model_name,
                "last_prediction_time": None,
                "last_training_time": None,
                "prediction_count": 0,
                "error_count": 0,
                "total_latency": 0.0,
                "total_confidence": 0.0,
                "input_data_samples": deque(maxlen=self.drift_window_size) # Store input data for drift
            }
            self.model_status[model_name] = ModelStatus.OPERATIONAL
            logger.info(f"Model '{model_name}' registered for monitoring.")
            # Use MetricType.STATUS for general operational messages
            self.create_alert(model_name, MetricType.STATUS, AlertSeverity.INFO, f"Model '{model_name}' registered and operational.")

    def record_prediction(self, model_name: str, prediction: Any, confidence: float, latency_ms: float, features: Dict):
        """
        Records a prediction event and its associated metrics.
        :param model_name: Name of the model.
        :param prediction: The output of the model.
        :param confidence: The confidence score of the prediction (0-1).
        :param latency_ms: Latency of the prediction in milliseconds.
        :param features: The input features used for the prediction.
        """
        with self._lock:
            if model_name not in self.models:
                logger.warning(f"Attempted to record prediction for unregistered model: {model_name}")
                return

            model_data = self.models[model_name]
            model_data["last_prediction_time"] = datetime.utcnow()
            model_data["prediction_count"] += 1
            model_data["total_latency"] += latency_ms
            model_data["total_confidence"] += confidence
            self.performance_history[model_name].append(
                ModelMetric(datetime.utcnow(), confidence, MetricType.CONFIDENCE_SCORE)
            )
            self.performance_history[model_name].append(
                ModelMetric(datetime.utcnow(), latency_ms, MetricType.LATENCY)
            )
            # Store input features for data drift detection
            if self.drift_detection_enabled:
                model_data["input_data_samples"].append(features)
                self.baseline_data[model_name].append(features) # Add to baseline for drift

            # Check for immediate alerts (e.g., very low confidence)
            if confidence < self.alert_thresholds.get('min_confidence', 0.4):
                self.create_alert(model_name, MetricType.CONFIDENCE_SCORE, AlertSeverity.WARNING,
                                  f"Low confidence prediction: {confidence:.2f}")

    def record_error(self, model_name: str, error_message: str, severity: AlertSeverity = AlertSeverity.ERROR):
        """
        Records an error event for a model.
        :param model_name: Name of the model.
        :param error_message: Description of the error.
        :param severity: Severity of the error.
        """
        with self._lock:
            if model_name not in self.models:
                logger.warning(f"Attempted to record error for unregistered model: {model_name}")
                return
            self.models[model_name]["error_count"] += 1
            self.create_alert(model_name, MetricType.ERROR_RATE, severity, f"Model error: {error_message}")
            self.model_status[model_name] = ModelStatus.ERROR

    def start_monitoring(self):
        """Starts the background monitoring thread."""
        if self._running:
            logger.warning("Monitoring is already running.")
            return
        self._running = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Model performance monitoring started.")

    def stop_monitoring(self):
        """Stops the background monitoring thread."""
        if not self._running:
            logger.warning("Monitoring is not running.")
            return
        self._running = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=self.monitoring_interval + 5) # Give it time to finish current cycle
        logger.info("Model performance monitoring stopped.")

    def _monitoring_loop(self):
        """Main monitoring loop running in a separate thread."""
        while self._running:
            logger.debug("Running model monitoring cycle...")
            try:
                self._check_all_models_health()
                self._generate_periodic_report()
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}", exc_info=True)
                # Log to notification manager as well
                self.notification_manager.send_alert(
                    "monitoring_system_error",
                    f"Model monitoring loop encountered an error: {e}",
                    AlertSeverity.CRITICAL.value # Use .value
                )
            time.sleep(self.monitoring_interval)

    def _check_all_models_health(self):
        """Iterates through all registered models and checks their health."""
        with self._lock:
            for model_name in list(self.models.keys()): # Iterate over a copy to allow modification
                self.check_model_health(model_name)
            self._check_system_resources()

    def _check_system_resources(self):
        """Checks overall system resource utilization."""
        cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking
        memory_percent = psutil.virtual_memory().percent

        if cpu_percent > self.alert_thresholds.get('max_cpu_utilization', 90):
            self.create_alert("system", MetricType.RESOURCE_UTILIZATION, AlertSeverity.CRITICAL,
                              f"High CPU utilization: {cpu_percent:.2f}%")
        if memory_percent > self.alert_thresholds.get('max_memory_utilization', 85):
            self.create_alert("system", MetricType.RESOURCE_UTILIZATION, AlertSeverity.CRITICAL,
                              f"High Memory utilization: {memory_percent:.2f}%")
        
        # Log resource usage for overall diagnostics
        self.performance_history["system_resources"].append(
            ModelPerformanceSnapshot(
                timestamp=datetime.utcnow(),
                model_name="system_resources",
                cpu_utilization_percent=cpu_percent,
                memory_utilization_percent=memory_percent
            )
        )

    def check_model_health(self, model_name: str):
        """
        Performs a comprehensive health check for a single model.
        This includes:
        - Liveness check (last prediction time)
        - Performance metric analysis (latency, confidence, error rate)
        - Data drift detection
        - Model drift detection (if applicable)
        - Fallback trigger if issues detected
        """
        with self._lock: # Ensure thread safety for model data access
            if model_name not in self.models:
                logger.warning(f"Health check requested for unregistered model: {model_name}")
                return

            model_data = self.models[model_name]
            current_status = ModelStatus.OPERATIONAL

            # 1. Liveness Check
            if model_data["last_prediction_time"] is None or \
               (datetime.utcnow() - model_data["last_prediction_time"]).total_seconds() > self.config.get('liveness_timeout', 300):
                self.create_alert(model_name, MetricType.STATUS, AlertSeverity.WARNING, "Model not making recent predictions.")
                current_status = ModelStatus.DEGRADED

            # 2. Performance Metric Analysis
            recent_confidences = [m.value for m in self.performance_history[model_name] if m.metric_type == MetricType.CONFIDENCE_SCORE]
            recent_latencies = [m.value for m in self.performance_history[model_name] if m.metric_type == MetricType.LATENCY]

            avg_confidence = statistics.mean(recent_confidences) if recent_confidences else 0.0
            avg_latency = statistics.mean(recent_latencies) if recent_latencies else 0.0
            error_rate = model_data["error_count"] / max(1, model_data["prediction_count"])

            if avg_confidence < self.alert_thresholds.get('min_avg_confidence', 0.6):
                self.create_alert(model_name, MetricType.CONFIDENCE_SCORE, AlertSeverity.ERROR,
                                  f"Average confidence dropped to {avg_confidence:.2f}")
                current_status = ModelStatus.DEGRADED
            if avg_latency > self.alert_thresholds.get('max_latency_ms', 500):
                self.create_alert(model_name, MetricType.LATENCY, AlertSeverity.WARNING,
                                  f"Average latency increased to {avg_latency:.2f}ms")
                current_status = ModelStatus.DEGRADED
            if error_rate > self.alert_thresholds.get('max_error_rate', 0.05):
                self.create_alert(model_name, MetricType.ERROR_RATE, AlertSeverity.CRITICAL,
                                  f"High error rate: {error_rate:.2%}")
                current_status = ModelStatus.ERROR

            # 3. Data Drift Detection (using Kolmogorov-Smirnov test)
            if self.drift_detection_enabled and len(model_data["input_data_samples"]) >= self.drift_window_size and \
               len(self.baseline_data[model_name]) >= self.drift_window_size:
                
                # Convert deque of dicts to pandas DataFrame for easier column-wise comparison
                current_data_df = pd.DataFrame(list(model_data["input_data_samples"]))
                baseline_data_df = pd.DataFrame(list(self.baseline_data[model_name]))

                drift_detected = False
                drift_details = {}
                
                # Check each common numerical column for drift
                common_numerical_cols = list(set(current_data_df.select_dtypes(include=np.number).columns) & 
                                             set(baseline_data_df.select_dtypes(include=np.number).columns))

                for col in common_numerical_cols:
                    try:
                        # Ensure enough data points for KS test
                        if len(current_data_df[col].dropna()) > 1 and len(baseline_data_df[col].dropna()) > 1:
                            statistic, p_value = ks_2samp(current_data_df[col].dropna(), baseline_data_df[col].dropna())
                            if p_value < self.ks_alpha:
                                drift_detected = True
                                drift_details[col] = f"Data drift detected (p={p_value:.4f})"
                    except Exception as e:
                        logger.warning(f"Error during KS test for column {col}: {e}")

                if drift_detected:
                    self.create_alert(model_name, MetricType.DATA_DRIFT, AlertSeverity.WARNING,
                                      f"Data drift detected in input features: {drift_details}")
                    current_status = ModelStatus.DEGRADED

            # 4. Model Drift Detection (simplified: compare current model performance to a baseline model's expected performance)
            # In a real scenario, this would involve comparing current model's predictions/performance
            # on a fixed dataset against its performance at an earlier point, or against a simpler model.
            # For now, we'll use a placeholder logic.
            if self.drift_detection_enabled and model_data["baseline_model_name"]:
                baseline_model_name = model_data["baseline_model_name"]
                # Simulate model drift if accuracy is consistently low compared to a hypothetical baseline
                if avg_confidence < self.alert_thresholds.get('model_drift_confidence_threshold', 0.5):
                     self.create_alert(model_name, MetricType.MODEL_DRIFT, AlertSeverity.WARNING,
                                      f"Potential model drift: Average confidence ({avg_confidence:.2f}) is below threshold.")
                     current_status = ModelStatus.DEGRADED

            # 5. Fallback Trigger
            if current_status in [ModelStatus.DEGRADED, ModelStatus.ERROR]:
                self.trigger_fallback(model_name, current_status)
            elif self.model_status[model_name] != ModelStatus.OPERATIONAL and current_status == ModelStatus.OPERATIONAL:
                # If model recovers, resolve alerts and update status
                self.resolve_alerts(model_name)
                self.create_alert(model_name, MetricType.STATUS, AlertSeverity.INFO, f"Model '{model_name}' has recovered and is operational.")
                self.model_status[model_name] = ModelStatus.OPERATIONAL

            self.model_status[model_name] = current_status
            logger.debug(f"Health check for {model_name} completed. Status: {current_status.value}")


    def trigger_fallback(self, model_name: str, current_status: ModelStatus):
        """
        Triggers the fallback mechanism for a degraded or erroneous model.
        """
        with self._lock:
            if self.model_status[model_name] == ModelStatus.FALLBACK:
                logger.info(f"Model '{model_name}' is already in fallback mode.")
                return

            baseline_model_name = self.models[model_name].get("baseline_model_name")
            if baseline_model_name and self.model_status[baseline_model_name] == ModelStatus.OPERATIONAL:
                # In a real system, you would switch routing to the baseline model
                self.model_status[model_name] = ModelStatus.FALLBACK
                self.create_alert(model_name, MetricType.STATUS, AlertSeverity.WARNING,
                                  f"Model '{model_name}' is {current_status.value}. Falling back to '{baseline_model_name}'.")
                logger.warning(f"Fallback triggered for '{model_name}'. Using '{baseline_model_name}'.")
            else:
                self.model_status[model_name] = ModelStatus.OFFLINE
                self.create_alert(model_name, MetricType.STATUS, AlertSeverity.CRITICAL,
                                  f"Model '{model_name}' is {current_status.value}. No suitable fallback model available. Model is offline.")
                logger.critical(f"Model '{model_name}' is offline. No fallback available.")

    def create_alert(self, model_name: str, metric: MetricType, severity: AlertSeverity, message: str):
        """
        Creates and logs an alert, and sends it via the notification manager.
        """
        alert_id = hashlib.sha256(f"{model_name}-{metric.value}-{message}-{datetime.utcnow().isoformat()}".encode()).hexdigest()
        alert = Alert(alert_id, model_name, metric, severity, message)
        self.active_alerts[alert_id] = alert
        
        logger.log(getattr(logging, severity.value.upper()), f"ALERT: {model_name} - {metric.value} - {message}") # Use .value for severity
        self.notification_manager.send_alert(f"model_{metric.value}_alert", message, severity.value) # Use .value for severity

    def resolve_alerts(self, model_name: str):
        """Resolves all active alerts for a given model."""
        with self._lock:
            resolved_count = 0
            alerts_to_remove = []
            for alert_id, alert in self.active_alerts.items():
                if alert.model_name == model_name and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.utcnow()
                    alert.resolution_notes = "Model recovered."
                    alerts_to_remove.append(alert_id)
                    resolved_count += 1
                    logger.info(f"Resolved alert {alert.alert_id} for model {model_name}.")
                    self.notification_manager.send_alert(f"model_recovery", f"Alert for {model_name} resolved: {alert.message}", AlertSeverity.INFO.value)
            for alert_id in alerts_to_remove:
                del self.active_alerts[alert_id]
            if resolved_count > 0:
                logger.info(f"Total {resolved_count} alerts resolved for model {model_name}.")

    def get_detailed_diagnostics(self, model_name: str) -> Dict:
        """
        Returns a detailed diagnostic report for a specific model.
        """
        with self._lock:
            if model_name not in self.models:
                return {"error": f"Model '{model_name}' not registered."}

            model_data = self.models[model_name]
            
            # Calculate current metrics
            recent_confidences = [m.value for m in self.performance_history[model_name] if m.metric_type == MetricType.CONFIDENCE_SCORE]
            recent_latencies = [m.value for m in self.performance_history[model_name] if m.metric_type == MetricType.LATENCY]

            avg_confidence = statistics.mean(recent_confidences) if recent_confidences else 0.0
            avg_latency = statistics.mean(recent_latencies) if recent_latencies else 0.0
            error_rate = model_data["error_count"] / max(1, model_data["prediction_count"])

            # Get active alerts for this model
            model_alerts = [asdict(alert) for alert_id, alert in self.active_alerts.items() if alert.model_name == model_name]

            diagnostics = {
                "model_name": model_name,
                "status": self.model_status[model_name].value,
                "last_prediction_time": model_data["last_prediction_time"].isoformat() if model_data["last_prediction_time"] else "N/A",
                "prediction_count": model_data["prediction_count"],
                "error_count": model_data["error_count"],
                "current_metrics": {
                    "average_confidence": f"{avg_confidence:.2f}",
                    "average_latency_ms": f"{avg_latency:.2f}",
                    "error_rate": f"{error_rate:.2%}"
                },
                "active_alerts": model_alerts,
                "config": self.config,
                "drift_detection_status": "Enabled" if self.drift_detection_enabled else "Disabled",
                "baseline_model": model_data["baseline_model_name"] if model_data["baseline_model_name"] else "N/A"
            }
            return diagnostics

    def _generate_periodic_report(self):
        """Generates and sends a periodic performance report."""
        if (datetime.utcnow() - self.last_report_time).total_seconds() < self.report_interval:
            return

        logger.info("Generating periodic performance report...")
        report_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": {name: status.value for name, status in self.model_status.items()},
            "active_alerts_count": len(self.active_alerts),
            "active_alerts_summary": [asdict(alert) for alert in self.active_alerts.values()],
            "model_summaries": {}
        }

        with self._lock:
            for model_name, model_data in self.models.items():
                recent_confidences = [m.value for m in self.performance_history[model_name] if m.metric_type == MetricType.CONFIDENCE_SCORE]
                recent_latencies = [m.value for m in self.performance_history[model_name] if m.metric_type == MetricType.LATENCY]

                avg_confidence = statistics.mean(recent_confidences) if recent_confidences else 0.0
                avg_latency = statistics.mean(recent_latencies) if recent_latencies else 0.0
                error_rate = model_data["error_count"] / max(1, model_data["prediction_count"])

                report_data["model_summaries"][model_name] = {
                    "status": self.model_status[model_name].value,
                    "prediction_count": model_data["prediction_count"],
                    "error_count": model_data["error_count"],
                    "avg_confidence": f"{avg_confidence:.2f}",
                    "avg_latency_ms": f"{avg_latency:.2f}",
                    "error_rate": f"{error_rate:.2%}"
                }
        
        self.notification_manager.send_report("model_performance_summary", report_data)
        self.last_report_time = datetime.utcnow()
        logger.info("Periodic performance report sent.")

# --- Standalone Testing Example ---
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')

    # Mock FinancialDataFramework
    class MockFinancialDataFramework:
        def get_connection(self):
            # In-memory SQLite for testing
            conn = sqlite3.connect(':memory:')
            conn.row_factory = sqlite3.Row # Enable dict-like access to rows
            # Create a dummy table for init_database if needed by monitor
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS model_performance_metrics (id INTEGER PRIMARY KEY)")
            conn.commit()
            return conn
        def init_database(self):
            logger.info("MockFinancialDataFramework: init_database called.")
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_performance_metrics (
                        timestamp TEXT,
                        model_name TEXT,
                        metric_type TEXT,
                        value REAL
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        alert_id TEXT PRIMARY KEY,
                        model_name TEXT,
                        metric TEXT,
                        severity TEXT,
                        message TEXT,
                        timestamp TEXT,
                        resolved INTEGER,
                        resolved_at TEXT,
                        resolution_notes TEXT
                    )
                ''')
                conn.commit()
        def get_usage_report(self):
            return {'apis': {'alpha_vantage': {'calls': 100, 'errors': 0}}}

    # Mock NotificationManager
    class MockNotificationManager:
        def __init__(self, config=None):
            logger.info("Mock NotificationManager initialized for testing.")
        def send_alert(self, alert_type: str, message: str, severity: str = "INFO"):
            logger.info(f"TEST ALERT: [{severity}] {alert_type}: {message}")
        def send_report(self, report_type: str, data: Dict):
            logger.info(f"TEST REPORT: {report_type} - {json.dumps(data, indent=2)}")

    mock_config = {
        'model_performance': {
            'monitoring_interval': 2, # Check every 2 seconds for testing
            'history_size': 50,
            'alert_thresholds': {
                'min_confidence': 0.5,
                'max_latency_ms': 200,
                'max_error_rate': 0.1,
                'max_cpu_utilization': 80,
                'max_memory_utilization': 75,
                'model_drift_confidence_threshold': 0.5
            },
            'drift_detection_enabled': True,
            'drift_window_size': 20, # Smaller window for testing
            'ks_alpha': 0.05,
            'report_interval': 10 # Generate report every 10 seconds for testing
        }
    }

    mock_db_manager = MockFinancialDataFramework()
    mock_notification_manager = MockNotificationManager()

    monitor = ModelPerformanceMonitor(mock_config, mock_db_manager, mock_notification_manager)
    mock_db_manager.init_database() # Ensure DB is initialized for the monitor

    # Mock a model instance
    class MockModelInstance:
        def predict(self, data):
            # Simulate prediction with varying confidence and latency
            confidence = np.random.uniform(0.3, 0.9)
            latency = np.random.uniform(50, 300)
            # Simulate occasional errors
            if np.random.rand() < 0.02: # 2% error rate
                raise ValueError("Simulated model error")
            return confidence, latency
        def get_name(self):
            return "MockModel"

    mock_model = MockModelInstance()
    monitor.register_model("gold_price_predictor", mock_model, "simple_moving_average_model")
    monitor.register_model("simple_moving_average_model", MockModelInstance()) # Register baseline model

    monitor.start_monitoring()

    # Simulate predictions and errors
    print("\nSimulating model predictions and errors...")
    for i in range(30): # Run for 30 cycles
        try:
            confidence, latency = mock_model.predict(f"data_point_{i}")
            monitor.record_prediction("gold_price_predictor", f"pred_{i}", confidence, latency, {"feature1": i, "feature2": i*2})
            if i % 5 == 0: # Simulate a low confidence period
                monitor.record_prediction("gold_price_predictor", f"pred_low_conf_{i}", 0.35, 180, {"feature1": i, "feature2": i*2})
        except ValueError as e:
            monitor.record_error("gold_price_predictor", str(e))
        
        # Simulate some activity for the baseline model too
        monitor.record_prediction("simple_moving_average_model", f"baseline_pred_{i}", np.random.uniform(0.6, 0.8), np.random.uniform(20, 80), {"feature1": i, "feature2": i*2})

        cpu_usage = psutil.cpu_percent(interval=None)
        mem_usage = psutil.virtual_memory().percent
        print(f"Cycle {i+1}: CPU: {cpu_usage:.1f}%, Mem: {mem_usage:.1f}%")
        time.sleep(1) # Simulate time passing

    # Give some time for the monitoring loop to run and calculate metrics
    print("Waiting for monitoring system to process metrics and potentially trigger alerts/fallback (10 seconds)...")
    await asyncio.sleep(10) 
    
    # Get diagnostics
    diagnostics = monitor.get_detailed_diagnostics("gold_price_predictor")
    print("\n--- Gold Price Predictor Model Diagnostics ---")
    print(json.dumps(diagnostics, indent=2, default=str)) # Use default=str for datetime objects

    print("\n--- Overall Model Status ---")
    for model_name, status in monitor.model_status.items():
        print(f"Model: {model_name}, Status: {status.value}")
    
    print("\n--- Active Alerts ---")
    if monitor.active_alerts:
        for alert_id, alert in monitor.active_alerts.items():
            print(f"- ID: {alert.alert_id}, Model: {alert.model_name}, Metric: {alert.metric.value}, Severity: {alert.severity.value}, Msg: {alert.message}")
    else:
        print("No active alerts.")

    # Graceful shutdown
    monitor.stop_monitoring()
    print("\nModel Performance Monitor testing complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Model Performance Monitor example interrupted by user.")
    except Exception as e:
        logger.critical(f"Model Performance Monitor example failed: {e}", exc_info=True)
        sys.exit(1)

