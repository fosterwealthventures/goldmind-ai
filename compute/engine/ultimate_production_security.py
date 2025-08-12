# ultimate_production_security.py
import os
import logging
import jwt
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import threading
import time
import secrets # For generating strong secret keys
from typing import Optional, Dict, Any, Callable, Tuple, List # Added for type hinting
import redis # Import redis directly for type hinting
import json # Import json for serialization/deserialization

# Try to import Flask-Limiter and Flask-Talisman, handle if not installed
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    HAS_LIMITER = True
except ImportError:
    Limiter = None
    get_remote_address = None
    HAS_LIMITER = False
    logging.getLogger(__name__).warning("Flask-Limiter not installed. Rate limiting will not be applied automatically.")

try:
    from flask_talisman import Talisman
    HAS_TALISMAN = True
except ImportError:
    Talisman = None
    HAS_TALISMAN = False
    logging.getLogger(__name__).warning("Flask-Talisman not installed. Security headers will not be applied automatically.")

# Import NotificationManager
try:
    from notification_system import NotificationManager
except ImportError:
    logging.critical("❌ Could not import NotificationManager in ultimate_production_security.py. Please ensure 'notification_system.py' is accessible. Using a mock.")
    class NotificationManager: # Mock for parsing
        def __init__(self, config=None):
            logging.warning("Mock NotificationManager initialized.")
        def send_alert(self, alert_type: str, message: str, severity: str = "INFO"): # Corrected method name
            logging.info(f"Mock Notification: [{severity}] {alert_type}: {message}")
        def send_report(self, report_type: str, data: Dict):
            logging.info(f"Mock Notification Report: {report_type} - {json.dumps(data)}")


logger = logging.getLogger(__name__)

class AdvancedSecurityManager:
    """
    Manages user authentication, authorization (JWT), session management,
    rate limiting, security headers, and audit logging.
    Integrates with Redis for session/rate limiting and NotificationManager for alerts.
    """
    def __init__(self, redis_client: Optional[redis.StrictRedis] = None, encryption_key_path: str = "security/encryption.key", notification_manager: Optional[NotificationManager] = None):
        self.redis = redis_client
        self.encryption_key_path = encryption_key_path
        self.notification_manager = notification_manager or NotificationManager(config={}) # Use provided or default mock
        self.jwt_secret = self._load_or_generate_jwt_secret()
        self.limiter: Optional[Limiter] = None
        self.talisman: Optional[Talisman] = None
        self.security_events: List[Dict] = [] # In-memory audit log
        self.max_login_attempts = 5
        self.block_duration_minutes = 15

        # Ensure security directory exists
        os.makedirs(os.path.dirname(self.encryption_key_path), exist_ok=True)
        logger.info("AdvancedSecurityManager initialized.")

    def _load_or_generate_jwt_secret(self) -> str:
        """Loads JWT secret from file or generates a new one."""
        if os.path.exists(self.encryption_key_path):
            with open(self.encryption_key_path, 'r') as f:
                key = f.read().strip()
                if len(key) >= 32: # Ensure key is sufficiently long
                    logger.info("Encryption key loaded.")
                    return key
                else:
                    logger.warning("Loaded encryption key is too short. Generating a new one.")
        
        # Generate a new, strong key
        new_key = secrets.token_urlsafe(32) # Generates a URL-safe text string, 32 bytes = 256 bits
        with open(self.encryption_key_path, 'w') as f:
            f.write(new_key)
        logger.info("New encryption key generated and saved.")
        return new_key

    def set_app(self, app):
        """
        Applies Flask-Limiter and Flask-Talisman to the Flask app.
        Must be called after Flask app initialization.
        """
        if HAS_LIMITER and self.redis:
            self.limiter = Limiter(
                app=app,
                key_func=get_remote_address,
                storage_uri=f"redis://{self.redis.connection_pool.connection_kwargs['host']}:{self.redis.connection_pool.connection_kwargs['port']}/{self.redis.connection_pool.connection_kwargs['db']}"
            )
            logger.info("Flask-Limiter enabled for rate limiting.")
        elif not self.redis:
            logger.warning("Redis client not available. Flask-Limiter will not be applied.")

        if HAS_TALISMAN:
            self.talisman = Talisman(app,
                                     content_security_policy={
                                         'default-src': ["'self'"],
                                         'script-src': ["'self'", "'unsafe-inline'", "https://cdn.tailwindcss.com"], # Allow Tailwind CDN
                                         'style-src': ["'self'", "'unsafe-inline'", "https://cdn.tailwindcss.com"], # Allow Tailwind CDN
                                         'img-src': ["'self'", "data:", "https://placehold.co"], # Allow placeholder images
                                         'connect-src': ["'self'", "ws:", "wss:", "https://generativelanguage.googleapis.com"], # For Gemini API
                                         'frame-ancestors': ["'self'"], # Prevent clickjacking
                                     },
                                     force_https=False # Set to True in production
                                    )
            logger.info("Flask-Talisman enabled for security headers.")

    def generate_jwt_token(self, user_id: int, expires_in_hours: int = 24) -> str:
        """Generates a JWT token for the given user ID."""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=expires_in_hours),
            'iat': datetime.utcnow()
        }
        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        self.log_security_event(user_id, "token_generated", f"JWT token issued for user {user_id}")
        return token

    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verifies a JWT token and returns its payload if valid."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            self.log_security_event(payload.get('user_id'), "token_verified", "JWT token successfully verified")
            return payload
        except jwt.ExpiredSignatureError:
            self.log_security_event(None, "token_expired", "Expired JWT token provided")
            return None
        except jwt.InvalidTokenError as e:
            self.log_security_event(None, "invalid_token", f"Invalid JWT token provided: {e}")
            return None

    def hash_password(self, password: str) -> str:
        """Hashes a password using Werkzeug's security module."""
        return generate_password_hash(password)

    def check_password(self, hashed_password: str, password: str) -> bool:
        """Checks a password against a hashed password."""
        return check_password_hash(hashed_password, password)

    def authenticate_user(self, username: str, password: str) -> Optional[int]:
        """
        Authenticates a user.
        In a real application, this would interact with a user database.
        For this example, we'll use a mock user store.
        """
        # Mock user store (replace with actual DB interaction)
        mock_users = {
            "testuser": {"id": 1, "password_hash": self.hash_password("testpass")},
            "admin": {"id": 2, "password_hash": self.hash_password("adminpass")}
        }

        if self.is_user_blocked(username):
            self.log_security_event(None, "login_attempt_blocked", f"Blocked login attempt for user: {username}")
            self.notification_manager.send_alert("security_alert", f"Blocked login attempt for user: {username} due to too many failed attempts.", "WARNING")
            return None

        user_data = mock_users.get(username)
        if user_data and self.check_password(user_data["password_hash"], password):
            self.reset_failed_login_attempts(username)
            self.log_security_event(user_data["id"], "login_success", f"User {username} logged in successfully")
            self.notification_manager.send_alert("auth_event", f"User {username} logged in successfully.", "INFO")
            return user_data["id"]
        else:
            self.record_failed_login_attempt(username)
            self.log_security_event(None, "login_failure", f"Failed login attempt for user: {username}")
            self.notification_manager.send_alert("auth_event", f"Failed login attempt for user: {username}.", "WARNING")
            return None

    def register_user(self, username: str, password: str) -> Optional[int]:
        """
        Registers a new user.
        In a real application, this would interact with a user database.
        For this example, we'll use a mock user store.
        """
        # Mock user store (replace with actual DB interaction)
        mock_users = {
            "testuser": {"id": 1, "password_hash": self.hash_password("testpass")},
            "admin": {"id": 2, "password_hash": self.hash_password("adminpass")}
        }

        if username in mock_users:
            self.log_security_event(None, "registration_failure", f"Attempted to register existing username: {username}")
            return None # Username already exists

        new_user_id = max([u["id"] for u in mock_users.values()]) + 1 if mock_users else 1
        mock_users[username] = {"id": new_user_id, "password_hash": self.hash_password(password)}
        self.log_security_event(new_user_id, "user_registration", f"New user registered: {username}")
        self.notification_manager.send_alert("auth_event", f"New user registered: {username}.", "INFO")
        return new_user_id

    def jwt_required(self, f: Callable) -> Callable:
        """
        Decorator for protecting API routes with JWT authentication.
        Ensures Flask's request and jsonify are available at runtime.
        """
        @wraps(f) # Crucial for Flask decorators to preserve function metadata
        def decorated_function(*args, **kwargs):
            # Import request and jsonify locally to ensure they are available within the Flask app context
            from flask import request, jsonify 

            token = None
            if 'Authorization' in request.headers:
                token = request.headers['Authorization'].split(" ")[1]

            if not token:
                return jsonify({"message": "Authorization token is missing!"}), 401

            payload = self.verify_jwt_token(token)
            if not payload:
                return jsonify({"message": "Token is invalid or expired!"}), 401

            request.user_id = payload.get('user_id') # Attach user_id to request object
            return f(*args, **kwargs)
        return decorated_function

    # --- Rate Limiting and Brute Force Protection (using Redis) ---
    def record_failed_login_attempt(self, username: str):
        """Records a failed login attempt for a user."""
        if not self.redis:
            logger.warning("Redis not configured. Cannot record failed login attempts.")
            return False

        key = f"failed_login:{username}"
        self.redis.incr(key)
        self.redis.expire(key, self.block_duration_minutes * 60) # Expire after block duration

        attempts = int(self.redis.get(key) or 0)
        if attempts >= self.max_login_attempts:
            self.block_user(username)
            self.log_security_event(None, "brute_force_block", f"User {username} blocked due to too many failed login attempts.")
            self.notification_manager.send_alert("security_alert", f"User {username} blocked due to {self.max_login_attempts} failed login attempts.", "CRITICAL")
        return True

    def reset_failed_login_attempts(self, username: str):
        """Resets failed login attempts for a user upon successful login."""
        if self.redis:
            self.redis.delete(f"failed_login:{username}")
            self.redis.delete(f"blocked_user:{username}") # Also unblock if they somehow got blocked and then logged in

    def is_user_blocked(self, username: str) -> bool:
        """Checks if a user is currently blocked."""
        if not self.redis:
            return False
        return self.redis.exists(f"blocked_user:{username}")

    def block_user(self, username: str):
        """Blocks a user for a specified duration."""
        if self.redis:
            self.redis.setex(f"blocked_user:{username}", self.block_duration_minutes * 60, "true")
            logger.warning(f"User {username} has been blocked for {self.block_duration_minutes} minutes.")

    # --- Audit Logging ---
    def log_security_event(self, user_id: Optional[int], event_type: str, message: str, severity: str = "INFO"):
        """Logs a security-related event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "event_type": event_type,
            "message": message,
            "severity": severity
        }
        self.security_events.append(event)
        logger.log(getattr(logging, severity.upper()), f"SECURITY EVENT: {event_type} - {message}")
        # In a production system, this would also write to a persistent, immutable audit log.

    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """Retrieves recent security audit logs."""
        return self.security_events[-limit:]

class PerformanceOptimizer:
    """
    Optimizes system performance through caching, resource management,
    and adaptive scaling recommendations.
    """
    def __init__(self, redis_client: Optional[redis.StrictRedis] = None):
        self.redis = redis_client
        logger.info("PerformanceOptimizer initialized.")

    def cache_data(self, key: str, data: Any, ttl: int = 3600):
        """Caches data in Redis with a Time-To-Live (TTL)."""
        if not self.redis:
            logger.warning("Redis not configured. Cannot cache data.")
            return False
        try:
            # Serialize complex objects to JSON
            if isinstance(data, (dict, list)):
                data_to_store = json.dumps(data)
            else:
                data_to_store = str(data) # Convert other types to string
            self.redis.setex(key, ttl, data_to_store)
            logger.debug(f"Cached data for key: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache data for key {key}: {e}")
            return False

    def get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieves cached data from Redis."""
        if not self.redis:
            return None
        try:
            cached_data = self.redis.get(key)
            if cached_data:
                # Attempt to deserialize from JSON first
                try:
                    return json.loads(cached_data)
                except json.JSONDecodeError:
                    return cached_data # Return as string if not JSON
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve cached data for key {key}: {e}")
            return None

    def get_performance_stats(self) -> Dict:
        """Retrieves basic performance statistics (mock for now)."""
        # In a real system, this would integrate with OS metrics or a monitoring agent
        cpu_usage = psutil.cpu_percent(interval=0.1) # Non-blocking
        memory_usage = psutil.virtual_memory().percent
        return {
            "cpu_utilization_percent": cpu_usage,
            "memory_utilization_percent": memory_usage,
            "redis_connected": self.redis is not None and self.redis.ping()
        }
        
class ProductionMonitoringSystem:
    """
    Monitors the overall production system health, performance, and security.
    Integrates with SecurityManager, PerformanceOptimizer, and FinancialDataFramework.
    """
    def __init__(self, security_manager: AdvancedSecurityManager, performance_optimizer: PerformanceOptimizer, data_framework: Any, notification_manager: NotificationManager):
        self.security_manager = security_manager
        self.performance_optimizer = performance_optimizer
        self.data_framework = data_framework # FinancialDataFramework instance
        self.notification_manager = notification_manager
        self._monitoring_thread: Optional[threading.Thread] = None
        self._running = False
        self.monitoring_interval = 30 # seconds
        logger.info("ProductionMonitoringSystem initialized.")
    
    def start_monitoring(self):
        """Starts the background monitoring loop."""
        if self._running:
            logger.warning("Production monitoring is already running.")
            return
        self._running = True
        self._monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("ProductionMonitoringSystem started.")
    
    def stop_monitoring(self):
        """Stops the background monitoring loop."""
        if not self._running:
            logger.warning("Production monitoring is not running.")
            return
        self._running = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=self.monitoring_interval + 5)
        logger.info("ProductionMonitoringSystem stopped.")

    def _monitor_loop(self):
        """The main monitoring loop."""
        while self._running:
            logger.debug("Running production monitoring cycle...")
            try:
                self._check_comprehensive_health()
            except Exception as e:
                logger.error(f"Production monitoring loop error: {e}", exc_info=True)
                self.notification_manager.send_alert("production_monitor_error", f"Production monitoring loop failed: {e}", "CRITICAL")
            time.sleep(self.monitoring_interval)
    
    def _check_comprehensive_health(self):
        """Checks the health of various system components."""
        overall_health = True
        health_details = {}

        # Check resource utilization
        perf_stats = self.performance_optimizer.get_performance_stats()
        health_details['performance'] = perf_stats
        if perf_stats['cpu_utilization_percent'] > 90 or perf_stats['memory_utilization_percent'] > 85:
            self.notification_manager.send_alert("resource_alert", "High system resource utilization detected!", "WARNING")
            overall_health = False

        # Check database connectivity and performance
        db_health = self._check_database_health()
        health_details['database'] = db_health
        if not db_health['connected'] or db_health['latency_ms'] > 100:
            self.notification_manager.send_alert("db_alert", "Database performance or connectivity issues detected!", "CRITICAL")
            overall_health = False

        # Check API usage and errors (if data_framework supports it)
        if hasattr(self.data_framework, 'get_usage_report'):
            api_usage = self.data_framework.get_usage_report()
            health_details['api_usage'] = api_usage
            for api, stats in api_usage.get('apis', {}).items():
                if stats.get('errors', 0) > stats.get('calls', 1) * 0.05: # More than 5% error rate
                    self.notification_manager.send_alert("api_error", f"High error rate for API: {api}", "ERROR")
                    overall_health = False

        # Check security events (e.g., failed login attempts)
        recent_security_events = self.security_manager.get_audit_log(limit=10)
        health_details['security_events'] = recent_security_events
        for event in recent_security_events:
            if event['event_type'] == 'login_failure' and event['severity'] == 'WARNING':
                # Alert already sent by security manager, but monitor confirms it's being logged
                pass 

        if not overall_health:
            self.notification_manager.send_alert("overall_health_warning", "Overall system health is degraded!", "WARNING")
        else:
            logger.info("Overall system health is operational.")
            
        # Send a periodic health report
        self.notification_manager.send_report("system_health_summary", health_details)

    def _check_database_health(self) -> Dict:
        """Checks database connectivity and basic query latency."""
        try:
            start_time = time.perf_counter()
            with self.data_framework.get_connection() as conn:
                # Perform a simple, fast query to check connectivity and latency
                conn.execute("SELECT 1")
            latency_ms = (time.perf_counter() - start_time) * 1000
            return {"connected": True, "latency_ms": latency_ms}
        except Exception as e:
            logger.error(f"Database health check failed: {e}", exc_info=False)
            return {"connected": False, "error": str(e)}

# --- Standalone Testing Example ---
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')

    # Mock Redis Client
    class MockRedis:
        def __init__(self):
            self.data = {}
            self.expirations = {}
            self.connection_pool = type('obj', (object,), {'connection_kwargs': {'host': 'mock_redis', 'port': 6379, 'db': 0}})()
        def ping(self): return True
        def incr(self, key): self.data[key] = self.data.get(key, 0) + 1; return self.data[key]
        def get(self, key): return self.data.get(key)
        def delete(self, key): self.data.pop(key, None); self.expirations.pop(key, None)
        def exists(self, key): return key in self.data
        def setex(self, key, ttl, value): self.data[key] = value; self.expirations[key] = datetime.utcnow() + timedelta(seconds=ttl)
        def close(self): pass # Mock close

    # Mock FinancialDataFramework
    class MockFinancialDataFramework:
        def get_connection(self):
            class MockConnection:
                def __enter__(self): return self
                def __exit__(self, exc_type, exc_val, exc_tb): pass
                def execute(self, query): pass # Mock execute
            return MockConnection()
        def get_usage_report(self):
            return {'apis': {'twelvedata': {'calls': 1000, 'errors': 10}, 'alpha_vantage': {'calls': 500, 'errors': 30}}}

    mock_redis = MockRedis()
    mock_fdf = MockFinancialDataFramework()
    mock_notification_manager = NotificationManager(config={}) # Use the real NotificationManager

    security_manager = AdvancedSecurityManager(redis_client=mock_redis, notification_manager=mock_notification_manager)
    performance_optimizer = PerformanceOptimizer(redis_client=mock_redis)
    monitoring_system = ProductionMonitoringSystem(security_manager, performance_optimizer, mock_fdf, mock_notification_manager)

    # Test JWT generation and verification
    user_id = 123
    token = security_manager.generate_jwt_token(user_id)
    print(f"Generated JWT: {token}")
    payload = security_manager.verify_jwt_token(token)
    print(f"Verified Payload: {payload}")

    # Test authentication and brute force protection
    print("\n--- Testing Authentication and Brute Force ---")
    security_manager.register_user("testuser_brute", "password123")
    for i in range(security_manager.max_login_attempts + 1):
        print(f"Login attempt {i+1} for testuser_brute...")
        auth_result = security_manager.authenticate_user("testuser_brute", "wrongpass")
        print(f"Auth result: {auth_result}")
        if auth_result is None and security_manager.is_user_blocked("testuser_brute"):
            print(f"User testuser_brute is blocked after {i+1} attempts.")
            break
        time.sleep(0.1)

    # Start monitoring system
    print("\n--- Starting Production Monitoring System ---")
    monitoring_system.start_monitoring()
    print("Monitoring will run in background for 10 seconds...")
    await asyncio.sleep(10) # Let it run for a bit

    # Get audit log
    print("\n--- Security Audit Log ---")
    audit_log = security_manager.get_audit_log()
    for entry in audit_log:
        print(entry)

    # Stop monitoring
    monitoring_system.stop_monitoring()
    print("\nUltimate Production Security testing complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Ultimate Production Security example interrupted by user.")
    except Exception as e:
        logger.critical(f"Ultimate Production Security example failed: {e}", exc_info=True)
        sys.exit(1)

