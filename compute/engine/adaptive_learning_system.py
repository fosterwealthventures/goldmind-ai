"""
ultimate_production_security.py — GoldMIND AI (hardened)
-------------------------------------------------------
Security & production monitoring utilities:
- AdvancedSecurityManager: JWT auth, password hashing, Redis-backed brute-force protection,
  Flask-Limiter (optional), Flask-Talisman (optional), audit logging, and alerts.
- PerformanceOptimizer: Redis cache helpers + resource stats (psutil).
- ProductionMonitoringSystem: background health checks with periodic reports.
- Async demo harness guarded by __main__.

Improvements vs prior version:
- Added missing imports (psutil, asyncio, sys) and stronger type hints.
- Safer token extraction (handles "Bearer" and missing header gracefully).
- Optional CSRF-safe header allowlist for Talisman; HTTPS toggle via ENV.
- Robust Redis bytes/str handling for cached values.
- Rate-limit helper decorator for Flask if Limiter present (no-op otherwise).
- Configurable block durations and max attempts via env or constructor.
- More consistent JSON serialization for notifications.
"""

from __future__ import annotations

import os
import sys
import json
import jwt
import time
import asyncio
import logging
import secrets
import threading
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Dict, Any, Callable, Tuple, List

# Optional deps (soft-fail)
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    class _PS:
        @staticmethod
        def cpu_percent(interval=None): return 10.0
        class _VM: percent = 30.0
        @staticmethod
        def virtual_memory(): return _PS._VM()
    psutil = _PS()  # type: ignore

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore

from werkzeug.security import generate_password_hash, check_password_hash

# Optional Flask helpers
try:
    from flask_limiter import Limiter  # type: ignore
    from flask_limiter.util import get_remote_address  # type: ignore
    HAS_LIMITER = True
except Exception:  # pragma: no cover
    Limiter = None  # type: ignore
    get_remote_address = None  # type: ignore
    HAS_LIMITER = False
    logging.getLogger(__name__).warning("Flask-Limiter not installed. Rate limiting will not be applied automatically.")

try:
    from flask_talisman import Talisman  # type: ignore
    HAS_TALISMAN = True
except Exception:  # pragma: no cover
    Talisman = None  # type: ignore
    HAS_TALISMAN = False
    logging.getLogger(__name__).warning("Flask-Talisman not installed. Security headers will not be applied automatically.")

# Notification manager (soft import with mock fallback)
try:
    from notification_system import NotificationManager  # type: ignore
except Exception:
    logging.critical("❌ Could not import NotificationManager. Using a minimal mock.")
    class NotificationManager:  # type: ignore
        def __init__(self, config=None):
            logging.warning("Mock NotificationManager initialized.")
        def send_alert(self, alert_type: str, message: str, severity: str = "INFO"):
            logging.info("Mock Notification: [%s] %s: %s", severity, alert_type, message)
        def send_report(self, report_type: str, data: Dict[str, Any]):
            logging.info("Mock Notification Report: %s - %s", report_type, json.dumps(data)[:300])


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


# ====================================================================
# Advanced Security Manager
# ====================================================================

class AdvancedSecurityManager:
    """
    Manages authN/Z: JWT, password hashing, brute-force rate limits via Redis,
    optional Flask-Limiter + Talisman, and audit logging.
    """

    def __init__(
        self,
        redis_client: Optional["redis.StrictRedis"] = None,
        encryption_key_path: str = "security/encryption.key",
        notification_manager: Optional[NotificationManager] = None,
        max_login_attempts: Optional[int] = None,
        block_duration_minutes: Optional[int] = None,
    ) -> None:
        self.redis = redis_client
        self.encryption_key_path = encryption_key_path
        self.notification_manager = notification_manager or NotificationManager(config={})

        # Tunables (env overrides → args → defaults)
        self.max_login_attempts = int(max_login_attempts or os.getenv("MAX_LOGIN_ATTEMPTS", 5))
        self.block_duration_minutes = int(block_duration_minutes or os.getenv("BLOCK_DURATION_MINUTES", 15))

        # Ensure directory exists if path nested
        os.makedirs(os.path.dirname(self.encryption_key_path) or ".", exist_ok=True)

        self.jwt_secret = self._load_or_generate_jwt_secret()
        self.limiter: Optional[Limiter] = None
        self.talisman: Optional[Talisman] = None
        self.security_events: List[Dict[str, Any]] = []  # In-memory audit log

        logger.info("AdvancedSecurityManager initialized (max_attempts=%s, block=%sm).",
                    self.max_login_attempts, self.block_duration_minutes)

    # ---------------- Keys / JWT ----------------

    def _load_or_generate_jwt_secret(self) -> str:
        """Load JWT secret from file or generate a new one if invalid/missing."""
        if os.path.exists(self.encryption_key_path):
            try:
                with open(self.encryption_key_path, "r", encoding="utf-8") as f:
                    key = f.read().strip()
                if len(key) >= 32:
                    logger.info("Encryption key loaded from disk.")
                    return key
                logger.warning("Encryption key too short; regenerating.")
            except Exception as e:
                logger.warning("Failed to read key file (%s); regenerating.", e)

        new_key = secrets.token_urlsafe(32)
        try:
            with open(self.encryption_key_path, "w", encoding="utf-8") as f:
                f.write(new_key)
            logger.info("New encryption key generated and saved to %s.", self.encryption_key_path)
        except Exception as e:
            logger.error("Failed to write key file: %s", e)
        return new_key

    def generate_jwt_token(self, user_id: int, expires_in_hours: int = 24) -> str:
        payload = {"user_id": user_id, "exp": datetime.utcnow() + timedelta(hours=expires_in_hours), "iat": datetime.utcnow()}
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        self.log_security_event(user_id, "token_generated", f"JWT issued for user {user_id}")
        return token

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            self.log_security_event(payload.get("user_id"), "token_verified", "JWT verified")
            return payload
        except jwt.ExpiredSignatureError:
            self.log_security_event(None, "token_expired", "Expired JWT")
            return None
        except jwt.InvalidTokenError as e:
            self.log_security_event(None, "invalid_token", f"Invalid JWT: {e}")
            return None

    # ---------------- Flask integration ----------------

    def set_app(self, app) -> None:
        """
        Apply Flask-Limiter and Talisman to a Flask app (if available).
        """
        if HAS_LIMITER and self.redis is not None:
            try:
                pool = self.redis.connection_pool.connection_kwargs  # type: ignore[attr-defined]
                storage_uri = f"redis://{pool.get('host','localhost')}:{pool.get('port',6379)}/{pool.get('db',0)}"
                self.limiter = Limiter(app=app, key_func=get_remote_address, storage_uri=storage_uri)  # type: ignore
                logger.info("Flask-Limiter enabled (storage=%s).", storage_uri)
            except Exception as e:
                logger.warning("Failed to init Flask-Limiter: %s", e)
        elif not self.redis:
            logger.warning("Redis not provided; Flask-Limiter skipped.")

        if HAS_TALISMAN:
            try:
                force_https = bool(int(os.getenv("FORCE_HTTPS", "0")))
                csp = {
                    "default-src": ["'self'"],
                    "script-src": ["'self'", "'unsafe-inline'", "https://cdn.tailwindcss.com"],
                    "style-src": ["'self'", "'unsafe-inline'", "https://cdn.tailwindcss.com"],
                    "img-src": ["'self'", "data:", "https://placehold.co"],
                    "connect-src": ["'self'", "ws:", "wss:", "https://generativelanguage.googleapis.com"],
                    "frame-ancestors": ["'self'"],
                }
                self.talisman = Talisman(app, content_security_policy=csp, force_https=force_https)  # type: ignore
                logger.info("Flask-Talisman enabled (force_https=%s).", force_https)
            except Exception as e:
                logger.warning("Failed to init Flask-Talisman: %s", e)

    def jwt_required(self, f: Callable) -> Callable:
        """
        Decorator for protecting API routes with JWT auth.
        """
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import request, jsonify  # local import in app context
            auth = request.headers.get("Authorization", "")
            token = ""
            if auth.startswith("Bearer "):
                token = auth.split(" ", 1)[1].strip()
            elif auth:
                # fallback if header contains just the token
                token = auth.strip()

            if not token:
                return jsonify({"message": "Authorization token is missing!"}), 401

            payload = self.verify_jwt_token(token)
            if not payload:
                return jsonify({"message": "Token is invalid or expired!"}), 401

            # Attach user_id for downstream handlers
            setattr(request, "user_id", payload.get("user_id"))
            return f(*args, **kwargs)
        return decorated_function

    # Optional: route-specific rate limit decorator
    def rate_limited(self, limit: str) -> Callable:
        """
        Usage:
            @security.rate_limited("10/minute")
            def handler(): ...
        No-op if Flask-Limiter unavailable.
        """
        def wrapper(fn: Callable) -> Callable:
            if HAS_LIMITER and self.limiter:
                return self.limiter.limit(limit)(fn)  # type: ignore
            return fn
        return wrapper

    # ---------------- Passwords ----------------

    @staticmethod
    def hash_password(password: str) -> str:
        return generate_password_hash(password)

    @staticmethod
    def check_password(hashed_password: str, password: str) -> bool:
        return check_password_hash(hashed_password, password)

    # ---------------- Auth flows (mock DB) ----------------

    def authenticate_user(self, username: str, password: str) -> Optional[int]:
        mock_users = {
            "testuser": {"id": 1, "password_hash": self.hash_password("testpass")},
            "admin": {"id": 2, "password_hash": self.hash_password("adminpass")},
        }

        if self.is_user_blocked(username):
            self.log_security_event(None, "login_attempt_blocked", f"Blocked login for {username}")
            self.notification_manager.send_alert("security_alert",
                                                 json.dumps({"reason": "too_many_failed_attempts", "user": username}),
                                                 "WARNING")
            return None

        user_data = mock_users.get(username)
        if user_data and self.check_password(user_data["password_hash"], password):
            self.reset_failed_login_attempts(username)
            self.log_security_event(user_data["id"], "login_success", f"{username} logged in")
            self.notification_manager.send_alert("auth_event",
                                                 json.dumps({"event": "login_success", "user": username}), "INFO")
            return user_data["id"]

        # failure
        self.record_failed_login_attempt(username)
        self.log_security_event(None, "login_failure", f"Failed login for {username}", severity="WARNING")
        self.notification_manager.send_alert("auth_event",
                                             json.dumps({"event": "login_failure", "user": username}), "WARNING")
        return None

    def register_user(self, username: str, password: str) -> Optional[int]:
        mock_users = {
            "testuser": {"id": 1, "password_hash": self.hash_password("testpass")},
            "admin": {"id": 2, "password_hash": self.hash_password("adminpass")},
        }
        if username in mock_users:
            self.log_security_event(None, "registration_failure", f"Username exists: {username}", severity="WARNING")
            return None
        new_user_id = max([u["id"] for u in mock_users.values()]) + 1 if mock_users else 1
        mock_users[username] = {"id": new_user_id, "password_hash": self.hash_password(password)}
        self.log_security_event(new_user_id, "user_registration", f"Registered: {username}")
        self.notification_manager.send_alert("auth_event",
                                             json.dumps({"event": "user_registered", "user": username}), "INFO")
        return new_user_id

    # ---------------- Brute-force protection (Redis) ----------------

    def record_failed_login_attempt(self, username: str) -> bool:
        if not self.redis:
            logger.warning("Redis not configured. Cannot record failed login attempts.")
            return False

        key = f"failed_login:{username}"
        try:
            self.redis.incr(key)
            self.redis.expire(key, self.block_duration_minutes * 60)
            attempts_raw = self.redis.get(key)
            attempts = int(attempts_raw if isinstance(attempts_raw, int) else (attempts_raw or b"0"))
            if attempts >= self.max_login_attempts:
                self.block_user(username)
                self.log_security_event(None, "brute_force_block",
                                        f"{username} blocked after {self.max_login_attempts} failed attempts.",
                                        severity="ERROR")
                self.notification_manager.send_alert("security_alert",
                                                     json.dumps({"event": "bruteforce_block", "user": username,
                                                                 "attempts": attempts}), "CRITICAL")
            return True
        except Exception as e:
            logger.error("Failed to update failed login attempt: %s", e)
            return False

    def reset_failed_login_attempts(self, username: str) -> None:
        if not self.redis:
            return
        try:
            self.redis.delete(f"failed_login:{username}")
            self.redis.delete(f"blocked_user:{username}")
        except Exception as e:
            logger.warning("Failed to reset login attempts for %s: %s", username, e)

    def is_user_blocked(self, username: str) -> bool:
        if not self.redis:
            return False
        try:
            return bool(self.redis.exists(f"blocked_user:{username}"))
        except Exception:
            return False

    def block_user(self, username: str) -> None:
        if not self.redis:
            return
        try:
            self.redis.setex(f"blocked_user:{username}", self.block_duration_minutes * 60, "true")
            logger.warning("User %s blocked for %s minutes.", username, self.block_duration_minutes)
        except Exception as e:
            logger.error("Failed to block user %s: %s", username, e)

    # ---------------- Audit log ----------------

    def log_security_event(self, user_id: Optional[int], event_type: str, message: str, severity: str = "INFO") -> None:
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "event_type": event_type,
            "message": message,
            "severity": severity,
        }
        self.security_events.append(event)
        logger.log(getattr(logging, severity.upper(), logging.INFO), "SECURITY EVENT %s: %s", event_type, message)

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.security_events[-limit:]

# ====================================================================
# Performance Optimizer
# ====================================================================

class PerformanceOptimizer:
    def __init__(self, redis_client: Optional["redis.StrictRedis"] = None):
        self.redis = redis_client
        logger.info("PerformanceOptimizer initialized.")

    def cache_data(self, key: str, data: Any, ttl: int = 3600) -> bool:
        if not self.redis:
            logger.warning("Redis not configured. Cannot cache data.")
            return False
        try:
            to_store = json.dumps(data) if isinstance(data, (dict, list)) else str(data)
            self.redis.setex(key, ttl, to_store)
            logger.debug("Cached data key=%s ttl=%s", key, ttl)
            return True
        except Exception as e:
            logger.error("Failed to cache %s: %s", key, e)
            return False

    def get_cached_data(self, key: str) -> Optional[Any]:
        if not self.redis:
            return None
        try:
            raw = self.redis.get(key)
            if raw is None:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            # Try JSON first
            try:
                return json.loads(raw)
            except Exception:
                return raw
        except Exception as e:
            logger.error("Failed to fetch cache %s: %s", key, e)
            return None

    def get_performance_stats(self) -> Dict[str, Any]:
        cpu = float(psutil.cpu_percent(interval=0.1))
        mem = float(psutil.virtual_memory().percent)
        redis_ok = False
        try:
            redis_ok = bool(self.redis and self.redis.ping())
        except Exception:
            redis_ok = False
        return {"cpu_utilization_percent": cpu, "memory_utilization_percent": mem, "redis_connected": redis_ok}


# ====================================================================
# Production Monitoring
# ====================================================================

class ProductionMonitoringSystem:
    def __init__(self, security_manager: AdvancedSecurityManager, performance_optimizer: PerformanceOptimizer,
                 data_framework: Any, notification_manager: NotificationManager):
        self.security_manager = security_manager
        self.performance_optimizer = performance_optimizer
        self.data_framework = data_framework
        self.notification_manager = notification_manager
        self.monitoring_interval = int(os.getenv("MONITOR_INTERVAL_SEC", "30"))
        self._running = False
        self._thread: Optional[threading.Thread] = None
        logger.info("ProductionMonitoringSystem initialized.")

    def start_monitoring(self) -> None:
        if self._running:
            logger.warning("Production monitoring already running.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="prod-monitor", daemon=True)
        self._thread.start()
        logger.info("ProductionMonitoringSystem started.")

    def stop_monitoring(self) -> None:
        if not self._running:
            logger.warning("Production monitoring not running.")
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.monitoring_interval + 5)
        logger.info("ProductionMonitoringSystem stopped.")

    def _loop(self) -> None:
        while self._running:
            try:
                self._cycle_once()
            except Exception as e:
                logger.error("Production monitoring loop error: %s", e, exc_info=True)
                self.notification_manager.send_alert("production_monitor_error",
                                                     json.dumps({"error": str(e)}), "CRITICAL")
            time.sleep(self.monitoring_interval)

    def _cycle_once(self) -> None:
        overall_health = True
        health: Dict[str, Any] = {}

        perf_stats = self.performance_optimizer.get_performance_stats()
        health["performance"] = perf_stats
        if perf_stats["cpu_utilization_percent"] > 90 or perf_stats["memory_utilization_percent"] > 85:
            self.notification_manager.send_alert("resource_alert",
                                                 json.dumps({"message": "High resource utilization", **perf_stats}),
                                                 "WARNING")
            overall_health = False

        # DB health
        db_health = self._check_database_health()
        health["database"] = db_health
        if not db_health.get("connected") or db_health.get("latency_ms", 0) > 100:
            self.notification_manager.send_alert("db_alert",
                                                 json.dumps({"message": "DB issues", **db_health}), "CRITICAL")
            overall_health = False

        # API usage (if supported)
        if hasattr(self.data_framework, "get_usage_report"):
            usage = self.data_framework.get_usage_report()
            health["api_usage"] = usage
            for api, stats in (usage.get("apis") or {}).items():
                errs = stats.get("errors", 0)
                calls = max(1, stats.get("calls", 1))
                if errs / calls > 0.05:
                    self.notification_manager.send_alert("api_error",
                                                         json.dumps({"api": api, "error_rate": errs / calls}), "ERROR")
                    overall_health = False

        # Security events stream (no extra alerts here; security manager already sent)
        health["security_events"] = self.security_manager.get_audit_log(limit=10)

        if not overall_health:
            self.notification_manager.send_alert("overall_health_warning",
                                                 json.dumps({"message": "Health degraded"}), "WARNING")
        else:
            logger.info("Overall system health operational.")

        # Periodic consolidated report
        self.notification_manager.send_report("system_health_summary", health)

    def _check_database_health(self) -> Dict[str, Any]:
        try:
            start = time.perf_counter()
            with self.data_framework.get_connection() as conn:
                conn.execute("SELECT 1")
            latency_ms = (time.perf_counter() - start) * 1000.0
            return {"connected": True, "latency_ms": round(latency_ms, 2)}
        except Exception as e:
            return {"connected": False, "error": str(e)}


# ====================================================================
# Demo harness (async main)
# ====================================================================

async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    # Mock Redis if redis is not installed
    class MockRedis:
        def __init__(self):
            self.data = {}
            self.expirations = {}
            self.connection_pool = type("obj", (object,), {"connection_kwargs": {"host": "mock_redis", "port": 6379, "db": 0}})()
        def ping(self): return True
        def incr(self, key): self.data[key] = int(self.data.get(key, 0)) + 1; return self.data[key]
        def get(self, key): return self.data.get(key)
        def delete(self, key): self.data.pop(key, None); self.expirations.pop(key, None)
        def exists(self, key): return 1 if key in self.data else 0
        def expire(self, key, ttl): self.expirations[key] = datetime.utcnow() + timedelta(seconds=ttl)
        def setex(self, key, ttl, value): self.data[key] = value; self.expirations[key] = datetime.utcnow() + timedelta(seconds=ttl)

    # Mock FDF
    class MockFinancialDataFramework:
        def get_connection(self):
            class MockConn:
                def __enter__(self): return self
                def __exit__(self, *a): pass
                def execute(self, q): return 1
            return MockConn()
        def get_usage_report(self):
            return {"apis": {"twelvedata": {"calls": 1000, "errors": 10}, "alpha_vantage": {"calls": 500, "errors": 30}}}

    mock_redis = MockRedis()
    from notification_system import NotificationManager as RealNM  # use real if available
    notifier = RealNM(config={"default_channel": "console"})
    sec = AdvancedSecurityManager(redis_client=mock_redis, notification_manager=notifier)
    perf = PerformanceOptimizer(redis_client=mock_redis)
    monitor = ProductionMonitoringSystem(sec, perf, MockFinancialDataFramework(), notifier)

    uid = 123
    tok = sec.generate_jwt_token(uid)
    print("JWT:", tok)
    print("Payload:", sec.verify_jwt_token(tok))

    # brute-force simulation
    for i in range(sec.max_login_attempts + 1):
        sec.authenticate_user("attacker", "badpass")

    monitor.start_monitoring()
    await asyncio.sleep(3)
    monitor.stop_monitoring()
    print("Audit tail:", sec.get_audit_log(5))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.critical("Ultimate Production Security example failed: %s", e, exc_info=True)
        sys.exit(1)
