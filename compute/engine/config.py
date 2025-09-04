"""
config.py — GoldMIND AI unified configuration (updated for new dashboard)
-------------------------------------------------------------------------
Load order (lowest → highest precedence):
1) Built-in DEFAULTS
2) config.json (if present)
3) .env (KEYS with __ map to nested dicts, e.g., GOLDMIND__SERVER__PORT=8080)
4) Environment variables (same nested mapping)
5) Runtime overrides (passed to Config(..., overrides=...))

Key additions in this update:
- "dashboard" section: refresh intervals and feature flags
- "notifications" section: default transports/channels
- "system_utilities" section: health thresholds and backup policy
- "adaptive_learning" + "regime_detector": new knobs to match added modules
- Safer type coercion and helpers: get(), set(), save(), as_dict()

Usage:
    cfg = Config()
    port = cfg.get("server.port")
    cfg.set("dashboard.features.system_cards", True)
    cfg.save("config.runtime.json")
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, Optional

# ---------------- Defaults ----------------

DEFAULTS: Dict[str, Any] = {
    "app_name": "GoldMIND AI",
    "env": "development",  # development|staging|production
    "server": {
        "host": "0.0.0.0",
        "port": 5000,
        "debug": True,
        "cors_origins": ["http://localhost:5173", "http://localhost:3000"],
        "force_https": False,
        "request_timeout_sec": 30,
    },
    "database": {
        "path": "goldmind_ai.db",
        "echo": False,
        "pool_size": 5,
    },
    "redis": {
        "enabled": False,
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": None,
    },
    "market_data": {
        "update_interval": 30,
        "providers": ["alpha_vantage", "twelve_data", "fred"],
        "alpha_vantage_key": "",
        "twelve_data_key": "",
        "fred_key": "",
        "demo_mode": True,
    },
    "ml_models": {
        "lstm_temporal": {"enabled": True, "window": 60, "epochs": 10},
        "analytical_pathway": {"enabled": True},
    },
    "analytics": {
        "enabled": True,
        "sample_rate": 1.0,
    },
    "security": {
        "jwt_hours": 24,
        "max_login_attempts": 5,
        "block_duration_minutes": 15,
        "force_https": False,
    },
    "auto_hedging": {
        "enabled": True,
        "risk_trigger": 7.0,
        "circuit_breaker": 9.5,
        "monitoring_interval": 300,
        "auto_execute": False,
    },
    "model_performance": {
        "enabled": True,
        "eval_interval_min": 15,
    },
    "dual_system": {
        "default_weights": {"ANALYTICAL": 0.5, "LSTM_TEMPORAL": 0.5},
    },
    "goldmind_api": {
        "base_url": "https://api.goldmind.ai/v1",
        "api_key": "",
        "timeout_sec": 30,
    },
    # NEW: dashboard knobs
    "dashboard": {
        "refresh": {
            "cards_sec": 10,
            "charts_sec": 30,
            "system_sec": 20,
        },
        "features": {
            "system_cards": True,
            "regime_cards": True,
            "hedging_cards": True,
            "bias_cards": True,
        },
    },
    # NEW: notifications (used by notification_system)
    "notifications": {
        "default_channel": "console",          # console|email|webhook|twilio
        "email": {"from": "alerts@goldmind.ai", "smtp_host": "", "smtp_port": 587, "username": "", "password": ""},
        "webhook": {"url": ""},
        "twilio": {"sid": "", "token": "", "from": ""},
        "rate_limit_per_min": 30,
    },
    # NEW: system utilities knobs
    "system_utilities": {
        "disk_free_gb_warning": 5.0,
        "disk_free_gb_error": 1.0,
        "log_max_mb": 100.0,
        "backup_zip": True,
        "backup_dir": "backups",
        "log_dir": "logs",
    },
    # NEW: adaptive learning knobs
    "adaptive_learning": {
        "ema_decay": 0.9,
        "min_weight": 0.15,
        "max_weight": 0.85,
        "reward_scale": 1.0,
    },
    # NEW: market regime detector knobs
    "regime_detector": {
        "short_window": 20,
        "medium_window": 50,
        "long_window": 120,
        "vol_window": 14,
        "trend_threshold": 0.04,
        "bearish_threshold": -0.04,
        "vol_threshold": 0.03,
    },
}


# ---------------- Helpers ----------------

def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _set_nested(d: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _get_nested(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _coerce(value: str) -> Any:
    """Best-effort type coercion from string env values."""
    v = value.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _apply_env_into(cfg: Dict[str, Any], env: Dict[str, str]) -> None:
    """
    Map ENV keys with __ to nested paths (GOLDMIND__SERVER__PORT=8080 → server.port).
    Also accept dot paths in ENV (SERVER.PORT=8080).
    Only keys starting with GOLDMIND__ or in ALLOWLIST are applied.
    """
    allow_prefixes = ("GOLDMIND__", "SERVER.", "DATABASE.", "REDIS.", "DASHBOARD.", "NOTIFICATIONS.", "SYSTEM_UTILITIES.", "ADAPTIVE_LEARNING.", "REGIME_DETECTOR.", "GOLDMIND_API.")
    for key, val in env.items():
        if key.startswith("GOLDMIND__"):
            path = key.replace("GOLDMIND__", "").lower().replace("__", ".")
        elif key.startswith(allow_prefixes):
            path = key.lower().replace("__", ".")
        else:
            continue
        try:
            _set_nested(cfg, path, _coerce(val))
        except Exception:
            _set_nested(cfg, path, val)


def _load_dotenv(path: str = ".env") -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not os.path.exists(path):
        return data
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return data


# ---------------- Config class ----------------

class Config:
    def __init__(self, *, config_path: str = "config.json", overrides: Optional[Dict[str, Any]] = None) -> None:
        cfg: Dict[str, Any] = json.loads(json.dumps(DEFAULTS))  # deep copy

        # From config.json
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    file_cfg = json.load(f)
                _deep_update(cfg, file_cfg)
            except Exception as e:
                print(f"[config] Warning: failed to read {config_path}: {e}")

        # From .env (supports nested via __)
        dotenv = _load_dotenv(".env")
        _apply_env_into(cfg, dotenv)

        # From environment variables
        _apply_env_into(cfg, dict(os.environ))

        # Runtime overrides
        if overrides:
            _deep_update(cfg, overrides)

        # Finalize booleans that depend on env string
        if isinstance(cfg.get("server", {}).get("force_https"), str):
            cfg["server"]["force_https"] = str(cfg["server"]["force_https"]).lower() == "true"

        self._cfg = cfg

    # Basic accessors
    def get(self, path: str, default: Any = None) -> Any:
        return _get_nested(self._cfg, path, default)

    def set(self, path: str, value: Any) -> None:
        _set_nested(self._cfg, path, value)

    def as_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._cfg))

    def save(self, path: str = "config.generated.json") -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._cfg, f, indent=2)


# ---------------- Demo ----------------
if __name__ == "__main__":
    cfg = Config()
    print(json.dumps(cfg.as_dict(), indent=2)[:2000])
    print("Server:", cfg.get("server.host"), cfg.get("server.port"), "HTTPS:", cfg.get("server.force_https"))
    print("Dashboard enabled:", cfg.get("dashboard.features.system_cards"))
