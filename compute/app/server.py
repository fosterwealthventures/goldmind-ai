from __future__ import annotations
import os, platform, time
from typing import Any, Dict
from flask import Flask, jsonify, request, make_response
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS

APP_VERSION = os.getenv("APP_VERSION", "v1.0.0")
ENV = os.getenv("ENV", "prod")
INTERNAL_SHARED_SECRET = os.getenv("INTERNAL_SHARED_SECRET", "")

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# CORS can be disabled; service is internal. Leaving it very restrictive:
CORS(app, resources={r"/*": {"origins": []}})

# In-memory settings for demo
_SETTINGS: Dict[str, Any] = {
    "default_symbol": "XAUUSD",
    "default_horizon": "1h",
    "risk_mode": "conservative"
}

def require_internal():
    def wrapper(fn):
        def inner(*args, **kwargs):
            hdr = request.headers.get("X-Internal-Secret", "")
            if not INTERNAL_SHARED_SECRET or hdr != INTERNAL_SHARED_SECRET:
                return make_response({"error": "unauthorized"}, 401)
            return fn(*args, **kwargs)
        inner.__name__ = fn.__name__
        return inner
    return wrapper

@app.get("/health")
def health() -> Any:
    return jsonify({"ok": True, "service": "compute", "app_version": APP_VERSION}), 200

@app.get("/version")
def version() -> Any:
    return jsonify({"service": "compute", "app_version": APP_VERSION, "python": platform.python_version(), "env": ENV}), 200

@app.post("/predict")
@require_internal()
def predict() -> Any:
    """
    Minimal placeholder that returns a deterministic 'recommendation'
    based on inputs (no randomness, so API can cache / degrade safely).
    """
    j = request.get_json(silent=True) or {}
    symbol = (j.get("symbol") or _SETTINGS["default_symbol"]).upper()
    horizon = j.get("horizon") or _SETTINGS["default_horizon"]
    amount = float(j.get("amount") or 0)
    style = j.get("style") or "day"

    # Toy logic: recommend BUY if amount <= 0 or style == 'swing'; else HOLD.
    rec = "BUY" if (amount <= 0 or style.lower() == "swing") else "HOLD"

    out = {
        "mode": "live",
        "symbol": symbol,
        "horizon": horizon,
        "inputs": {"amount": amount, "style": style},
        "recommendation": rec,
        "confidence": 0.62 if rec == "BUY" else 0.55,
        "generated_at": int(time.time()),
    }
    return make_response(out, 200)

@app.get("/settings")
@require_internal()
def get_settings() -> Any:
    return jsonify(_SETTINGS), 200

@app.post("/settings")
@require_internal()
def set_settings() -> Any:
    j = request.get_json(silent=True) or {}
    _SETTINGS.update({k: v for k, v in j.items() if k in _SETTINGS})
    return jsonify({"ok": True, "settings": _SETTINGS}), 200

@app.get("/")
def root() -> Any:
    return jsonify({"service": "compute", "status": "ok", "version": APP_VERSION}), 200
