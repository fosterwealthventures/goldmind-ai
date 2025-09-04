#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute service (Flask) for Cloud Run.
- Public:   GET /health
- Secreted: GET /market/series?symbol=XXXX&period=6mo&interval=1d
            GET /predict?symbol=XXXX&period=6mo&interval=1d
- Optional: GET /settings, GET /version
The "secreted" routes require header:  X-Internal-Secret: <value>
where <value> is supplied to the service via the INTERNAL_SHARED_SECRET
environment variable (ideally from a Secret Manager ref).
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, Optional, Tuple, List

from flask import Flask, jsonify, request, make_response
try:
    from flask_cors import CORS  # type: ignore
except Exception:  # pragma: no cover
    CORS = None  # CORS stay disabled if library not installed

# Quiet TensorFlow logs unless overridden
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", os.getenv("TF_CPP_MIN_LOG_LEVEL", "2"))

# --- TensorFlow / Keras (lazy) ------------------------------------------------
tf = None
load_model = None  # type: ignore

try:
    import tensorflow as _tf  # type: ignore
    tf = _tf
except Exception:
    tf = None

# loader compat for TF/Keras 2.15.x on CPU
if load_model is None:
    try:
        from tensorflow.keras.models import load_model as _lm  # type: ignore
        load_model = _lm
    except Exception:
        try:
            from keras.models import load_model as _lm2  # type: ignore
            load_model = _lm2
        except Exception:
            load_model = None  # pragma: no cover

# Optional deps
try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # graceful fallback later


# ------------------------ configuration helpers -------------------------------

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip())
    except Exception:
        return default

def env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [s.strip() for s in raw.split(",") if s.strip()]


SERVICE_NAME = os.getenv("SERVICE_NAME", "goldmind-compute")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
CORS_ALLOW_ORIGINS = env_list("CORS_ALLOW_ORIGINS", [])
CORS_ALLOW_METHODS = env_list("CORS_ALLOW_METHODS", ["GET", "POST", "OPTIONS"])
CORS_ALLOW_HEADERS = env_list("CORS_ALLOW_HEADERS", ["Accept", "Content-Type", "Authorization", "X-Requested-With", "X-Internal-Secret"])

HEALTH_ENABLE = env_bool("HEALTH_ENABLE", True)
PREDICT_ENABLE = env_bool("PREDICT_ENABLE", True)
SETTINGS_ENABLE = env_bool("SETTINGS_ENABLE", True)
VERSION_ENABLE = env_bool("VERSION_ENABLE", True)

# model/scaler config
MODEL_PATH = os.getenv("MODEL_PATH", "/app/lstm_models/SPY_lstm_model.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "/app/lstm_models/scalers.joblib")
MODEL_SHA256 = os.getenv("MODEL_SHA256", "")
SEQ_LEN = env_int("SEQ_LEN", 60)
FEATURES = env_list("FEATURES", ["close"])

# threading hints (safe with a single worker / few threads)
NUM_THREADS = env_int("NUM_THREADS", 1)
if tf is not None:
    try:
        tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
    except Exception:
        pass

# rate limiting toggles (noop here, but kept for compatibility)
RATE_LIMIT_ENABLED = env_bool("RATE_LIMIT_ENABLED", True)
RATE_LIMIT_WINDOW_SECONDS = env_int("RATE_LIMIT_WINDOW_SECONDS", 60)
RATE_LIMIT_MAX_REQUESTS = env_int("RATE_LIMIT_MAX_REQUESTS", 120)

JSONIFY_PRETTYPRINT = env_bool("JSONIFY_PRETTYPRINT", False)
DISABLE_SERVER_SIGNATURE = env_bool("DISABLE_SERVER_SIGNATURE", True)

# Shared secret (use Secret Manager on Cloud Run)
INTERNAL_SHARED_SECRET = os.getenv("INTERNAL_SHARED_SECRET", "").strip()

# ------------------------ app & logging ---------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(SERVICE_NAME)

app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = JSONIFY_PRETTYPRINT

if CORS and CORS_ALLOW_ORIGINS:
    CORS(app, resources={r"/*": {"origins": CORS_ALLOW_ORIGINS}}, methods=CORS_ALLOW_METHODS, allow_headers=CORS_ALLOW_HEADERS)
    log.info("CORS enabled for: %s", ", ".join(CORS_ALLOW_ORIGINS))
else:
    log.info("CORS disabled or not installed")

# Remove werkzeug server header if requested
if DISABLE_SERVER_SIGNATURE:
    @app.after_request
    def _strip_server_header(resp):
        resp.headers.pop("Server", None)
        return resp

# ------------------------ lazy loaders ----------------------------------------

_model = None
_scaler = None

def get_model():
    global _model
    if _model is not None:
        return _model
    if load_model is None:
        log.warning("Keras/TensorFlow load_model not available; predictions will degrade to naive.")
        return None
    if not os.path.exists(MODEL_PATH):
        log.warning("MODEL_PATH not found: %s", MODEL_PATH)
        return None
    try:
        _model = load_model(MODEL_PATH, compile=False)
        log.info("Model loaded from %s", MODEL_PATH)
    except Exception as e:
        log.exception("Failed to load model: %s", e)
        _model = None
    return _model

def get_scaler():
    global _scaler
    if _scaler is not None:
        return _scaler
    if joblib is None:
        return None
    if not os.path.exists(SCALER_PATH):
        return None
    try:
        _scaler = joblib.load(SCALER_PATH)  # type: ignore
        log.info("Scaler loaded from %s", SCALER_PATH)
    except Exception as e:
        log.warning("Failed to load scaler: %s", e)
        _scaler = None
    return _scaler


# ------------------------ helpers ---------------------------------------------

def _json(payload: Dict[str, Any], status: int = 200):
    resp = make_response(jsonify(payload), status)
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["Content-Type"] = "application/json; charset=utf-8"
    return resp

def _require_internal_secret() -> Optional[Tuple[Dict[str, Any], int]]:
    """Enforce shared secret for protected endpoints. Public routes must **not** call this."""
    if not INTERNAL_SHARED_SECRET:
        # If no secret configured, permit (useful for local dev)
        return None
    supplied = request.headers.get("X-Internal-Secret", "") or request.args.get("x_secret", "")
    if supplied != INTERNAL_SHARED_SECRET:
        return {"detail": "missing or invalid X-Internal-Secret", "error": "unauthorized"}, 401
    return None


# ------------------------ routes ----------------------------------------------

@app.get("/health")
def health():
    """Public health endpoint. MUST NOT require secret."""
    if not HEALTH_ENABLE:
        return _json({"detail": "disabled"}, 404)
    # don't force model load here—keep this light
    info = {
        "ok": True,
        "service": SERVICE_NAME,
        "revision": os.getenv("K_REVISION", ""),
        "model_present": os.path.exists(MODEL_PATH),
        "model_loaded": _model is not None,
        "scaler_present": os.path.exists(SCALER_PATH),
    }
    return _json(info, 200)


@app.get("/settings")
def settings():
    if not SETTINGS_ENABLE:
        return _json({"detail": "disabled"}, 404)
    guard = _require_internal_secret()
    if guard:
        return _json(*guard)
    data = {
        "SERVICE_NAME": SERVICE_NAME,
        "LOG_LEVEL": LOG_LEVEL,
        "MODEL_PATH": MODEL_PATH,
        "SCALER_PATH": SCALER_PATH,
        "MODEL_SHA256": MODEL_SHA256,
        "SEQ_LEN": SEQ_LEN,
        "FEATURES": FEATURES,
        "NUM_THREADS": NUM_THREADS,
        "CORS_ALLOW_ORIGINS": CORS_ALLOW_ORIGINS,
        "INTERNAL_SHARED_SECRET_len": len(INTERNAL_SHARED_SECRET),
    }
    return _json(data)


@app.get("/version")
def version():
    if not VERSION_ENABLE:
        return _json({"detail": "disabled"}, 404)
    return _json({"version": os.getenv("APP_VERSION", ""), "service": SERVICE_NAME})


@app.get("/market/series")
def market_series():
    guard = _require_internal_secret()
    if guard:
        return _json(*guard)
    if yf is None:
        return _json({"detail": "yfinance not installed"}, 503)
    symbol = (request.args.get("symbol") or "SPY").upper().strip()
    period = request.args.get("period", "6mo")
    interval = request.args.get("interval", "1d")
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return _json({"labels": [], "prices": [], "symbol": symbol})
        closes = df["Close"].astype(float)
        labels = [d.strftime("%Y-%m-%d") for d in closes.index.to_pydatetime()]
        prices = [round(float(x), 6) for x in closes.tolist()]
        return _json({"labels": labels, "prices": prices, "symbol": symbol})
    except Exception as e:
        log.exception("yfinance download failed: %s", e)
        return _json({"labels": [], "prices": [], "symbol": symbol, "error": "data_error"}, 502)


@app.get("/predict")
def predict():
    if not PREDICT_ENABLE:
        return _json({"detail": "disabled"}, 404)
    guard = _require_internal_secret()
    if guard:
        return _json(*guard)
    symbol = (request.args.get("symbol") or "SPY").upper().strip()
    period = request.args.get("period", "6mo")
    interval = request.args.get("interval", "1d")

    # Get the series first (re-use implementation)
    if yf is None:
        return _json({"detail": "yfinance not installed"}, 503)
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return _json({"symbol": symbol, "prediction": None, "method": "naive"})
        closes = df["Close"].astype(float).tolist()
    except Exception as e:
        log.exception("yfinance error: %s", e)
        return _json({"symbol": symbol, "prediction": None, "method": "naive"}, 502)

    # Lazy model load
    model = get_model()
    scaler = get_scaler()

    pred_val = None
    method = "naive"

    try:
        if model is not None and tf is not None and len(closes) >= SEQ_LEN:
            import numpy as np  # local import to keep boot fast
            window = closes[-SEQ_LEN:]
            X = np.array(window, dtype="float32").reshape(1, SEQ_LEN, 1)
            if scaler is not None:
                try:
                    X = scaler.transform(X.reshape(-1, 1)).reshape(1, SEQ_LEN, 1)  # type: ignore
                except Exception:
                    pass
            yhat = model.predict(X, verbose=0)
            val = float(yhat.squeeze())
            # If scaler is a standard scaler trained on y, inverse if possible
            try:
                if scaler is not None and hasattr(scaler, "inverse_transform"):
                    val = float(scaler.inverse_transform([[val]])[0][0])  # type: ignore
            except Exception:
                pass
            pred_val = val
            method = "lstm"
        else:
            # fallback: simple moving average of last N
            n = min(SEQ_LEN, max(1, len(closes)))
            pred_val = sum(closes[-n:]) / n
            method = "sma"
    except Exception as e:
        log.exception("prediction error: %s", e)
        pred_val = None
        method = "error"

    return _json({"symbol": symbol, "prediction": pred_val, "method": method})


# --------------- local dev runner (not used in Cloud Run with gunicorn) -------
if __name__ == "__main__":
    host = os.getenv("COMPUTE_HOST", "0.0.0.0")
    port = int(os.getenv("COMPUTE_PORT", os.getenv("PORT", "8080")))
    debug = LOG_LEVEL == "DEBUG"
    app.run(host=host, port=port, debug=debug)
