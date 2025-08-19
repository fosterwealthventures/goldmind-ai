from __future__ import annotations
import os, platform
from typing import Any
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from .aggregator import forward_json

APP_VERSION = os.getenv("APP_VERSION", "v1.0.0")
ENV = os.getenv("ENV", "prod")
_CORS_RAW = os.getenv("CORS_ALLOW_ORIGINS", "https://fwvgoldmindai.com,https://www.fwvgoldmindai.com")
ALLOWED_ORIGINS = [o.strip() for o in _CORS_RAW.split(",") if o.strip()]

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=False)

@app.get("/health")
def health() -> Any:
    comp_json, comp_status = forward_json("/health", None, method="GET")
    return jsonify({
        "ok": True,
        "service": "api",
        "app_version": APP_VERSION,
        "compute_ok": comp_status == 200,
        "compute_state": comp_json,
    }), 200

@app.get("/version")
def version() -> Any:
    return jsonify({"service": "api", "app_version": APP_VERSION, "python": platform.python_version(), "env": ENV}), 200

@app.post("/predict")
def predict() -> Any:
    payload = request.get_json(silent=True) or {}
    data, status = forward_json("/predict", payload, method="POST")
    return make_response(data, status)

# --- SETTINGS endpoints (fixed: GET now implemented) ---
@app.get("/settings")
def get_settings() -> Any:
    data, status = forward_json("/settings", None, method="GET")
    return make_response(data, status)

@app.put("/settings")
def put_settings() -> Any:
    payload = request.get_json(silent=True) or {}
    data, status = forward_json("/settings", payload, method="POST")
    return make_response(data, status)

@app.get("/")
def root() -> Any:
    return jsonify({"service": "api", "status": "ok", "version": APP_VERSION}), 200

# Gunicorn looks for `app`
