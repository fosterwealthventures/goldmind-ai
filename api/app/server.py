import os
import logging
from typing import Dict, Any
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from .aggregator import forward_to_compute

APP_VERSION = os.getenv("APP_VERSION", "v1.0.0")
COMPUTE_URL = os.getenv("COMPUTE_URL", "").rstrip("/")
INTERNAL_SHARED_SECRET = os.getenv("INTERNAL_SHARED_SECRET", "")

def create_app() -> Flask:
    app = Flask(__name__)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1)  # Cloud Run proxy
    CORS(app)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - api - %(levelname)s - %(message)s"
    )
    log = logging.getLogger("api")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "api", "version": APP_VERSION}), 200

    @app.get("/version")
    def version():
        return jsonify({"version": APP_VERSION}), 200

    @app.post("/predict")
    def predict():
        # Very lean validation—no heavy libs here
        payload: Dict[str, Any] = request.get_json(silent=True) or {}
        symbol = payload.get("symbol", "XAU")
        user = payload.get("user", {})
        if not COMPUTE_URL:
            return jsonify({"error": "COMPUTE_URL not configured"}), 500

        try:
            result = forward_to_compute(
                compute_url=f"{COMPUTE_URL}/compute/predict",
                json_payload={"symbol": symbol, "user": user, "options": payload.get("options", {})},
                shared_secret=INTERNAL_SHARED_SECRET,
                timeout_sec=int(os.getenv("COMPUTE_TIMEOUT_SEC", "55"))
            )
            return jsonify(result), 200
        except Exception as e:
            log.exception("Predict forward failed")
            return jsonify({"error": "compute_unavailable", "detail": str(e)}), 502

    @app.post("/feedback")
    def feedback():
        # Keep this tiny; optionally forward/store later
        data = request.get_json(silent=True) or {}
        log.info(f"User feedback: {data}")
        return jsonify({"status": "received"}), 200

    return app

# Gunicorn entrypoint expects `app`
app = create_app()
