import os
import logging
from typing import Any, Dict
from flask import Flask, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix

# --- Import your existing local modules (no model_loader) ---
# These should already be in the repo root or vendored here. If they're in repo root,
# consider copying them under compute/app/ or making a compute-specific package.
from engine.financial_data_framework import FinancialDataFramework
from engine.lstm_temporal_analysis import LSTMTemporalAnalysis
from engine.cognitive_bias_detector import CognitiveBiasDetector
from engine.dual_system_conflict_resolution import DualSystemConflictResolver
from engine.BIAS_AWARE_DUAL_SYSTEM_INTEGRATION import UltimateBiasAwareManager


APP_VERSION = os.getenv("APP_VERSION", "v1.0.0")
INTERNAL_SHARED_SECRET = os.getenv("INTERNAL_SHARED_SECRET", "")

def create_app() -> Flask:
    app = Flask(__name__)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - compute - %(levelname)s - %(message)s"
    )
    log = logging.getLogger("compute")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "compute", "version": APP_VERSION}), 200

    def require_internal_secret():
        secret = request.headers.get("X-Internal-Secret", "")
        if INTERNAL_SHARED_SECRET and secret != INTERNAL_SHARED_SECRET:
            return False
        return True

    @app.post("/compute/predict")
    def compute_predict():
        if not require_internal_secret():
            return jsonify({"error": "unauthorized"}), 401

        payload: Dict[str, Any] = request.get_json(silent=True) or {}
        symbol = payload.get("symbol", "XAU")
        user   = payload.get("user", {})    # trading style, risk, target, etc.
        opts   = payload.get("options", {}) # seq_len, features, etc.

        try:
            # Initialize core components (quickly). You may cache singletons if needed.
            fdf = FinancialDataFramework()
            lstm = LSTMTemporalAnalysis(
                instrument=symbol,
                seq_len=int(opts.get("seq_len", 60)),
                features=opts.get("features", ["Close"])
            )
            bias = CognitiveBiasDetector()
            dsys = DualSystemConflictResolver()
            mgr  = UltimateBiasAwareManager()

            # Fetch data / run LSTM
            price_pred = lstm.predict_next_close(fdf=fdf)

            # Bias/sentiment + decision fusion
            bias_report = bias.analyze(symbol=symbol, context={"user": user})
            decision    = dsys.resolve(price_prediction=price_pred, bias_report=bias_report, user=user)
            final_plan  = mgr.recommend(decision=decision, user=user, market="gold")

            result = {
                "symbol": symbol,
                "price_prediction": price_pred,
                "bias_report": bias_report,
                "decision": decision,
                "final_plan": final_plan,
                "version": APP_VERSION
            }
            return jsonify(result), 200

        except Exception as e:
            log.exception("compute error")
            return jsonify({"error": "compute_failed", "detail": str(e)}), 500

    return app

app = create_app()
