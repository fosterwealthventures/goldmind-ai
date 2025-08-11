import os
import logging
import asyncio
from flask import Flask, jsonify, request
from flask_cors import CORS

# --- Imports from your system ---
from config import ProductionConfigManager
from financial_data_framework import FinancialDataFramework
from lstm_temporal_analysis import LSTMTemporalAnalysis
from cognitive_bias_detector import CognitiveBiasDetector
from goldmind_client import GoldMINDClient
from dual_system_conflict_resolution import DualSystemConflictResolver

# --------- LOGGING SETUP ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Entrypoint")

# --------- FLASK APP SETUP ----------
app = Flask(__name__)
CORS(app)

# --------- CONFIGURATION LOAD ----------
config_manager = ProductionConfigManager()
config_manager.load_config_from_file("config.json")
config_manager.load_env_file(".env")
config = config_manager.config_data

# --------- MODEL & SCALER LOAD ----------
MODEL_PATH = "lstm_models/SPY_lstm_model.h5"
SCALER_PATH = "lstm_models/scalers.joblib"

try:
    lstm_analysis = LSTMTemporalAnalysis(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH
    )
    logger.info("✅ LSTM model and scaler loaded successfully.")
except Exception as e:
    logger.critical(f"❌ Failed to load LSTM model or scaler: {e}")
    lstm_analysis = None

# --------- GOLDMIND CLIENT AND BIAS DETECTOR ----------
try:
    goldmind_api_url = config.get("GOLDMIND_API_URL", "http://127.0.0.1:8000/v1")
    goldmind_client = GoldMINDClient(api_url=goldmind_api_url)
    cognitive_detector = CognitiveBiasDetector(goldmind_client)
    logger.info("✅ CognitiveBiasDetector initialized with GoldMIND client.")
except Exception as e:
    logger.critical(f"❌ Failed to initialize CognitiveBiasDetector or GoldMINDClient: {e}")
    cognitive_detector = None

# --------- DUAL SYSTEM CONFLICT RESOLVER ----------
try:
    conflict_resolver = DualSystemConflictResolver()
    logger.info("✅ DualSystemConflictResolver loaded.")
except Exception as e:
    logger.critical(f"❌ Failed to load DualSystemConflictResolver: {e}")
    conflict_resolver = None

# --------- ENDPOINTS ----------

@app.route("/health", methods=["GET"])
def health_check():
    try:
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        logger.exception("Health check error:")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if lstm_analysis is None:
        logger.error("LSTM model is not loaded.")
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.json
        if not data or "input" not in data:
            return jsonify({"error": "Missing 'input' in request body."}), 400

        user_input = data["input"]
        logger.info(f"Received prediction request: {user_input}")

        predicted, confidence = lstm_analysis.predict(user_input)

        return jsonify({
            "confidence": float(confidence),
            "input": user_input,
            "predicted_gold_price": float(predicted)
        }), 200
    except Exception as e:
        logger.exception("Prediction error:")
        return jsonify({"error": str(e)}), 500

@app.route("/analyze/text", methods=["POST"])
def analyze_text():
    if cognitive_detector is None:
        logger.error("CognitiveBiasDetector is not loaded.")
        return jsonify({"error": "Text analysis module not loaded"}), 500
    try:
        data = request.json
        if not data or "text" not in data or "user_id" not in data:
            return jsonify({"error": "Missing 'text' or 'user_id' in request body."}), 400

        input_text = data["text"]
        user_id = data["user_id"]
        logger.info(f"Received text analysis request: {input_text} for user {user_id}")

        analysis_result = asyncio.run(cognitive_detector.analyze_input(input_text, user_id))

        return jsonify({
            "input": input_text,
            "user_id": user_id,
            "analysis": analysis_result
        }), 200
    except Exception as e:
        logger.exception("Text analysis error:")
        return jsonify({"error": str(e)}), 500

@app.route("/resolve", methods=["POST"])
def resolve():
    if lstm_analysis is None or cognitive_detector is None or conflict_resolver is None:
        logger.error("A required subsystem is not loaded.")
        return jsonify({"error": "One or more systems not loaded."}), 500
    try:
        data = request.json
        if not data or "input" not in data or "text" not in data or "user_id" not in data:
            return jsonify({"error": "Missing 'input', 'text', or 'user_id' in request body."}), 400

        user_input = data["input"]
        input_text = data["text"]
        user_id = data["user_id"]
        logger.info(f"Received resolve request: input={user_input}, text='{input_text}', user_id={user_id}")

        # 1. Get LSTM prediction
        predicted, confidence = lstm_analysis.predict(user_input)

        # 2. Get cognitive bias analysis (async)
        bias_result = asyncio.run(cognitive_detector.analyze_input(input_text, user_id))

        # 3. Combine with dual/conflict resolution logic
        # You may need to update this call based on your real class method signatures:
        final_result = conflict_resolver.resolve(
            prediction=predicted,
            confidence=confidence,
            user_text=input_text,
            bias_analysis=bias_result
        )

        return jsonify({
            "input": user_input,
            "text": input_text,
            "user_id": user_id,
            "lstm_prediction": float(predicted),
            "lstm_confidence": float(confidence),
            "bias_analysis": bias_result,
            "final_decision": final_result
        }), 200
    except Exception as e:
        logger.exception("Resolve error:")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
