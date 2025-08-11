# app/server.py
import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path

SETTINGS_FILE = Path("data/settings.json")
SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_SETTINGS = {
    "smooth_lines": True,
    "default_trading_style": "",
    "default_time_frame": ""
}

# Allowed values (adjust as you like)
ALLOWED_STYLES = {"Day Trading", "Swing Trading", "Position Trading", "Scalping", ""}
ALLOWED_TIMEFRAMES = {"15m", "30m", "1h", "4h", "1d", ""}

def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()

def save_settings(payload: dict) -> None:
    tmp = SETTINGS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(SETTINGS_FILE)

# --------------------------------------------------------------------------------------
# Optional imports from your project (models + storage)
# If not present, we provide inline fallbacks so this file works standalone.
# --------------------------------------------------------------------------------------
USE_EXTERNAL_MODELS = True
USE_EXTERNAL_TRACE_STORAGE = True

try:
    from app.models import Trace, TraceStep, Decision, iso_now  # type: ignore
except Exception:
    USE_EXTERNAL_MODELS = False

try:
    from app.storage_trace import save_trace, get_trace  # type: ignore
except Exception:
    USE_EXTERNAL_TRACE_STORAGE = False


# --------------------------------------------------------------------------------------
# Inline fallbacks (only used if your app.models / app.storage_trace are missing)
# --------------------------------------------------------------------------------------
if not USE_EXTERNAL_MODELS:
    from dataclasses import dataclass, asdict

    def iso_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @dataclass
    class TraceStep:
        ts: str
        type: str
        detail: str

    @dataclass
    class Decision:
        action: str
        entry: float
        target: float
        pips: int

    @dataclass
    class Trace:
        id: str
        created_at: str
        symbol: str
        decision: Optional[Decision]
        steps: List[TraceStep]
        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)

if not USE_EXTERNAL_TRACE_STORAGE:
    # Local JSON storage fallback (ephemeral in Cloud Run; fine for testing)
    import threading
    _DATA_DIR = "data"
    _TRACE_FILE = os.path.join(_DATA_DIR, "traces.json")
    os.makedirs(_DATA_DIR, exist_ok=True)
    _lock = threading.Lock()

    try:
        from google.cloud import firestore  # type: ignore
        _USE_FIRESTORE = True
    except Exception:
        _USE_FIRESTORE = False

    def _fs_client():
        from google.cloud import firestore  # type: ignore
        return firestore.Client()

    def _load_local_json(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_local_json(path: str, data: Dict[str, Any]):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_trace(doc: Dict[str, Any]):
        tid = doc.get("id")
        if not tid:
            raise ValueError("Trace must include 'id'")
        if _USE_FIRESTORE:
            db = _fs_client()
            db.collection("traces").document(tid).set(doc)
        else:
            with _lock:
                data = _load_local_json(_TRACE_FILE)
                data[tid] = doc
                _save_local_json(_TRACE_FILE, data)

    def get_trace(trace_id: str) -> Optional[Dict[str, Any]]:
        if _USE_FIRESTORE:
            db = _fs_client()
            snap = db.collection("traces").document(trace_id).get()
            return snap.to_dict() if snap.exists else None
        else:
            with _lock:
                data = _load_local_json(_TRACE_FILE)
                return data.get(trace_id)

# --------------------------------------------------------------------------------------
# Settings + Feedback storage (Firestore if available, else local JSON)
# --------------------------------------------------------------------------------------
_DATA_DIR = "data"
_SETTINGS_FILE = os.path.join(_DATA_DIR, "settings.json")
_FEEDBACK_FILE = os.path.join(_DATA_DIR, "feedback.json")
os.makedirs(_DATA_DIR, exist_ok=True)

try:
    from google.cloud import firestore  # type: ignore
    _FS_AVAILABLE = True
except Exception:
    _FS_AVAILABLE = False

def _fs():
    from google.cloud import firestore  # type: ignore
    return firestore.Client()

def load_settings(user_key: str = "global") -> Dict[str, Any]:
    if _FS_AVAILABLE:
        doc = _fs().collection("settings").document(user_key).get()
        if doc.exists:
            return doc.to_dict() or {}
        return {}
    else:
        if not os.path.exists(_SETTINGS_FILE):
            return {}
        try:
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                all_s = json.load(f)
            return all_s.get(user_key, {})
        except Exception:
            return {}

def save_settings(payload: Dict[str, Any], user_key: str = "global"):
    if _FS_AVAILABLE:
        _fs().collection("settings").document(user_key).set(payload)
    else:
        if os.path.exists(_SETTINGS_FILE):
            try:
                with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                    all_s = json.load(f)
            except Exception:
                all_s = {}
        else:
            all_s = {}
        all_s[user_key] = payload
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_s, f, indent=2)

def record_feedback(doc: Dict[str, Any]):
    if _FS_AVAILABLE:
        _fs().collection("feedback").add(doc)
    else:
        if os.path.exists(_FEEDBACK_FILE):
            try:
                with open(_FEEDBACK_FILE, "r", encoding="utf-8") as f:
                    arr = json.load(f)
            except Exception:
                arr = []
        else:
            arr = []
        arr.append(doc)
        with open(_FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(arr, f, indent=2)

# --------------------------------------------------------------------------------------
# App & helpers
# --------------------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # Allow your dashboard to call this API from the browser

API_VERSION = "v1.0.0"

def ok(payload: Dict[str, Any], code: int = 200):
    return jsonify(payload), code

def err(message: str, code: int = 400):
    return jsonify({"error": message}), code

def require_json() -> Dict[str, Any]:
    data = request.get_json(silent=True)
    if data is None:
        raise ValueError("Invalid or missing JSON body.")
    return data

def gen_rec_id(user_id: str = "anon") -> str:
    return f"REC-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{(user_id or 'anon')[:32]}"

# --------------------------------------------------------------------------------------
# Health

@app.route("/healthz", methods=["GET"])
@app.route("/health", methods=["GET"])
def healthz():
    return ok({"status": "ok", "version": API_VERSION})


# --------------------------------------------------------------------------------------
# Settings (GET, PUT)
# --------------------------------------------------------------------------------------
# ---------------- Settings (GET, PUT) ----------------
@app.route("/settings", methods=["GET"])
def settings_get():
    # (Optional: scope by user/token later)
    data = load_settings()
    return jsonify(data), 200

@app.route("/settings", methods=["PUT"])
def settings_put():
    payload = request.get_json(silent=True) or {}
    updated = {}

    # Validate and accept only known fields
    if "smooth_lines" in payload:
        if not isinstance(payload["smooth_lines"], bool):
            return jsonify({"error": "smooth_lines must be a boolean"}), 400
        updated["smooth_lines"] = payload["smooth_lines"]

    if "default_trading_style" in payload:
        v = str(payload["default_trading_style"]).strip()
        if v not in ALLOWED_STYLES:
            return jsonify({"error": f"default_trading_style must be one of {sorted(ALLOWED_STYLES)}"}), 400
        updated["default_trading_style"] = v

    if "default_time_frame" in payload:
        v = str(payload["default_time_frame"]).strip()
        if v not in ALLOWED_TIMEFRAMES:
            return jsonify({"error": f"default_time_frame must be one of {sorted(ALLOWED_TIMEFRAMES)}"}), 400
        updated["default_time_frame"] = v

    if not updated:
        return jsonify({"error": "no valid fields provided"}), 400

    current = load_settings()
    current.update(updated)
    save_settings(current)
    return jsonify(current), 200

# --------------------------------------------------------------------------------------
# Predict (LSTM stub)
# --------------------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected body: { "input": [numbers] }
    Returns: { "prediction": float, "model": "stub-lstm", "feature_importance": {...} }
    """
    try:
        body = require_json()
    except ValueError as e:
        return err(str(e), 400)

    arr = body.get("input")
    if not isinstance(arr, list) or not all(isinstance(x, (int, float)) for x in arr):
        return err("'input' must be a list of numbers", 400)

    if not arr:
        return err("'input' cannot be empty", 400)

    # Simple heuristic stub: next value ~ last + (last - mean)/2
    last = float(arr[-1])
    mean = sum(arr) / len(arr)
    pred = last + (last - mean) * 0.5

    fi = {"trend": 0.9, "volume": 0.7, "sentiment": 0.6, "economy": 0.5, "volatility": 0.4}

    return ok({
        "model": "stub-lstm",
        "input_len": len(arr),
        "prediction": round(pred, 5),
        "feature_importance": fi
    })

# --------------------------------------------------------------------------------------
# Analyze Text (Bias/Sentiment stub)
# --------------------------------------------------------------------------------------
@app.route("/analyze/text", methods=["POST"])
def analyze_text():
    """
    Expected body: { "text": "...", "user_id": "..." }
    Returns sentiment + simple bias flags (stub)
    """
    try:
        body = require_json()
    except ValueError as e:
        return err(str(e), 400)

    text = (body.get("text") or "").strip()
    user_id = (body.get("user_id") or "").strip()

    if not text or not user_id:
        return err("Both 'text' and 'user_id' are required", 400)

    # Super-naive sentiment: count pos/neg keywords as demo
    pos_kw = ["breakout", "bull", "up", "buy", "strong", "support"]
    neg_kw = ["breakdown", "bear", "down", "sell", "weak", "resistance"]
    pos = sum(1 for k in pos_kw if k in text.lower())
    neg = sum(1 for k in neg_kw if k in text.lower())
    score = pos - neg  # (-inf..+inf), just a toy

    biases = []
    if "sure" in text.lower() or "guarantee" in text.lower():
        biases.append("overconfidence")
    if "because last time" in text.lower():
        biases.append("anchoring")
    if "only read" in text.lower():
        biases.append("confirmation")

    return ok({
        "user_id": user_id,
        "sentiment_score": score,
        "biases": biases,
        "notes": "Stub analyzer; replace with your model/service."
    })

# --------------------------------------------------------------------------------------
# Resolve (Decision Engine) — creates a recommendation + persists a Trace
# --------------------------------------------------------------------------------------
@app.route("/resolve", methods=["POST"])
def resolve():
    """
    Expected body:
    {
      "input":[...], "text":"...", "user_id":"...", "trading_style":"...",
      "investment_amount": 1000, "time_frame":"1h", "symbol":"XAUUSD" (optional)
    }
    Returns: { id, symbol, decision:{action,entry,target,pips}, message }
    """
    try:
        body = require_json()
    except ValueError as e:
        return err(str(e), 400)

    seq = body.get("input", [])
    text = (body.get("text") or "").strip()
    user_id = (body.get("user_id") or "").strip() or "unknown"
    trading_style = (body.get("trading_style") or "").strip()
    time_frame = (body.get("time_frame") or "").strip()
    investment_amount = float(body.get("investment_amount", 0) or 0)
    symbol = (body.get("symbol") or "XAUUSD").upper()

    if (not isinstance(seq, list)) or not all(isinstance(x, (int, float)) for x in seq):
        return err("'input' must be a list of numbers", 400)
    if not seq or not text or not trading_style or not time_frame or investment_amount <= 0:
        return err("Missing or invalid required fields.", 400)

    # --- Very simple decision heuristic for demo ---
    # If last > first: BUY; else SELL
    direction_up = float(seq[-1]) >= float(seq[0])
    action = "BUY" if direction_up else "SELL"
    entry = float(seq[-1])
    # pips here are in cents for gold (1 pip = 0.01) — just a placeholder
    pips = 200 if direction_up else -200
    target = entry + (pips * 0.01)

    rec_id = gen_rec_id(user_id)
    created_at = iso_now()

    # Build Trace steps
    steps = [
        TraceStep(ts=created_at, type="input",
                  detail=f"Seq:{seq} user:{user_id} style:{trading_style} tf:{time_frame} invest:{investment_amount}"),
        TraceStep(ts=iso_now(), type="feature_importance",
                  detail="Trend 0.90, Volume 0.70, Sentiment 0.60, Econ 0.50, Vol 0.40"),
        TraceStep(ts=iso_now(), type="bias_check",
                  detail="Overconfidence mitigated; anchor risk low"),
        TraceStep(ts=iso_now(), type="decision",
                  detail=f"{action} {entry:.2f} → {target:.2f} ({pips} pips)"),
    ]

    trace = Trace(
        id=rec_id,
        created_at=created_at,
        symbol=symbol,
        decision=Decision(action=action, entry=entry, target=target, pips=abs(pips)),
        steps=steps
    )
    # Persist via Firestore or local JSON
    save_trace(trace.to_dict())

    return ok({
        "id": rec_id,
        "symbol": symbol,
        "decision": {
            "action": action,
            "entry": round(entry, 2),
            "target": round(target, 2),
            "pips": abs(pips)
        },
        "message": "Final decision computed."
    })

# --------------------------------------------------------------------------------------
# Trace endpoints (used by the frontend Scenario Traceability UI)
# --------------------------------------------------------------------------------------
@app.route("/trace/<trace_id>", methods=["GET"])
def api_trace_get(trace_id):
    doc = get_trace(trace_id)
    if not doc:
        return err("not found", 404)
    return ok(doc)

@app.route("/trace", methods=["POST"])
def api_trace_post():
    data = request.get_json(silent=True) or {}
    trace_id = (data.get("id") or "").strip()
    if not trace_id:
        return err("id is required", 400)
    doc = get_trace(trace_id)
    if not doc:
        return err("not found", 404)
    return ok(doc)

# --------------------------------------------------------------------------------------
# Feedback
# --------------------------------------------------------------------------------------
@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Expected:
    {
      "user_id":"...", "trading_style":"...", "investment_amount": 1000,
      "followed_recommendation":"yes|no|partial", "feedback_text":"..."
    }
    """
    data = request.get_json(silent=True) or {}
    user_id = (data.get("user_id") or "").strip()
    trading_style = (data.get("trading_style") or "").strip()
    investment_amount = float(data.get("investment_amount", 0) or 0)
    followed = (data.get("followed_recommendation") or "").strip().lower()
    feedback_text = (data.get("feedback_text") or "").strip()

    if not (user_id and trading_style and investment_amount > 0 and followed in {"yes","no","partial"} and feedback_text):
        return err("Invalid feedback payload", 400)

    doc = {
        "user_id": user_id,
        "trading_style": trading_style,
        "investment_amount": investment_amount,
        "followed_recommendation": followed,
        "feedback_text": feedback_text,
        "created_at": iso_now()
    }
    record_feedback(doc)
    return ok({"message": "feedback recorded"})

# --------------------------------------------------------------------------------------
# Logout (stateless demo)
# --------------------------------------------------------------------------------------
@app.route("/logout", methods=["POST"])
def logout():
    # Invalidate tokens in your real auth provider if needed.
    return ok({"message": "logged out"})

# --------------------------------------------------------------------------------------
# Error handlers
# --------------------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_):
    return err("endpoint not found", 404)

@app.errorhandler(405)
def method_not_allowed(_):
    return err("method not allowed", 405)

@app.errorhandler(500)
def internal_err(e):
    return err(f"internal error: {getattr(e, 'description', str(e))}", 500)

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # For local dev; Cloud Run will use gunicorn or the container's CMD.
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=bool(os.environ.get("DEBUG")))
