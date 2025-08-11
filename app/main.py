from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist, Field, validator
from typing import List, Optional
import logging, os, time

APP_NAME = "GoldMIND API"
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")

# ---------- Logging (Cloud Run friendly JSON) ----------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return '{"severity":"%s","message":"%s","logger":"%s","timestamp":%d}' % (
            record.levelname, record.getMessage().replace('"','\\"'),
            record.name, int(time.time())
        )
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
root = logging.getLogger()
root.setLevel(logging.INFO)
root.handlers = [handler]

log = logging.getLogger("goldmind")

# ---------- App ----------
app = FastAPI(title=APP_NAME, version=APP_VERSION)

# CORS: adjust to your dashboard origin(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fwvgoldmindai.com",
        "https://www.fwvgoldmindai.com",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schemas ----------
class PredictIn(BaseModel):
    input: conlist(float, min_items=2) = Field(..., description="Series of prices")
    @validator("input")
    def no_nan(cls, v):
        if any(x is None for x in v):
            raise ValueError("input contains null/None")
        return v

class AnalyzeIn(BaseModel):
    text: str = Field(..., min_length=3)
    user_id: str = Field(..., min_length=1)

class ResolveIn(BaseModel):
    input: conlist(float, min_items=2)
    text: str = Field(..., min_length=3)
    user_id: str = Field(..., min_length=1)
    trading_style: str = Field(..., min_length=2)
    investment_amount: float = Field(..., gt=0)
    time_frame: str = Field(..., min_length=1)

class FeedbackIn(BaseModel):
    user_id: str
    trading_style: str
    investment_amount: float = Field(..., gt=0)
    followed_recommendation: str
    feedback_text: str

# ---------- Health ----------
@app.get("/health")
def health():
    # Add any quick dependency checks here (envs, model file exists, etc.)
    return {"status": "ok", "service": APP_NAME, "version": APP_VERSION}

@app.get("/ready")
def ready():
    # Use for readiness—do lightweight checks that would fail if deps missing
    # e.g., env var present, model warm, etc.
    required_envs = ["SECRET_KEY"]  # adjust if you need
    missing = [e for e in required_envs if not os.getenv(e)]
    if missing:
        return {"ready": False, "missing_env": missing}
    return {"ready": True}

# ---------- Routes (safe fallbacks – no crashes) ----------
@app.post("/predict")
def predict(payload: PredictIn):
    # TODO: call your model
    # Return deterministic placeholder to prove path stability
    seq = payload.input
    if len(seq) >= 2:
        delta = seq[-1] - seq[-2]
    else:
        delta = 0
    yhat = seq[-1] + delta
    log.info("predict ok")
    return {"prediction": yhat, "last": seq[-1], "delta": delta}

@app.post("/analyze/text")
def analyze_text(payload: AnalyzeIn):
    # TODO: call your sentiment/bias pipeline
    log.info("analyze ok")
    return {
        "user_id": payload.user_id,
        "bias": {"overconfidence": 0.1, "anchoring": 0.2},
        "sentiment": {"label": "neutral", "score": 0.51},
    }

@app.post("/resolve")
def resolve(payload: ResolveIn):
    # TODO: call decision engine
    allocation = min(payload.investment_amount, 10000)  # toy logic
    decision = "BUY" if "buy" in payload.text.lower() else "HOLD"
    log.info("resolve ok")
    return {
        "decision": decision,
        "risk": "balanced",
        "allocation": allocation,
        "time_frame": payload.time_frame,
    }

@app.post("/feedback")
def feedback(payload: FeedbackIn):
    # TODO: persist feedback
    log.info("feedback ok")
    return {"message": "Feedback received. Thank you!"}

# ---------- Error handler (avoid ugly stacktraces) ----------
@app.middleware("http")
async def add_context_and_errors(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        log.error(f"Unhandled error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": "Something went wrong."},
        )

# FastAPI imports JSONResponse after definition to keep top tidy
from fastapi.responses import JSONResponse  # noqa
