# compute/server.py
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_NAME = "goldmind-compute"
APP_VERSION = os.getenv("APP_VERSION", "v1.2.1-insights")
REVISION = os.getenv("K_REVISION") or os.getenv("REVISION_ID") or "dev"
INTERNAL_SHARED_SECRET = os.getenv("INTERNAL_SHARED_SECRET", "")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/production_goldmind_v1.h5")
MODEL_SHA256 = os.getenv("MODEL_SHA256", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# -----------------------------------------------------------------------------
# CORS (harmless here, but nice for local debugging)
# -----------------------------------------------------------------------------
_allow_origins = os.getenv("ALLOW_ORIGINS", "")
ALLOW_ORIGINS: List[str] = []
if _allow_origins.strip():
    ALLOW_ORIGINS = [o.strip() for o in _allow_origins.split(",") if o.strip()]
if "*" in ALLOW_ORIGINS:
    ALLOW_ORIGINS = ["*"]

# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class PredictBody(BaseModel):
    """
    Minimal, stable payload shape.
    - `symbol` is the instrument (e.g., "XAUUSD")
    - `horizon` is a friendly window descriptor (e.g., "1d", "1w", "1m")
    - `amount` optional sizing (frontend uses it for scenario calculations)
    - `style` "day"|"close"|"intraday" etc. (freeform; used by callers)
    - `interval` granular candle interval (default "1d")
    - `indicators` which technical/macro features the caller wants computed
    - `options` bag for future extension (thresholds, lookbacks, etc)
    """
    symbol: str = Field(..., examples=["XAUUSD", "GLD", "GC=F"])
    horizon: str = Field(..., examples=["1d", "1w", "1m"])
    amount: Optional[float] = Field(default=None, examples=[1000])
    style: Optional[str] = Field(default="day")
    interval: Optional[str] = Field(default="1d")
    indicators: Optional[List[str]] = Field(default=None)
    options: Optional[Dict[str, Any]] = Field(default=None)


class PredictResult(BaseModel):
    ok: bool
    symbol: str
    method: str
    prediction: Optional[float] = None
    confidence: Optional[float] = None
    indicators_requested: Optional[List[str]] = None
    indicators: Optional[Dict[str, Any]] = None
    service: str = APP_NAME
    detail: Optional[str] = None
    meta: Dict[str, Any]


class HealthResult(BaseModel):
    env: str = Field(default=os.getenv("ENV", "prod"))
    service: str = APP_NAME
    status: str = "ok"
    version: str = APP_VERSION
    time: str = ""
    model_loaded: bool = False
    model_present: bool = False
    scaler_present: bool = True
    revision: str = REVISION


# -----------------------------------------------------------------------------
# Tiny model shim (you can replace with your real loader)
# -----------------------------------------------------------------------------
class _Model:
    def __init__(self, path: str, sha256: str = ""):
        self.path = path
        self.sha256 = sha256
        self.present = bool(path)  # we treat path-existing as "present" flag
        self.loaded = False
        self.detail = {"path": path, "sha256": sha256}

    def load(self) -> None:
        # Replace with your actual load logic (tensorflow/torch/sklearn)
        # Keep it quick—Cloud Run cold start does better with lazy load
        self.loaded = self.present  # pretend load works if present

    def predict(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a small, stable surface for the API layer to consume.
        If you have a real model, compute the output and a confidence.
        """
        if not self.loaded:
            return {"method": "naive", "prediction": None, "confidence": None}

        # Example deterministic pseudo prediction (no randomness for testability)
        # You should replace this with your real logic.
        symbol = body.get("symbol", "UNK")
        horizon = body.get("horizon", "1d")
        # make a simple hash-ish signal to prove data flows end-to-end
        key = f"{symbol}:{horizon}"
        score = float(abs(hash(key)) % 1000) / 1000.0  # 0.000 - 0.999
        conf = 0.50 + (score / 2.0)  # 0.50 - 0.999
        return {"method": "model", "prediction": round(score, 3), "confidence": round(conf, 3)}


MODEL = _Model(MODEL_PATH, MODEL_SHA256)


def meta() -> Dict[str, Any]:
    return {
        "version": APP_VERSION,
        "revision": REVISION,
        "model": MODEL.detail,
    }


def _require_internal_secret(x_internal_secret: Optional[str]) -> None:
    if not INTERNAL_SHARED_SECRET:
        # If you *really* want to run fully open, set INTERNAL_SHARED_SECRET=""
        # Not recommended in prod—this is a guard-rail.
        return
    if not x_internal_secret or x_internal_secret != INTERNAL_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized (internal secret required)")


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

if ALLOW_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOW_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthResult, tags=["health"])
@app.get("/admin/compute/health", response_model=HealthResult, include_in_schema=False)
def health() -> HealthResult:
    hr = HealthResult()
    hr.time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    hr.model_present = MODEL.present
    hr.model_loaded = MODEL.loaded
    return hr


@app.get("/version", tags=["meta"])
def version() -> Dict[str, Any]:
    return {"service": APP_NAME, "version": APP_VERSION, "revision": REVISION}


@app.post("/predict", response_model=PredictResult, tags=["predict"])
def predict(
    body: PredictBody,
    x_internal_secret: Optional[str] = Header(default=None),
):
    """
    Primary compute entry: returns a basic prediction + any requested indicators.
    Secured by X-Internal-Secret so only the API layer can call this.
    """
    _require_internal_secret(x_internal_secret)

    if not MODEL.loaded:
        # Lazy-load on first use to keep cold starts snappy
        MODEL.load()

    # 1) Model output (real code goes inside MODEL.predict)
    model_out = MODEL.predict(body.dict())

    # 2) (Optional) Lightweight indicator synthesis (quick & local)
    #    Real/expensive data gathering belongs in the API's "insights" layer.
    requested = body.indicators or []
    indicators: Dict[str, Any] = {}
    if requested:
        # Put simple, deterministic placeholders here so the API can enrich
        for name in requested:
            indicators[name] = {"ok": True, "note": "computed upstream in insights", "value": None}

    return PredictResult(
        ok=True,
        symbol=body.symbol.upper(),
        method=model_out.get("method", "naive"),
        prediction=model_out.get("prediction"),
        confidence=model_out.get("confidence"),
        indicators_requested=requested or None,
        indicators=indicators or None,
        detail=None,
        meta=meta(),
    )


@app.post("/admin/reload", tags=["admin"])
def admin_reload(x_internal_secret: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """
    Reload model/assets at runtime. Requires the internal secret.
    """
    _require_internal_secret(x_internal_secret)
    MODEL.load()
    return {"ok": True, "detail": MODEL.detail, "meta": meta()}


# -----------------------------------------------------------------------------
# Dev entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # For local runs: uvicorn compute.server:app --reload --port 8081
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8081")),
        reload=bool(os.getenv("DEV_RELOAD", "0") == "1"),
    )
