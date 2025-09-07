# api/app/server.py
from __future__ import annotations

from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

APP_NAME = "GoldMIND AI"
APP_VERSION = "v1.2.1-insights"

app = FastAPI(title=APP_NAME, version=APP_VERSION)

# ---- CORS (loose now; restrict in prod if needed) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Models ----
class PredictPayload(BaseModel):
    symbol: str
    horizon: str
    amount: float
    style: Optional[str] = None
    indicators: Optional[List[str]] = None


# ---- Basic routes ----
@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "goldmind-api", "version": APP_VERSION}


@app.get("/api/health", include_in_schema=False)
async def health_alias() -> Dict[str, Any]:
    # Alias for setups that expect an /api prefix
    return {"status": "ok", "service": "goldmind-api", "version": APP_VERSION}


@app.get("/")
async def index() -> List[str]:
    return [
        "/health",
        "/predict",
        "/compute/{path}",
        # Aliases (for setups that mount behind an /api prefix)
        "/api/health",
        "/api/predict",
        "/api/compute/{path}",
    ]


# ---- Core prediction handler ----
async def _do_predict(payload: PredictPayload, x_api_key: Optional[str]) -> Dict[str, Any]:
    # TODO: swap in your real model + data fusion here
    return {
        "input": payload.model_dump(),
        "prediction": {"signal": "hold", "confidence": 0.50},
        "explanations": {
            "macro": [
                k
                for k in (payload.indicators or [])
                if k in {"dxy_corr", "real_yields", "cot", "miners", "vix_corr"}
            ]
        },
        "env": "prod",
        "service": "goldmind-api",
        "version": APP_VERSION,
    }


@app.post("/predict")
async def predict(
    payload: PredictPayload, x_api_key: Optional[str] = Header(default=None)
):
    return await _do_predict(payload, x_api_key)


# Back-compat & prefix aliases (not in schema to avoid clutter)
@app.post("/predict/", include_in_schema=False)
async def predict_trailing(
    payload: PredictPayload, x_api_key: Optional[str] = Header(default=None)
):
    return await _do_predict(payload, x_api_key)


@app.post("/api/predict", include_in_schema=False)
async def predict_api_alias(
    payload: PredictPayload, x_api_key: Optional[str] = Header(default=None)
):
    return await _do_predict(payload, x_api_key)


# ---- Compute proxy (and its /api alias) ----
ALLOWED_COMPUTE_HEADS = {
    "predict",
    "spot",
    "series",
    "summary",
    "structural",
    "midlayer",
    "blended",
}

async def _compute_proxy_impl(path: str, request: Request, x_api_key: Optional[str]):
    # Normalize path and split into segments
    norm = (path or "").strip().strip("/")
    segments = [seg for seg in norm.split("/") if seg]
    head = segments[0] if segments else ""

    if head not in ALLOWED_COMPUTE_HEADS:
        raise HTTPException(status_code=404, detail=f"Path '{head or path}' not allowed")

    if head == "predict":
        if request.method != "POST":
            raise HTTPException(status_code=405, detail="Method not allowed; use POST")
        try:
            payload_json = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        try:
            payload = PredictPayload(**payload_json)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Bad payload: {e}")
        return await _do_predict(payload, x_api_key)

    raise HTTPException(status_code=404, detail=f"Unhandled compute path '{head}'")


# Make compute endpoints POST-only for clarity (GET not needed for predict)
@app.post("/compute/{path:path}")
async def compute_proxy(
    path: str, request: Request, x_api_key: Optional[str] = Header(default=None)
):
    return await _compute_proxy_impl(path, request, x_api_key)


@app.post("/api/compute/{path:path}", include_in_schema=False)
async def compute_proxy_api_alias(
    path: str, request: Request, x_api_key: Optional[str] = Header(default=None)
):
    return await _compute_proxy_impl(path, request, x_api_key)
