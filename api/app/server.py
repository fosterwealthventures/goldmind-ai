# api/app/server.py
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Iterable

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from starlette.status import HTTP_200_OK

# -----------------------
# Configuration (env vars)
# -----------------------
SERVICE_NAME = os.getenv("SERVICE_NAME", "goldmind-api")
APP_VERSION = os.getenv("APP_VERSION", "v1.2.1-insights")
ENV = os.getenv("ENV", "prod")

# Compute wiring
COMPUTE_URL = (os.getenv("COMPUTE_URL", "") or "").rstrip("/")
INTERNAL_SHARED_SECRET = os.getenv("INTERNAL_SHARED_SECRET")

# Optional: limit which internal paths can be proxied (comma-separated)
# e.g. "health,job/run,predict"
ALLOWED_COMPUTE_PATHS: set[str] = set(
    p.strip() for p in os.getenv("COMPUTE_ALLOWED_PATHS", "health").split(",") if p.strip()
)

# -------------
# App lifecycle
# -------------
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # One shared outbound HTTP client for the process
    app.state.http = httpx.AsyncClient(timeout=30)
    try:
        yield
    finally:
        await app.state.http.aclose()

app = FastAPI(title=SERVICE_NAME, version=APP_VERSION, lifespan=lifespan)

# ----------------
# Helper functions
# ----------------
def now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _require_compute_config() -> None:
    if not COMPUTE_URL or not INTERNAL_SHARED_SECRET:
        raise HTTPException(
            status_code=500,
            detail="COMPUTE_URL or INTERNAL_SHARED_SECRET not configured on API service.",
        )

HOP_BY_HOP_HEADERS: Iterable[str] = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

def _passthrough_headers(src: Dict[str, str]) -> Dict[str, str]:
    # Drop hop-by-hop headers; keep everything else
    return {k: v for k, v in src.items() if k.lower() not in HOP_BY_HOP_HEADERS}

# -------
# Routes
# -------
@app.get("/health", status_code=HTTP_200_OK)
async def health() -> dict:
    return {
        "env": ENV,
        "service": SERVICE_NAME,
        "status": "ok",
        "version": APP_VERSION,
        "time": now_iso(),
    }

@app.get("/version", include_in_schema=False)
async def version() -> dict:
    return {"service": SERVICE_NAME, "version": APP_VERSION}

# Admin check that the API can reach Compute's internal health
@app.get("/admin/compute/health")
async def admin_compute_health():
    _require_compute_config()
    url = f"{COMPUTE_URL}/internal/health"
    async with app.state.http as client:
        r = await client.get(url, headers={"X-Internal-Secret": INTERNAL_SHARED_SECRET})
    if r.status_code >= 400:
        raise HTTPException(r.status_code, f"Compute health check failed: {r.text}")
    return r.json()

# Minimal, guarded proxy → Compute internal routes.
# Example: GET /compute/health  ->  GET {COMPUTE_URL}/internal/health
@app.api_route("/compute/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def compute_proxy(path: str, request: Request):
    _require_compute_config()

    # Restrict to a small allowlist to avoid becoming an open proxy
    # (Customize with COMPUTE_ALLOWED_PATHS env var)
    normalized = path.strip("/")
    if normalized not in ALLOWED_COMPUTE_PATHS:
        raise HTTPException(403, f"Path '{path}' not allowed")

    target = f"{COMPUTE_URL}/internal/{normalized}"
    # Forward body and query string; set internal auth header
    body = await request.body()
    headers = _passthrough_headers(dict(request.headers))
    headers["X-Internal-Secret"] = INTERNAL_SHARED_SECRET

    async with app.state.http as client:
        r = await client.request(
            method=request.method,
            url=target,
            content=body if body else None,
            headers=headers,
            params=dict(request.query_params),
        )

    # Build a FastAPI Response with upstream content & status
    content_type = r.headers.get("content-type", "application/octet-stream")
    return Response(content=r.content, status_code=r.status_code, media_type=content_type)

# Local dev convenience
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("DEV_RELOAD", "")),
    )
