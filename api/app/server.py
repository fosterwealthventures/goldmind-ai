# api/app/server.py
"""
GoldMIND API - robust server (FastAPI)

- Loads local .env in dev (skips on Cloud Run)
- Structural + midlayer snapshots (safe fallbacks)
- /api/series via provider cascade, /market/gold/spot via providers
- Legacy compatibility routes (awaited)
- Proxies /v1/* to COMPUTE_URL when set
"""

from __future__ import annotations

import os
import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Optional deps
try:
    import httpx
except Exception:
    httpx = None  # type: ignore

try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore


def _load_env_local() -> None:
    """Load .env files locally (skip on Cloud Run)."""
    if os.getenv("K_SERVICE"):
        return
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    here = Path(__file__).resolve()
    api_dir = here.parent
    repo_root = api_dir.parent.parent if (api_dir.name == "app" and api_dir.parent.name == "api") else api_dir.parent

    for root in (api_dir, api_dir.parent, repo_root):
        for name in (".env.local", ".env"):
            p = root / name
            if p.exists():
                load_dotenv(p.as_posix(), override=False)

_load_env_local()

# App + CORS
SERVICE_NAME = os.getenv("SERVICE_NAME", "goldmind-api")
APP_VERSION = os.getenv("APP_VERSION", "v1.2.1-insights")

app = FastAPI(title=SERVICE_NAME, version=APP_VERSION)

raw_origins = os.getenv("ALLOW_ORIGINS", "*").strip()
allow_origins = ["*"] if raw_origins == "*" else [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# -------- helpers (safe import) --------
def _import_optional(name: str):
    try:
        from importlib import import_module
        return import_module(f".{name}", package=__package__)
    except Exception:
        try:
            return __import__(name)
        except Exception:
            return None

# -------- structural (sync wrapper) --------
def structural_snapshot() -> Dict[str, Any]:
    mod = _import_optional("structural_insights")
    if mod:
        for attr in ("snapshot", "structural_snapshot", "get_structural", "insights"):
            fn = getattr(mod, attr, None)
            if callable(fn):
                try:
                    data = fn()
                    return data if isinstance(data, dict) else {"data": data, "as_of": now_iso()}
                except Exception as e:
                    return {"ok": False, "error": f"{type(e).__name__}: {e}", "as_of": now_iso()}
    return {
        "source": "FRED",
        "ok": True,
        "errors": [],
        "inflation_yoy": None,
        "real_10y_pct": None,
        "dollar_change_3m": None,
        "gold_trend_6m_slope": None,
        "gold_change_6m": None,
        "headline": "Macro snapshot unavailable",
        "as_of": now_iso(),
    }

# -------- midlayer (async wrapper) --------
async def midlayer_snapshot() -> dict:
    mod = _import_optional("midlayer_insight") or _import_optional("midlayer_insights")
    if not mod:
        return {"ok": False, "errors": ["midlayer_insight not found"], "as_of": now_iso()}

    # Prefer 'snapshot' function
    fn = getattr(mod, "snapshot", None)
    if not callable(fn):
        return {"ok": False, "errors": ["midlayer_insight.snapshot not available"], "as_of": now_iso()}

    try:
        result = fn()
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, dict) else {"ok": True, "data": result, "as_of": now_iso()}
    except Exception as e:
        return {"ok": False, "errors": [f"{type(e).__name__}: {e}"], "as_of": now_iso()}

# -------- yfinance fallback (used only if providers fail) --------
async def yf_series(symbol: str = None, interval: str = "1d", range_: str = "6mo") -> Dict[str, Any]:
    """
    Try multiple Yahoo tickers for gold to avoid transient symbol issues.
    Priority: requested -> XAUUSD=X -> GC=F -> GLD
    """
    if yf is None:
        return {"symbol": symbol or "XAUUSD=X", "as_of": now_iso(), "spot": {"price_usd": None},
                "series": {"labels": [], "values": []}, "note": "yfinance not installed"}

    candidates = []
    for s in (symbol or os.getenv("YF_SYMBOL", "XAUUSD=X"), "XAUUSD=X", "GC=F", "GLD"):
        if s not in candidates:
            candidates.append(s)

    last_error = None
    for s in candidates:
        try:
            t = yf.Ticker(s)
            df = t.history(period=range_, interval=interval, auto_adjust=True)
            if df is None or df.empty:
                last_error = f"No price data found (symbol={s}, period={range_}, interval={interval})"
                continue
            labels = [d.strftime("%Y-%m-%d") for d in df.index]
            values = [float(v) for v in df["Close"].tolist()]
            spot = float(values[-1]) if values else None
            return {"symbol": s, "as_of": now_iso(), "spot": {"price_usd": spot},
                    "series": {"labels": labels, "values": values}}
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            continue

    return {"symbol": candidates[0], "as_of": now_iso(), "spot": {"price_usd": None},
            "series": {"labels": [], "values": []}, "error": last_error or "unavailable"}

# --------------------- Endpoints ---------------------
@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"env": os.getenv("ENV", "prod"), "service": SERVICE_NAME, "status": "ok",
            "version": APP_VERSION, "time": now_iso()}

@app.get("/api/series")
async def api_series(symbol: str = "XAU/USD", interval: str = "1day", range: str = "6mo"):  # noqa: A002
    """
    Primary series endpoint: try provider cascade via midlayer; fallback to Yahoo.
    """
    try:
        from .midlayer_insight import get_midlayer_series  # type: ignore
    except Exception:
        from .midlayer_insights import get_midlayer_series  # type: ignore

    try:
        s = await get_midlayer_series()
        if s.get("values"):
            return {
                "symbol": symbol,
                "as_of": now_iso(),
                "spot": {"price_usd": None},  # spot handled separately
                "series": {"labels": s["labels"], "values": s["values"]},
                "providers": s["providers"],
            }
    except Exception:
        pass

    # Fallback to Yahoo (robust cascade)
    return await yf_series(symbol=os.getenv("YF_SYMBOL", "XAUUSD=X"), interval="1d", range_=os.getenv("YF_PERIOD", "6mo"))

@app.get("/market/gold/spot")
async def market_gold_spot() -> dict:
    """
    Spot: Metalprice → Goldpricez → Yahoo (last close) fallback.
    """
    try:
        from .midlayer_insight import fetch_metalprice_spot, fetch_goldpricez_spot  # type: ignore
    except Exception:
        from .midlayer_insights import fetch_metalprice_spot, fetch_goldpricez_spot  # type: ignore

    # 1) Metalprice
    try:
        mp, mp_err = await fetch_metalprice_spot()
        if mp is not None:
            return {"symbol": "XAUUSD", "as_of": now_iso(),
                    "spot": {"price_usd": mp, "source": "metalprice"}, "error": mp_err}
    except Exception:
        pass

    # 2) Goldpricez
    try:
        gp, gp_err = await fetch_goldpricez_spot("USD")
        if gp is not None:
            return {"symbol": "XAUUSD", "as_of": now_iso(),
                    "spot": {"price_usd": gp, "source": "goldpricez"}, "error": gp_err}
    except Exception:
        pass

    # 3) Yahoo last close as a soft fallback
    data = await yf_series(symbol=os.getenv("YF_SYMBOL", "XAUUSD=X"), interval="1d", range_="1mo")
    return {"symbol": data.get("symbol"), "as_of": data.get("as_of"), "spot": data.get("spot")}

@app.get("/api/insights/structural")
async def api_structural() -> Dict[str, Any]:
    return structural_snapshot()

@app.get("/api/insights/midlayer")
async def api_midlayer() -> dict:
    return await midlayer_snapshot()

@app.get("/api/blended")
async def api_blended(symbol: str = "GLD", interval: str = "1d", range: str = "6mo") -> dict:  # noqa: A002
    series = await yf_series(symbol=symbol, interval=interval, range_=range)
    structural = structural_snapshot()          # sync
    midlayer = await midlayer_snapshot()        # async
    return {
        "env": os.getenv("ENV", "prod"),
        "service": SERVICE_NAME,
        "status": "ok",
        "version": APP_VERSION,
        "time": now_iso(),
        "series": series,
        "structural": structural,
        "midlayer": midlayer,
    }

# -------- Legacy compatibility (now awaited) --------
@app.get("/summary")
async def legacy_summary() -> Dict[str, Any]:
    return {"env": os.getenv("ENV", "prod"), "service": SERVICE_NAME, "status": "ok",
            "version": APP_VERSION, "time": now_iso()}

@app.get("/insight/structural", include_in_schema=False)
async def legacy_structural() -> Dict[str, Any]:
    return structural_snapshot()

@app.get("/insight/midlayer", include_in_schema=False)
async def legacy_midlayer() -> Dict[str, Any]:
    # FIX: await the async snapshot
    return await midlayer_snapshot()

# -------- Proxy to COMPUTE_URL --------
COMPUTE_URL = (os.getenv("COMPUTE_URL", "") or "").rstrip("/")

def _compute_ready() -> bool:
    return bool(COMPUTE_URL) and (httpx is not None)

async def _proxy(request: Request, path: str) -> Dict[str, Any]:
    if not _compute_ready():
        return {
            "status": 501,
            "detail": "COMPUTE_URL not set (or httpx missing). Set COMPUTE_URL, e.g. https://goldmind-compute-...run.app",
            "path": path,
        }

    target = f"{COMPUTE_URL}{path}"
    method = request.method.upper()
    params = dict(request.query_params)
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    body = await request.body()

    async with httpx.AsyncClient(timeout=30.0) as client:  # type: ignore[call-arg]
        resp = await client.request(method, target, params=params, content=body or None, headers=headers)
        try:
            payload = resp.json()
        except Exception:
            payload = {"text": resp.text}
        return {"status": resp.status_code, "upstream": target, "data": payload}

PROXY_GET = ["GET"]
PROXY_POST = ["POST"]
PROXY_BOTH = ["GET", "POST"]

def add_proxy_route(route: str, methods: List[str]) -> None:
    async def handler(req: Request):
        return await _proxy(req, route)
    app.add_api_route(route, handler, methods=methods)

# Register v1/* routes
add_proxy_route("/v1/predict", PROXY_BOTH)
add_proxy_route("/v1/alerts", PROXY_GET)
add_proxy_route("/feature-importance", PROXY_BOTH)
add_proxy_route("/v1/bias", PROXY_BOTH)
add_proxy_route("/v1/bias/influence", PROXY_GET)
add_proxy_route("/v1/settings", PROXY_BOTH)
add_proxy_route("/settings", PROXY_BOTH)
add_proxy_route("/v1/resolve", PROXY_BOTH)
add_proxy_route("/resolve", PROXY_BOTH)
add_proxy_route("/feedback", PROXY_POST)
add_proxy_route("/trace", PROXY_GET)

# Local runner
if __name__ == "__main__":
    import uvicorn  # type: ignore
    uvicorn.run("api.app.server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)
