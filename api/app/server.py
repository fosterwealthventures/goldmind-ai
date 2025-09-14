# api/app/server.py
from __future__ import annotations
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

APP_NAME = "GoldMIND AI"
APP_VERSION = "v1.2.1-insights"

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictPayload(BaseModel):
    symbol: str
    horizon: str
    amount: float
    style: Optional[str] = None
    indicators: Optional[List[str]] = None

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "goldmind-api", "version": APP_VERSION}

@app.get("/api/health", include_in_schema=False)
async def health_alias() -> Dict[str, Any]:
    return await health()

@app.get("/")
async def index() -> List[str]:
    return [
        "/health",
        "/market/gold/series",
        "/api/summary",
        "/v1/predict?symbol=XAUUSD",
        "/v1/bias?symbol=XAUUSD",
        "/v1/feature-importance",
        "/v1/bias/influence",
        "/v1/alerts",
        "/v1/trace?id=demo",
    ]

async def _do_predict(payload: PredictPayload, x_api_key: Optional[str]) -> Dict[str, Any]:
    return {
        "input": payload.model_dump(),
        "prediction": {"signal": "hold", "confidence": 0.50},
        "explanations": {
            "macro": [k for k in (payload.indicators or []) if k in {"dxy_corr","real_yields","cot","miners","vix_corr"}]
        },
        "env": "prod",
        "service": "goldmind-api",
        "version": APP_VERSION,
    }

@app.post("/predict")
async def predict(payload: PredictPayload, x_api_key: Optional[str] = Header(default=None)):
    return await _do_predict(payload, x_api_key)

@app.post("/api/predict", include_in_schema=False)
async def predict_api_alias(payload: PredictPayload, x_api_key: Optional[str] = Header(default=None)):
    return await _do_predict(payload, x_api_key)

# ---------- Frontend GET endpoints ----------
def _mock_series(days: int = 60):
    now = datetime.now(timezone.utc)
    ts = [(now - timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0) for d in range(days)][::-1]
    base = 1950.0
    px = [round(base + 5 * ((i % 7) - 3) + 0.1 * i, 2) for i in range(days)]
    return ts, px

@app.get("/market/gold/series")
async def market_gold_series() -> Dict[str, Any]:
    ts, px = _mock_series()
    return {
        "symbol": "XAUUSD",
        "price": px[-1],
        "change_24h": round(((px[-1]-px[-2]) / px[-2]) * 100, 2) if len(px) > 1 else 0.0,
        "timestamps": [t.isoformat() for t in ts],
        "prices": px,
    }

@app.get("/api/summary")
async def api_summary() -> Dict[str, Any]:
    return {
        "regime": "neutral",
        "momentum": "flat",
        "top_factors": ["real yields", "USD (DXY)", "miners breadth", "COT position"],
        "signals": [
            {"name": "trend_strength", "value": "moderate", "note": "50/200 cross stable"},
            {"name": "macro_bias", "value": "mixed", "note": "real yields off highs"},
        ],
        "version": APP_VERSION,
        "now": _utc_now(),
    }

@app.get("/v1/predict")
async def v1_predict(symbol: str = Query("XAUUSD"), x_api_key: Optional[str] = Header(default=None)):
    payload = PredictPayload(symbol=symbol, horizon="1d", amount=1.0, indicators=["sma","ema","rsi"])
    return await _do_predict(payload, x_api_key)

@app.get("/v1/bias")
async def v1_bias(symbol: str = Query("XAUUSD")) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "biases": {"confirmation": 0.3, "recency": 0.4, "overconfidence": 0.2, "loss_aversion": 0.6},
        "summary": "Mild loss-aversion risk; watch for chasing recent moves.",
        "version": APP_VERSION,
    }

@app.get("/v1/bias/influence")
async def v1_bias_influence() -> Dict[str, Any]:
    return {"labels": ["Loss Aversion","Confirmation","Recency","Anchoring","Overconfidence"], "values": [60,30,45,35,25]}

@app.get("/v1/feature-importance")
async def v1_feature_importance() -> Dict[str, Any]:
    return {"features": ["SMA(50)","EMA(21)","RSI(14)","DXY","Real Yields"], "importances": [0.22,0.18,0.14,0.28,0.18]}

@app.get("/v1/alerts")
async def v1_alerts() -> Dict[str, Any]:
    return {"alerts": [
        {"title":"Price Alert","message":"Gold broke $2,040 resistance","severity":"medium","when":"2m ago"},
        {"title":"Volume Spike","message":"Unusual GC volume detected","severity":"low","when":"5m ago"},
        {"title":"Fed Update","message":"FOMC minutes released","severity":"high","when":"15m ago"},
    ]}

@app.get("/v1/trace")
async def v1_trace(id: str = Query(...)) -> Dict[str, Any]:
    return {"id": id, "status": "ok", "steps": [
        {"name":"fetch_market","ok":True},{"name":"compute_features","ok":True},{"name":"model_infer","ok":True}
    ]}

# ---------- Compute proxy skeleton ----------
ALLOWED_COMPUTE_HEADS = {"predict","spot","series","summary","structural","midlayer","blended"}

async def _compute_proxy_impl(path: str, request: Request, x_api_key: Optional[str]):
    norm = (path or "").strip().strip("/")
    segs = [s for s in norm.split("/") if s]
    head = segs[0] if segs else ""
    if head not in ALLOWED_COMPUTE_HEADS:
        raise HTTPException(status_code=404, detail=f"Path '{head or path}' not allowed")
    if head == "predict":
        if request.method != "POST": raise HTTPException(status_code=405, detail="Use POST")
        data = await request.json()
        return await _do_predict(PredictPayload(**data), x_api_key)
    # Stubs for now
    return {"status":"ok","head":head,"now":_utc_now(),"note":"stub"}

@app.post("/compute/{path:path}")
async def compute_proxy(path: str, request: Request, x_api_key: Optional[str] = Header(default=None)):
    return await _compute_proxy_impl(path, request, x_api_key)

@app.post("/api/compute/{path:path}", include_in_schema=False)
async def compute_proxy_api_alias(path: str, request: Request, x_api_key: Optional[str] = Header(default=None)):
    return await _compute_proxy_impl(path, request, x_api_key)


# =======================
# NEW DASHBOARD ENDPOINTS (with yfinance + Twelve Data + fallbacks)
# =======================
import os, requests
from fastapi import HTTPException
import yfinance as yf
import pandas_ta as ta
import pandas as pd
from datetime import datetime

ALPHA_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
TWELVE_KEY = os.getenv("TWELVE_DATA_API_KEY")
METALPRICE_KEY = os.getenv("METALPRICE_API_KEY")
COMPUTE_URL = os.getenv("COMPUTE_URL", "http://localhost:8001")
INTERNAL_SHARED_SECRET = os.getenv("INTERNAL_SHARED_SECRET")

# ---- Spot Price ----
@app.get("/market/gold/spot", tags=["market"])
async def get_spot():
    try:
        ticker = yf.Ticker("GC=F")
        data = ticker.history(period="1d", interval="5m")
        return {
            "symbol": "GC=F",
            "last_price": float(data["Close"].iloc[-1]),
            "timestamp": str(data.index[-1])
        }
    except Exception:
        try:
            url = f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={TWELVE_KEY}"
            r = requests.get(url).json()
            return {"symbol": "XAUUSD", "last_price": float(r.get("price")), "timestamp": datetime.utcnow().isoformat()}
        except Exception:
            try:
                url = f"https://metalpriceapi.com/api/latest?api_key={METALPRICE_KEY}&base=USD&currencies=XAU"
                r = requests.get(url).json()
                return {"symbol": "XAU", "last_price": r["rates"]["XAU"], "timestamp": datetime.utcnow().isoformat()}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Spot fetch failed: {str(e)}")

# ---- Futures ----
@app.get("/market/gold/futures", tags=["market"])
async def get_futures():
    try:
        ticker = yf.Ticker("GC=F")
        data = ticker.history(period="1d", interval="1d")
        return {
            "symbol": "GC=F",
            "last_price": float(data["Close"].iloc[-1]),
            "timestamp": str(data.index[-1])
        }
    except Exception:
        try:
            url = f"https://metalpriceapi.com/api/latest?api_key={METALPRICE_KEY}&base=USD&currencies=XAU"
            r = requests.get(url).json()
            return {"symbol": "GC=F", "last_price": r["rates"]["XAU"], "timestamp": datetime.utcnow().isoformat()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Futures fetch failed: {str(e)}")

# ---- ETF ----
@app.get("/market/gold/etf", tags=["market"])
async def get_etf():
    try:
        ticker = yf.Ticker("GLD")
        data = ticker.history(period="1d", interval="1d")
        return {
            "symbol": "GLD",
            "last_price": float(data["Close"].iloc[-1]),
            "timestamp": str(data.index[-1])
        }
    except Exception:
        try:
            url = f"https://api.twelvedata.com/price?symbol=GLD&apikey={TWELVE_KEY}"
            r = requests.get(url).json()
            return {"symbol": "GLD", "last_price": float(r.get("price")), "timestamp": datetime.utcnow().isoformat()}
        except Exception:
            try:
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=GLD&apikey={ALPHA_KEY}"
                r = requests.get(url).json()
                last = list(r["Time Series (Daily)"].items())[0]
                return {"symbol": "GLD", "last_price": float(last[1]["4. close"]), "timestamp": last[0]}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"ETF fetch failed: {str(e)}")

# ---- Options (ETF options from GLD) ----
@app.get("/market/gold/options", tags=["market"])
async def get_options():
    try:
        ticker = yf.Ticker("GLD")
        exp = ticker.options[0]
        chain = ticker.option_chain(exp)
        return {
            "symbol": "GLD",
            "expiration": exp,
            "calls": chain.calls.head(5).to_dict(),
            "puts": chain.puts.head(5).to_dict()
        }
    except Exception:
        return {"message": "Options data not available on free sources"}

# ---- Indicators ----
@app.get("/v1/indicators", tags=["indicators"])
async def get_indicators():
    try:
        ticker = yf.Ticker("GC=F")
        df = ticker.history(period="6mo", interval="1d")
        df["SMA"] = ta.sma(df["Close"], length=20)
        df["EMA"] = ta.ema(df["Close"], length=20)
        df["RSI"] = ta.rsi(df["Close"], length=14)
        macd = ta.macd(df["Close"])
        df["MACD"] = macd.iloc[:,0]
        bb = ta.bbands(df["Close"])
        df["Bollinger_Upper"] = bb.iloc[:,0]
        df["Bollinger_Lower"] = bb.iloc[:,2]
        df["Stochastic"] = ta.stoch(df["High"], df["Low"], df["Close"])
        df["DXY"] = 104
        df["Real_Yields"] = 1.9
        df["COT"] = 120000
        df["Miners"] = 28.5
        df["VIX"] = 15.2
        df["Central_Bank"] = 9200
        latest = df.tail(50)
        return latest.to_dict(orient="list")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indicator fetch failed: {str(e)}")

# ---- Decision Engine ----
@app.get("/v1/decision", tags=["decision"])
async def get_decision(symbol: str = "XAUUSD"):
    try:
        r = requests.post(
            f"{COMPUTE_URL}/predict",
            json={"symbol": symbol},
            headers={"X-Internal-Secret": INTERNAL_SHARED_SECRET}
        )
        pred = r.json()
        risk_score = float(pred.get("confidence", 0.5))
        allocation_pct = round(risk_score * 100 * 0.5, 2)
        return {
            "timestamps": [datetime.utcnow().isoformat()],
            "risk_score": [risk_score],
            "allocation_pct": [allocation_pct]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decision fetch failed: {str(e)}")

# =======================
# END DASHBOARD ENDPOINTS
# =======================
