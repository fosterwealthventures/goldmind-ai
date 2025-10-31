# compute/server.py
import os
import time
import copy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

load_dotenv()
import numpy as np
from shared import aggregator

APP_NAME = "goldmind-compute"
APP_VERSION = os.getenv("APP_VERSION", "v1.2.1-insights")
REVISION = os.getenv("K_REVISION") or os.getenv("REVISION_ID") or "dev"
INTERNAL_SHARED_SECRET = os.getenv("INTERNAL_SHARED_SECRET", "")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/production_goldmind_v1.h5")
MODEL_SHA256 = os.getenv("MODEL_SHA256", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
TWELVE_KEY = os.getenv("TWELVE_DATA_API_KEY")
METALPRICE_KEY = os.getenv("METALPRICE_API_KEY")
GOLDPRICEZ_KEY = os.getenv("GOLDPRICEZ_API_KEY")
GOLDPRICEZ_BASE = os.getenv("GOLDPRICEZ_BASE", "https://www.goldapi.io/api")
INSIGHTS_CACHE_TTL = int(os.getenv("INSIGHTS_CACHE_TTL", "90"))
_INSIGHTS_CACHE: Dict[Tuple[str, str, str], Tuple[datetime, Dict[str, Any]]] = {}

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
    model_config = ConfigDict(protected_namespaces=())
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


def _safe_last(series: pd.Series) -> float:
    if series.empty:
        raise ValueError("Series empty")
    return float(series.iloc[-1])


def _safe_index_last(index: pd.Index) -> str:
    if index.empty:
        raise ValueError("Index empty")
    return str(index[-1])


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(length).mean()
    roll_down = pd.Series(down, index=series.index).rolling(length).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def _bollinger(series: pd.Series, length: int, stdev: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(length).mean()
    sd = series.rolling(length).std()
    upper = mid + stdev * sd
    lower = mid - stdev * sd
    return upper, mid, lower


def _stoch(df: pd.DataFrame, k_period: int, d_period: int) -> tuple[pd.Series, pd.Series]:
    low_k = df["Low"].rolling(k_period).min()
    high_k = df["High"].rolling(k_period).max()
    percent_k = 100 * (df["Close"] - low_k) / (high_k - low_k)
    percent_d = percent_k.rolling(d_period).mean()
    return percent_k, percent_d


def _mock_history(days: int = 120) -> pd.DataFrame:
    idx = pd.date_range(end=datetime.utcnow(), periods=days, freq="D")
    values = np.arange(days, dtype=float)
    close = 1900.0 + 0.6 * values + 12.0 * np.sin(values / 6.0)
    open_ = close + 1.5 * np.sin(values / 5.0)
    high = close + 5.0
    low = close - 5.0
    data = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
        },
        index=idx,
    )
    return data


def _predict_response(body: PredictBody) -> PredictResult:
    if not MODEL.loaded:
        MODEL.load()

    model_out = MODEL.predict(body.dict())
    requested = body.indicators or []
    indicators: Dict[str, Any] = {}
    if requested:
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


def _indicator_highlights(indicators: Dict[str, Any]) -> List[str]:
    highlights: List[str] = []
    rsi = indicators.get("RSI")
    if rsi:
        val = rsi.get("value")
        signal = rsi.get("signal")
        if val is not None and signal:
            highlights.append(f"RSI at {float(val):.1f} ({signal})")
    macd = indicators.get("MACD")
    if macd:
        hist = macd.get("hist")
        hint = macd.get("signal_hint")
        if hist is not None and hint:
            highlights.append(f"MACD histogram {float(hist):+.3f} ({hint})")
    sma = indicators.get("SMA")
    if sma:
        sig = sma.get("signal")
        if sig:
            highlights.append(f"SMA trend looks {sig}")
    ema = indicators.get("EMA")
    if ema:
        sig = ema.get("signal")
        if sig:
            highlights.append(f"EMA trend looks {sig}")
    dxy = indicators.get("DXY_CORR")
    if dxy:
        rho = dxy.get("rho")
        signal = dxy.get("signal")
        if rho is not None and signal:
            highlights.append(f"DXY correlation at {float(rho):+.2f} ({signal})")
    real_yields = indicators.get("REAL_YIELDS")
    if real_yields:
        tips = real_yields.get("tips10")
        signal = real_yields.get("signal")
        if tips is not None and signal:
            highlights.append(f"Real yields near {float(tips):.2f}% ({signal})")
    vix = indicators.get("VIX_CORR")
    if vix and vix.get("signal"):
        highlights.append("VIX correlation appears normal")
    cb = indicators.get("CB_POLICY")
    if cb and cb.get("stance"):
        highlights.append(f"Central bank stance: {cb['stance']}")
    return highlights[:5]


async def _compute_insights(symbol: str, timeframe: str, range_: str) -> Dict[str, Any]:
    catalog = aggregator.list_indicators()
    ids = [item["id"] for item in catalog]
    params = {item["id"]: item.get("params", {}) for item in catalog}
    upstream_symbol = symbol
    if "/" not in symbol:
        sym_upper = symbol.upper()
        if sym_upper in {"XAUUSD", "XAU"}:
            upstream_symbol = "XAU/USD"
    return await aggregator.run_insights(upstream_symbol, timeframe, range_, ids, params)


def _coerce_predict_body(payload: Dict[str, Any]) -> PredictBody:
    symbol = str(payload.get("symbol") or "XAUUSD")
    horizon = str(payload.get("horizon") or payload.get("timeframe") or "1d")
    raw_amount = payload.get("amount")
    try:
        amount = float(raw_amount) if raw_amount is not None else 1.0
    except (TypeError, ValueError):
        amount = 1.0
    style = str(payload.get("style") or payload.get("view") or "day")
    interval = str(payload.get("interval") or "1d")
    indicators = payload.get("indicators")
    options = payload.get("options")
    return PredictBody(
        symbol=symbol,
        horizon=horizon,
        amount=amount,
        style=style,
        interval=interval,
        indicators=indicators,
        options=options,
    )


def _twelvedata_price(symbol: str) -> Optional[Dict[str, Any]]:
    if not TWELVE_KEY:
        return None
    url = "https://api.twelvedata.com/price"
    params = {"symbol": symbol, "apikey": TWELVE_KEY}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    price = data.get("price")
    if price is None:
        raise ValueError(f"TwelveData missing price: {data}")
    return {
        "symbol": symbol.replace("/", ""),
        "last_price": float(price),
        "timestamp": datetime.utcnow().isoformat(),
        "source": "TwelveData",
    }


def _metalprice_price(symbol: str = "XAU") -> Optional[Dict[str, Any]]:
    if not METALPRICE_KEY:
        return None
    url = "https://metalpriceapi.com/api/latest"
    params = {"api_key": METALPRICE_KEY, "base": "USD", "currencies": symbol}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    rate = data.get("rates", {}).get(symbol)
    if rate is None:
        raise ValueError(f"Metalprice missing rate: {data}")
    return {
        "symbol": symbol,
        "last_price": float(rate),
        "timestamp": datetime.utcnow().isoformat(),
        "source": "Metalprice API",
    }


def _goldpricez_price() -> Optional[Dict[str, Any]]:
    if not GOLDPRICEZ_KEY:
        return None
    url = f"{GOLDPRICEZ_BASE}/XAU/USD"
    headers = {"x-access-token": GOLDPRICEZ_KEY, "Content-Type": "application/json"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    js = resp.json()
    price = js.get("price") or js.get("price_gram_24k") or js.get("ask")
    ts = js.get("timestamp") or js.get("time")
    if price is None:
        raise ValueError(f"GoldPriceZ missing price: {js}")
    if isinstance(ts, (int, float)):
        timestamp = datetime.utcfromtimestamp(ts).isoformat()
    else:
        timestamp = datetime.utcnow().isoformat()
    return {
        "symbol": "XAUUSD",
        "last_price": float(price),
        "timestamp": timestamp,
        "source": "GoldPriceZ",
    }


def _alphavantage_price(symbol: str) -> Optional[Dict[str, Any]]:
    if not ALPHA_KEY:
        return None
    url = "https://www.alphavantage.co/query"
    params = {"function": "TIME_SERIES_DAILY", "symbol": symbol, "apikey": ALPHA_KEY}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    series = data.get("Time Series (Daily)")
    if not series:
        raise ValueError(f"AlphaVantage missing series: {data}")
    last_date, last_row = next(iter(series.items()))
    close_price = last_row.get("4. close")
    if close_price is None:
        raise ValueError(f"AlphaVantage missing close: {last_row}")
    return {"symbol": symbol, "last_price": float(close_price), "timestamp": last_date}


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

    return _predict_response(body)


@app.post("/v1/predict", response_model=PredictResult, tags=["predict"])
def public_predict_post(payload: Dict[str, Any]) -> PredictResult:
    body = _coerce_predict_body(payload or {})
    return _predict_response(body)


@app.get("/v1/predict", response_model=PredictResult, tags=["predict"])
def public_predict_get(
    symbol: str = "XAUUSD",
    timeframe: str = "1d",
    style: str = "day",
) -> PredictResult:
    body = PredictBody(
        symbol=symbol,
        horizon=timeframe,
        amount=1.0,
        style=style,
        interval="1d",
    )
    return _predict_response(body)


@app.get("/v1/bias", tags=["bias"])
def public_bias(symbol: str = "XAU", timeframe: str = "1d") -> Dict[str, Any]:
    biases = {
        "confirmation": 0.3,
        "recency": 0.4,
        "overconfidence": 0.2,
        "loss_aversion": 0.6,
    }
    summary = "Mild loss-aversion risk; watch for chasing recent moves."
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "biases": biases,
        "summary": summary,
        "version": APP_VERSION,
        "source": "compute",
    }


@app.get("/insights/narrative", tags=["insights"])
async def insights_narrative(
    symbol: str = "XAUUSD",
    timeframe: str = "1d",
    range_: str = "6m",
) -> Dict[str, Any]:
    key = (symbol, timeframe, range_)
    now = datetime.utcnow()
    cached = _INSIGHTS_CACHE.get(key)
    if cached:
        ts, payload = cached
        if (now - ts).total_seconds() < INSIGHTS_CACHE_TTL:
            return copy.deepcopy(payload)

    try:
        insights = await _compute_insights(symbol, timeframe, range_)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Insights run failed: {exc}") from exc

    summary = insights.get("summary", {})
    narrative_parts: List[str] = []
    technical = summary.get("technical")
    if technical:
        narrative_parts.append(f"Technical tone is {technical}.")
    macro = summary.get("macro")
    if macro:
        narrative_parts.append(f"Macro backdrop reads {macro}.")
    risk = summary.get("risk")
    if risk:
        narrative_parts.append(f"Risk posture is {risk}.")
    narrative = " ".join(narrative_parts) or "Insights generated."

    indicators = insights.get("indicators", {})
    highlights = _indicator_highlights(indicators)
    errors = insights.get("errors", [])
    if errors and not highlights:
        highlights = [f"Data provider issue at {err.get('stage')}: {err.get('detail')}" for err in errors][:3]
        if narrative == "Insights generated.":
            narrative = "Live market data is temporarily unavailable; showing provider diagnostics."

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "range": range_,
        "generated_at": insights.get("asof"),
        "narrative": narrative,
        "summary": summary,
        "highlights": highlights,
        "indicators": indicators,
        "sources": insights.get("sources", []),
        "errors": errors,
    }

    _INSIGHTS_CACHE[key] = (now, copy.deepcopy(result))
    return result


@app.get("/market/gold/spot", tags=["market"])
def market_gold_spot(x_internal_secret: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _require_internal_secret(x_internal_secret)
    try:
        ticker = yf.Ticker("GC=F")
        data = ticker.history(period="1d", interval="5m")
        if data.empty:
            raise ValueError("no data from yfinance")
        return {
            "symbol": "GC=F",
            "last_price": _safe_last(data["Close"]),
            "timestamp": _safe_index_last(data.index),
        }
    except Exception:
        for fallback in (
            lambda: _twelvedata_price("XAU/USD"),
            _metalprice_price,
            _goldpricez_price,
        ):
            try:
                out = fallback()
                if out:
                    return out
            except Exception:
                continue
        df = _mock_history(30)
        return {
            "symbol": "XAUUSD",
            "last_price": float(df["Close"].iloc[-1]),
            "timestamp": df.index[-1].isoformat(),
            "note": "mock-fallback",
            "source": "mock",
        }


@app.get("/market/gold/futures", tags=["market"])
def market_gold_futures(x_internal_secret: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _require_internal_secret(x_internal_secret)
    try:
        ticker = yf.Ticker("GC=F")
        data = ticker.history(period="1d", interval="1d")
        if data.empty:
            raise ValueError("no data from yfinance")
        return {
            "symbol": "GC=F",
            "last_price": _safe_last(data["Close"]),
            "timestamp": _safe_index_last(data.index),
        }
    except Exception:
        out = None
        try:
            out = _metalprice_price("XAU")
        except Exception:
            out = None
        if not out:
            try:
                out = _goldpricez_price()
            except Exception:
                out = None
        if out:
            out["symbol"] = "GC=F"
            return out
        df = _mock_history(60)
        return {
            "symbol": "GC=F",
            "last_price": float(df["Close"].iloc[-1]),
            "timestamp": df.index[-1].isoformat(),
            "note": "mock-fallback",
            "source": "mock",
        }


@app.get("/market/gold/etf", tags=["market"])
def market_gold_etf(x_internal_secret: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _require_internal_secret(x_internal_secret)
    try:
        ticker = yf.Ticker("GLD")
        data = ticker.history(period="1d", interval="1d")
        if data.empty:
            raise ValueError("no data from yfinance")
        return {
            "symbol": "GLD",
            "last_price": _safe_last(data["Close"]),
            "timestamp": _safe_index_last(data.index),
        }
    except Exception:
        for fallback in (
            lambda: _twelvedata_price("GLD"),
            lambda: _alphavantage_price("GLD"),
            _goldpricez_price,
        ):
            try:
                out = fallback()
                if out:
                    return out
            except Exception:
                continue
        df = _mock_history(60)
        return {
            "symbol": "GLD",
            "last_price": float(df["Close"].iloc[-1]),
            "timestamp": df.index[-1].isoformat(),
            "note": "mock-fallback",
            "source": "mock",
        }


@app.get("/market/gold/options", tags=["market"])
def market_gold_options(x_internal_secret: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _require_internal_secret(x_internal_secret)
    try:
        ticker = yf.Ticker("GLD")
        expirations = ticker.options
        if not expirations:
            raise ValueError("no option expirations available")
        exp = expirations[0]
        chain = ticker.option_chain(exp)
        return {
            "symbol": "GLD",
            "expiration": exp,
            "calls": chain.calls.head(5).to_dict(),
            "puts": chain.puts.head(5).to_dict(),
        }
    except Exception:
        return {"message": "Options data not available on free sources"}


@app.get("/v1/indicators", tags=["indicators"])
def indicators(x_internal_secret: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _require_internal_secret(x_internal_secret)
    try:
        ticker = yf.Ticker("GC=F")
        df = ticker.history(period="6mo", interval="1d")
        if df.empty:
            raise ValueError("no data from yfinance")
        df["SMA"] = _sma(df["Close"], 20)
        df["EMA"] = _ema(df["Close"], 20)
        df["RSI"] = _rsi(df["Close"], 14)
        macd_line, sig_line, _ = _macd(df["Close"], 12, 26, 9)
        df["MACD"] = macd_line
        df["MACD_Signal"] = sig_line
        upper, mid, lower = _bollinger(df["Close"], 20, 2)
        df["Bollinger_Upper"] = upper
        df["Bollinger_Middle"] = mid
        df["Bollinger_Lower"] = lower
        percent_k, percent_d = _stoch(df, 14, 3)
        df["Stochastic_K"] = percent_k
        df["Stochastic_D"] = percent_d
        df["DXY"] = 104
        df["Real_Yields"] = 1.9
        df["COT"] = 120000
        df["Miners"] = 28.5
        df["VIX"] = 15.2
        df["Central_Bank"] = 9200
        latest = df.tail(50)
        return latest.to_dict(orient="list")
    except HTTPException:
        raise
    except Exception as exc:
        df = _mock_history(120)
        df["SMA"] = _sma(df["Close"], 20)
        df["EMA"] = _ema(df["Close"], 20)
        df["RSI"] = _rsi(df["Close"], 14)
        macd_line, sig_line, _ = _macd(df["Close"], 12, 26, 9)
        df["MACD"] = macd_line
        df["MACD_Signal"] = sig_line
        upper, mid, lower = _bollinger(df["Close"], 20, 2)
        df["Bollinger_Upper"] = upper
        df["Bollinger_Middle"] = mid
        df["Bollinger_Lower"] = lower
        percent_k, percent_d = _stoch(df, 14, 3)
        df["Stochastic_K"] = percent_k
        df["Stochastic_D"] = percent_d
        df["DXY"] = 104
        df["Real_Yields"] = 1.9
        df["COT"] = 120000
        df["Miners"] = 28.5
        df["VIX"] = 15.2
        df["Central_Bank"] = 9200
        latest = df.tail(50)
        return latest.to_dict(orient="list")


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
