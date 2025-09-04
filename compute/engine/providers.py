"""
providers.py — GoldMIND AI (production-ready)
---------------------------------------------
Thin provider adapters with:
- Clean, normalized outputs for time series: {"labels": [...ISO8601...], "values": [float...]}
- Provider fallback order and graceful degradation
- Retries with exponential backoff + timeouts
- Minimal dependencies (requests, optional yfinance)

Public surface:
    get_series(symbol_map, *, interval="1h", range_hint="1month") -> dict
    get_spot(symbol_map) -> dict
    fred_series(series_id, limit=1000) -> dict

symbol_map example:
    {
      "twelvedata": "XAU/USD",
      "alpha_vantage": "XAUUSD",
      "yfinance": "GC=F"
    }
"""

from __future__ import annotations

import os
import time
import math
import json
from typing import Dict, Any, List, Tuple, Optional, Callable

import requests

ProviderResp = Dict[str, Any]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

_DEFAULT_TIMEOUT = float(os.getenv("PROVIDER_HTTP_TIMEOUT_SEC", "12"))
_MAX_RETRIES = int(os.getenv("PROVIDER_MAX_RETRIES", "2"))
_BACKOFF = float(os.getenv("PROVIDER_BACKOFF_SEC", "0.5"))  # base

def _http_get(url: str, params: Dict[str, Any], timeout: float = _DEFAULT_TIMEOUT) -> Any:
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    ct = (resp.headers.get("content-type") or "").lower()
    if "application/json" in ct or resp.text.strip().startswith("{"):
        return resp.json()
    return resp.text

def _retrying(fn: Callable[[], Any]) -> Any:
    last_err: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            # exponential backoff with jitter
            sleep = _BACKOFF * (2 ** attempt) + (0.1 * (attempt + 1))
            time.sleep(sleep)
    raise last_err if last_err else RuntimeError("unknown provider error")

def _to_series(points: List[Tuple[str, float]]) -> ProviderResp:
    # Normalize & sort by time (ISO8601 strings allowed)
    pts = sorted(((str(t), float(v)) for t, v in points if v is not None), key=lambda x: x[0])
    return {"labels": [p[0] for p in pts], "values": [p[1] for p in pts]}


# ---------------------------------------------------------------------
# Individual providers
# ---------------------------------------------------------------------

def twelve_data_series(symbol_map: Dict[str, str], interval: str = "1h", range_hint: str = "1month") -> ProviderResp:
    api_key = os.getenv("TWELVE_DATA_KEY") or ""
    sym = symbol_map.get("twelvedata", "XAU/USD")
    if not api_key:
        raise RuntimeError("TWELVE_DATA_KEY missing")
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": sym,
        "interval": interval,
        "outputsize": 5000,
        "apikey": api_key,
        "format": "JSON",
        "timezone": "UTC",
        "range": range_hint,
        "order": "ASC",
    }
    data = _retrying(lambda: _http_get(url, params))
    if not isinstance(data, dict) or "values" not in data:
        raise RuntimeError(f"twelve data error: {data}")
    pts = [(row["datetime"], float(row["close"])) for row in data["values"]]
    return _to_series(pts)


def alpha_v_series(symbol_map: Dict[str, str], interval: str = "60min", range_hint: str = "1month") -> ProviderResp:
    api_key = os.getenv("ALPHA_VANTAGE_KEY") or ""
    _ = range_hint  # unused but kept for API symmetry
    if not api_key:
        raise RuntimeError("ALPHA_VANTAGE_KEY missing")

    # Alpha Vantage intraday for XAUUSD via FX_INTRADAY
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_INTRADAY",
        "from_symbol": "XAU",
        "to_symbol": "USD",
        "interval": interval,
        "apikey": api_key,
        "outputsize": "full",
    }
    data = _retrying(lambda: _http_get(url, params))
    key = f"Time Series FX ({interval})"
    if not isinstance(data, dict) or key not in data:
        # surface common throttle note to caller
        note = data.get("Note") if isinstance(data, dict) else None
        raise RuntimeError(f"alpha vantage error: {note or list(data.keys()) if isinstance(data, dict) else data}")
    pts = [(k, float(v["4. close"])) for k, v in data[key].items()]
    return _to_series(pts)


def yfinance_series(symbol_map: Dict[str, str], interval: str = "1d", period: str = "6mo") -> ProviderResp:
    try:
        import yfinance as yf  # lazy import
        sym = symbol_map.get("yfinance", "GC=F")
        def _sync():
            df = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=False)
            if df is None or df.empty:
                return {"labels": [], "values": []}
            labels = [d.isoformat() for d in df.index]
            vals = [float(x) for x in df["Close"].tolist()]
            return {"labels": labels, "values": vals}
        return _retrying(_sync)
    except Exception:
        return {"labels": [], "values": []}


def metalprice_spot(symbol_map: Dict[str, str]) -> Dict[str, Any]:
    api_key = os.getenv("METALPRICE_KEY") or ""
    if not api_key:
        raise RuntimeError("METALPRICE_KEY missing")
    url = "https://api.metalpriceapi.com/v1/latest"
    params = {"api_key": api_key, "base": "XAU", "currencies": "USD"}
    data = _retrying(lambda: _http_get(url, params))
    price = float((data or {}).get("rates", {}).get("USD", 0.0))
    ts = int((data or {}).get("timestamp", int(time.time())))
    iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
    return {"labels": [iso], "values": [price]}


def fred_series(series_id: str, limit: int = 1000) -> ProviderResp:
    api_key = os.getenv("FRED_API_KEY") or ""
    if not api_key:
        raise RuntimeError("FRED_API_KEY missing")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "api_key": api_key,
        "series_id": series_id,
        "file_type": "json",
        "sort_order": "asc",
        "limit": int(limit),
    }
    data = _retrying(lambda: _http_get(url, params))
    obs = (data or {}).get("observations", [])
    pts = [(o["date"] + "T00:00:00Z", float(o["value"])) for o in obs if o.get("value") not in (".", "NaN", "", None)]
    return _to_series(pts)


# ---------------------------------------------------------------------
# Facades with fallback logic
# ---------------------------------------------------------------------

def get_series(symbol_map: Dict[str, str], *, interval: str = "1h", range_hint: str = "1month") -> ProviderResp:
    """
    Try multiple providers in order until one returns data.
    """
    providers: List[Callable[[], ProviderResp]] = [
        lambda: twelve_data_series(symbol_map, interval=interval, range_hint=range_hint),
        lambda: alpha_v_series(symbol_map, interval="60min" if interval.endswith("h") else "1day", range_hint=range_hint),
        lambda: yfinance_series(symbol_map, interval="1d", period="6mo"),
    ]
    errors: List[str] = []
    for p in providers:
        try:
            out = p()
            if out.get("labels"):
                return out
        except Exception as e:
            errors.append(str(e))
            continue
    # If everything failed, return an empty, well-formed structure with error note
    return {"labels": [], "values": [], "error": "; ".join(errors)}

def get_spot(symbol_map: Dict[str, str]) -> ProviderResp:
    """
    Spot price via metalprice → yfinance fallback.
    """
    try:
        return metalprice_spot(symbol_map)
    except Exception as e1:
        fallback = yfinance_series(symbol_map, interval="1d", period="5d")
        if fallback.get("values"):
            # Use last close as spot proxy
            return {"labels": [fallback["labels"][-1]], "values": [fallback["values"][-1]], "proxy": "yfinance/last_close"}
        return {"labels": [], "values": [], "error": str(e1)}
