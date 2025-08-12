# providers.py — adapters for external data (minimal, no heavy deps)

from __future__ import annotations
import os, time, json
from typing import Dict, Any, List, Tuple
import requests

ProviderResp = Dict[str, Any]

def _http_get(url: str, params: Dict[str,Any], timeout=12) -> Any:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    ct = r.headers.get("content-type","")
    return r.json() if "application/json" in ct else r.text

def _to_series(points: List[Tuple[str, float]]) -> ProviderResp:
    # Normalize & sort by time
    pts = sorted(points, key=lambda x: x[0])
    return {"labels":[p[0] for p in pts], "values":[float(p[1]) for p in pts]}

# ---------- Twelve Data ----------
def twelve_data_series(symbol_map: Dict[str,str], interval="1h", rng="1month") -> ProviderResp:
    api_key = os.getenv("TWELVE_DATA_KEY") or ""
    sym = symbol_map.get("twelve_data","XAU/USD")
    if not api_key: raise RuntimeError("TWELVE_DATA_KEY missing")
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": sym, "interval": interval, "outputsize": 5000, "apikey": api_key, "format":"JSON", "timezone":"UTC", "country":"US", "range": rng}
    data = _http_get(url, params)
    if "values" not in data: raise RuntimeError(f"twelve data error: {data}")
    pts = [(row["datetime"], float(row["close"])) for row in data["values"]]
    return _to_series(pts)

# ---------- Alpha Vantage ----------
def alpha_v_series(symbol_map: Dict[str,str], interval="60min", rng="1month") -> ProviderResp:
    api_key = os.getenv("ALPHA_VANTAGE_KEY") or ""
    sym = symbol_map.get("alpha_vantage","XAUUSD")
    if not api_key: raise RuntimeError("ALPHA_VANTAGE_KEY missing")
    # Use FX_INTRADAY for XAUUSD
    url = "https://www.alphavantage.co/query"
    params = {"function":"FX_INTRADAY", "from_symbol":"XAU", "to_symbol":"USD", "interval":interval, "apikey": api_key, "outputsize":"full"}
    data = _http_get(url, params)
    key = f"Time Series FX ({interval})"
    if key not in data: raise RuntimeError(f"alpha vantage error: {data.get('Note') or list(data.keys())}")
    # data[key] -> dict of "YYYY-MM-DD HH:MM:SS": {"1. open":.. "4. close":..}
    pts = [(k, float(v["4. close"])) for k,v in data[key].items()]
    return _to_series(pts)

# ---------- Metalprice (spot) ----------
def metalprice_spot(symbol_map: Dict[str,str]) -> Dict[str,Any]:
    # Minimal spot fetch (adjust to your plan)
    api_key = os.getenv("METALPRICE_KEY") or ""
    if not api_key: raise RuntimeError("METALPRICE_KEY missing")
    url = "https://api.metalpriceapi.com/v1/latest"
    params = {"api_key": api_key, "base":"XAU", "currencies":"USD"}
    data = _http_get(url, params)
    price = float(data.get("rates",{}).get("USD", 0.0))
    ts = int(data.get("timestamp", int(time.time())))
    iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
    return {"labels":[iso], "values":[price]}

# ---------- FRED (macro) ----------
def fred_series(series_id: str, limit=1000) -> ProviderResp:
    api_key = os.getenv("FRED_API_KEY") or ""
    if not api_key: raise RuntimeError("FRED_API_KEY missing")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"api_key": api_key, "series_id": series_id, "file_type":"json", "sort_order":"asc", "limit": limit}
    data = _http_get(url, params)
    obs = data.get("observations", [])
    pts = [(o["date"]+"T00:00:00Z", float(o["value"])) for o in obs if o.get("value") not in (".","NaN","")]
    return _to_series(pts)

# ---------- yfinance fallback ----------
def yfinance_series(symbol_map: Dict[str,str], interval="1d", period="6mo") -> ProviderResp:
    try:
        import yfinance as yf
        sym = symbol_map.get("yfinance","GC=F")
        hist = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=False)
        if hist is None or hist.empty:
            return {"labels": [], "values": []}
        labels = [d.isoformat() for d in hist.index]
        vals = [float(x) for x in hist["Close"].tolist()]
        return {"labels": labels, "values": vals}
    except Exception:
        return {"labels": [], "values": []}
