# api/app/midlayer_insight.py
from __future__ import annotations
import os, math, re
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

try:
    import httpx  # async-capable client
except Exception:
    httpx = None  # type: ignore

try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore

_NUMERIC_RE = re.compile(r"(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)")

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _mk_client() -> "httpx.AsyncClient | None":
    if httpx is None:
        return None
    return httpx.AsyncClient(timeout=20.0, follow_redirects=True)  # type: ignore[call-arg]

# ---------- helpers ----------
def _parse_numeric(text: Any) -> float | None:
    if text is None:
        return None
    s = str(text)
    m = _NUMERIC_RE.findall(s)
    if not m:
        return None
    val = m[-1].replace(",", "")
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except Exception:
        return None

def _momentum_20d(values: List[float]) -> float | None:
    if not values or len(values) < 21:
        return None
    try:
        return (values[-1] / values[-21]) - 1.0
    except Exception:
        return None

# ---------- SERIES PROVIDERS ----------
async def fetch_twelvedata_series(symbol: str = "XAU/USD", interval: str = "1day", outputsize: int = 200) -> Tuple[List[str], List[float], str | None]:
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    base = os.getenv("TWELVE_DATA_BASE", "https://api.twelvedata.com")
    if not api_key or httpx is None:
        return [], [], "twelvedata not configured"

    url = f"{base.rstrip('/')}/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": str(outputsize), "apikey": api_key}
    async with _mk_client() as client:
        r = await client.get(url, params=params)
        try:
            data = r.json()
        except Exception:
            return [], [], f"twelvedata non-json (status {r.status_code})"

        vals = data.get("values") or data.get("data") or []
        labels, values = [], []
        for row in reversed(vals):  # chronological
            dt = row.get("datetime") or row.get("date") or row.get("time")
            close = row.get("close") or row.get("price") or row.get("value")
            if dt is None or close is None:
                continue
            try:
                labels.append(str(dt)[:10])
                values.append(float(close))
            except Exception:
                continue

        if values:
            return labels, values, None
        msg = data.get("message") or data.get("status") or f"empty (status {r.status_code})"
        return [], [], f"twelvedata: {msg}"

async def fetch_alphavantage_series(symbol: str, function: str = "TIME_SERIES_DAILY") -> Tuple[List[str], List[float], str | None]:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    base = os.getenv("ALPHA_VANTAGE_BASE", "https://www.alphavantage.co")
    if not api_key or httpx is None:
        return [], [], "alphavantage not configured"

    url = f"{base.rstrip('/')}/query"
    params = {"function": function, "symbol": symbol, "apikey": api_key, "datatype": "json"}
    async with _mk_client() as client:
        r = await client.get(url, params=params)
        try:
            data = r.json()
        except Exception:
            return [], [], f"alphavantage non-json (status {r.status_code})"

        for k, v in data.items():
            if "Time Series" in k and isinstance(v, dict):
                items = sorted(v.items())
                labels = [k for k, _ in items]
                closes: List[float] = []
                for _, row in items:
                    close = row.get("4. close") or row.get("4. Close") or next((row[x] for x in row if "close" in x.lower()), None)
                    try:
                        closes.append(float(close))
                    except Exception:
                        closes.append(math.nan)
                closes = [c for c in closes if not math.isnan(c)]
                if closes:
                    return labels, closes, None

        err = data.get("Error Message") or data.get("Note") or f"empty (status {r.status_code})"
        return [], [], f"alphavantage: {err}"

async def fetch_yahoo_series(symbol: str = "XAUUSD=X", period: str = "6mo", interval: str = "1d") -> Tuple[List[str], List[float], str | None]:
    if yf is None:
        return [], [], "yfinance not installed"
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False, auto_adjust=True)
        if df is not None and not df.empty:
            closes = df["Close"] if "Close" in df else df.iloc[:, 0]
            labels = [idx.strftime("%Y-%m-%d") for idx in closes.index]
            values = [float(v) for v in closes.tolist()]
            return labels, values, None

        t = yf.Ticker(symbol)
        h = t.history(period=period, interval=interval, auto_adjust=True)
        if h is not None and not h.empty:
            closes = h["Close"] if "Close" in h else h.iloc[:, 0]
            labels = [idx.strftime("%Y-%m-%d") for idx in closes.index]
            values = [float(v) for v in closes.tolist()]
            return labels, values, None
        return [], [], "yahoo empty"
    except Exception as e:
        return [], [], f"yahoo: {type(e).__name__}: {e}"

async def get_midlayer_series() -> Dict[str, Any]:
    providers = {"twelvedata": {"ok": False}, "alphavantage": {"ok": False}, "yahoo": {"ok": False}}
    labels: List[str] = []
    values: List[float] = []
    errors: List[str] = []

    l, v, err = await fetch_twelvedata_series()
    if v:
        providers["twelvedata"].update(ok=True, length=len(v))
        labels, values = l, v
    else:
        errors.append(err or "twelvedata unknown")
        providers["twelvedata"].update(ok=False, length=0)

    if not values:
        l, v, err = await fetch_alphavantage_series(symbol=os.getenv("AV_SYMBOL", "GLD"))
        if v:
            providers["alphavantage"].update(ok=True, length=len(v))
            labels, values = l, v
        else:
            errors.append(err or "alphavantage unknown")
            providers["alphavantage"].update(ok=False, length=0)

    if not values:
        l, v, err = await fetch_yahoo_series(symbol=os.getenv("YF_SYMBOL", "XAUUSD=X"))
        if v:
            providers["yahoo"].update(ok=True, length=len(v))
            labels, values = l, v
        else:
            errors.append(err or "yahoo unknown")
            providers["yahoo"].update(ok=False, length=0)

    return {"ok": bool(values), "providers": providers, "labels": labels, "values": values, "as_of": _now_iso(), "errors": errors}

# ---------- SPOT PROVIDERS ----------
async def fetch_metalprice_spot(symbol: str | None = None, quote: str | None = None) -> tuple[float | None, str | None]:
    api_key = os.getenv("METALPRICE_API_KEY")
    base = (os.getenv("METALPRICE_BASE") or "").rstrip("/")
    if httpx is None or not api_key or not base:
        return None, "metalprice not configured"

    sym = (symbol or os.getenv("METALPRICE_SYMBOL") or "XAU").upper()
    q = (quote or os.getenv("METALPRICE_QUOTE") or "USD").upper()

    url_candidates = [base]
    for suffix in ("/latest", "/v1/latest"):
        if not base.endswith(suffix):
            url_candidates.append(f"{base}{suffix}")

    param_candidates = [
        {"api_key": api_key, "base": q, "symbols": sym},
        {"apikey": api_key, "base": q, "symbols": sym},
        {"api_key": api_key, "currency": q, "metal": sym},
        {"apikey": api_key, "currency": q, "metal": sym},
    ]

    async with _mk_client() as client:
        for url in url_candidates:
            for params in param_candidates:
                try:
                    r = await client.get(url, params=params)
                    data = r.json()
                except Exception:
                    continue

                # Common shape
                rates = data.get("rates")
                if isinstance(rates, dict) and sym in rates:
                    try:
                        rate = float(rates[sym])
                        resp_base = (data.get("base") or data.get("currency") or q).upper()
                        price = (1.0 / rate) if (resp_base == q and rate < 10) else rate
                        return price, None
                    except Exception:
                        pass

                # Flat shapes
                for key in (f"{sym}{q}", "price_usd", "price", "rate", "result", "value"):
                    val = _parse_numeric(data.get(key))
                    if val is not None:
                        v = 1.0 / val if val and val < 10 else val
                        return v, None

    return None, "metalprice: no numeric spot found"

async def fetch_goldpricez_spot(currency: str = "USD") -> Tuple[float | None, str | None]:
    api_key = os.getenv("GOLDPRICEZ_API_KEY")
    base = os.getenv("GOLDPRICEZ_BASE", "").rstrip("/")
    if not api_key or not base or httpx is None:
        return None, "goldpricez not configured"

    candidates = [
        (base, {"api_key": api_key, "currency": currency.lower()}),
        (f"{base}/{currency.lower()}", {"api_key": api_key}),
    ]
    async with _mk_client() as client:
        for url, params in candidates:
            r = await client.get(url, params=params)
            try:
                data = r.json()
            except Exception:
                continue
            # Try several fields, but parse robustly
            for key in ("price", "rate", "price_gram_24k", "price_ounce", "price_usd", "usd"):
                val = _parse_numeric(data.get(key))
                if val is not None:
                    return float(val), None
    return None, "goldpricez: no numeric price"

# ---------- SNAPSHOT ----------
async def snapshot() -> Dict[str, Any]:
    series = await get_midlayer_series()

    mp, mp_err = await fetch_metalprice_spot()
    gp, gp_err = await fetch_goldpricez_spot("USD")

    spot_price = mp if mp is not None else gp
    spot_source = "metalprice" if mp is not None else ("goldpricez" if gp is not None else None)
    spot_err = mp_err if mp is not None else (gp_err if gp is not None else "no spot provider configured")

    providers = series["providers"].copy()
    providers["metalprice"] = {"ok": mp is not None, "error": mp_err}
    providers["goldpricez"] = {"ok": gp is not None, "error": gp_err}

    mom20 = _momentum_20d(series.get("values", []))

    return {
        "source": "multi-provider",
        "ok": series["ok"],
        "errors": series["errors"],
        "providers": providers,
        "series": {"labels": series["labels"], "values": series["values"]},
        "spot": {"price_usd": spot_price, "source": spot_source, "error": spot_err},
        "momentum_20d": mom20,
        "as_of": _now_iso(),
    }
