#!/usr/bin/env python3
import asyncio, os, json, time, re
from typing import Dict, Any, Optional, Tuple, List
import httpx

CACHE_PATH = ".probe_cache.json"
CACHE_TTL  = 60  # seconds

# ---------- helpers ----------
def env_str(name: str, default: Optional[str]=None) -> Optional[str]:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v.strip()

def load_cache() -> Dict[str, Any]:
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_cache(c: Dict[str, Any]) -> None:
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(c, f, indent=2)
    except Exception:
        pass

def get_cached(key: str) -> Optional[Any]:
    c = load_cache()
    rec = c.get(key)
    if not rec: return None
    if time.time() - rec.get("ts", 0) > CACHE_TTL:
        return None
    return rec.get("val")

def set_cached(key: str, val: Any) -> None:
    c = load_cache()
    c[key] = {"ts": time.time(), "val": val}
    save_cache(c)

def _fmt_price(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:,.2f}"

async def _get_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any]=None, headers: Dict[str, str]=None) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    try:
        r = await client.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json(), None
    except httpx.HTTPStatusError as e:
        try:
            data = e.response.json()
        except Exception:
            data = {"error": str(e)}
        return None, f"HTTP {e.response.status_code}: {data}"
    except Exception as e:
        return None, str(e)

# ---------- price providers ----------
async def test_yfinance(client: httpx.AsyncClient) -> Tuple[str, str, Optional[float], Optional[str]]:
    """GLD ETF as proxy (intraday) -> fallback to last daily close."""
    try:
        import yfinance as yf  # requires yfinance in requirements
    except Exception:
        return ("yfinance", "FAIL", None, "yfinance not installed")

    # Try GLD intraday (fast)
    try:
        t = yf.Ticker("GLD")
        # live price (fast attr) may be None off-hours; then fallback to last close
        price = None
        try:
            info = t.fast_info
            price = float(info.get("lastPrice")) if info and info.get("lastPrice") else None
        except Exception:
            pass
        if price is None:
            hist = t.history(period="5d", interval="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        if price:
            return ("yfinance", "PASS", price, "GLD")
        return ("yfinance", "FAIL", None, "No data (GLD)")
    except Exception as e:
        return ("yfinance", "FAIL", None, str(e))

async def test_twelvedata(client: httpx.AsyncClient) -> Tuple[str, str, Optional[float], Optional[str]]:
    key = env_str("TWELVE_DATA_API_KEY")
    if not key:
        return ("twelvedata", "SKIP", None, "TWELVE_DATA_API_KEY not set")
    # use XAU/USD spot if your plan supports it; otherwise GLD
    # Twelve Data metals are often premium; GLD is safer for free tier
    url = "https://api.twelvedata.com/quote"
    # try GLD first
    data, err = await _get_json(client, url, params={"symbol":"GLD", "apikey": key})
    if err:
        return ("twelvedata", "FAIL", None, err)
    price = None
    try:
        price = float(data.get("price")) if data and data.get("price") else None
    except Exception:
        pass
    return ("twelvedata", "PASS" if price else "FAIL", price, None if price else str(data))

async def test_alpha_vantage(client: httpx.AsyncClient) -> Tuple[str, str, Optional[float], Optional[str]]:
    key = env_str("ALPHA_VANTAGE_API_KEY")
    if not key:
        return ("alphavantage", "SKIP", None, "ALPHA_VANTAGE_API_KEY not set")
    # Free tier: GLOBAL_QUOTE works for equities/ETFs; use GLD
    url = "https://www.alphavantage.co/query"
    params = {"function":"GLOBAL_QUOTE","symbol":"GLD","apikey":key}
    data, err = await _get_json(client, url, params=params)
    if err:
        return ("alphavantage","FAIL",None,err)
    q = (data or {}).get("Global Quote", {})
    px = q.get("05. price") or q.get("price") or q.get("02. open")
    try:
        price = float(px) if px else None
    except Exception:
        price = None
    note = None if price else json.dumps(data)[:180]
    return ("alphavantage", "PASS" if price else "FAIL", price, note)

async def test_metalprice(client: httpx.AsyncClient) -> Tuple[str, str, Optional[float], Optional[str]]:
    key = env_str("METALPRICE_API_KEY")
    if not key:
        return ("metalprice","SKIP",None,"METALPRICE_API_KEY not set")
    # https://api.metalpriceapi.com/v1/latest?api_key=...&base=USD&currencies=XAU
    url = "https://api.metalpriceapi.com/v1/latest"
    data, err = await _get_json(client, url, params={"api_key":key,"base":"USD","currencies":"XAU"})
    if err:
        return ("metalprice","FAIL",None,err)
    try:
        # XAU per USD -> invert to USD per XAU (troy ounce)
        xau_per_usd = float((data.get("rates") or {}).get("XAU"))
        price = 1.0 / xau_per_usd if xau_per_usd else None
    except Exception:
        price = None
    return ("metalprice","PASS" if price else "FAIL",price,None if price else json.dumps(data)[:180])

async def test_goldpricez(client: httpx.AsyncClient):
    key = os.getenv("GOLDPRICEZ_API_KEY")
    if not key:
        return ("goldpricez", "SKIP", None, "API key not set")
    try:
        url = f"https://goldpricez.com/api/rates/currency/usd?api_key={key}"
        r = await client.get(url, timeout=10.0)
        data = r.json()
        if r.status_code == 200 and "gold" in data:
            price = float(data["gold"]["price"])
            return ("goldpricez", "PASS", price, None)
        else:
            return ("goldpricez", "FAIL", None, data.get("error", "unexpected response"))
    except Exception as e:
        return ("goldpricez", "FAIL", None, str(e))

# ---------- FRED (macro insight) ----------
FRED_SERIES = [
    ("CPIAUCSL",  "CPI (Index)"),
    ("DGS10",     "10Y Treasury Yield (%)"),
    ("T10YIE",    "10Y Breakeven Inflation (%)"),
    ("DTWEXBGS",  "Broad Dollar Index"),
]

async def _fred_latest(client: httpx.AsyncClient, api_key: str, series_id: str) -> Tuple[Optional[float], Optional[str]]:
    """Fetch latest observation value for a FRED series."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1
    }
    data, err = await _get_json(client, url, params=params)
    if err:
        return None, err
    try:
        obs = (data or {}).get("observations") or []
        if not obs:
            return None, "No observations"
        val_str = obs[0].get("value")
        # FRED sometimes returns "." for missing
        if val_str in (None, ".", ""):
            return None, "Missing value"
        return float(val_str), None
    except Exception as e:
        return None, str(e)

async def test_fred_macro(client: httpx.AsyncClient) -> Tuple[str, str, Optional[float], Optional[str]]:
    """Treat FRED as macro context, not a price feed.
       PASS if at least one reference series returns a valid numeric value.
       Note returns a JSON string of fetched macro metrics.
    """
    api_key = env_str("FRED_API_KEY")
    if not api_key:
        return ("fred","SKIP",None,"FRED_API_KEY not set")

    # 60s cache
    cached = get_cached("fred_macro")
    if cached:
        # No single "price" for FRED; return None and put all details in note
        return ("fred","PASS",None,json.dumps(cached))

    results: Dict[str, Any] = {}
    ok_count = 0
    for sid, label in FRED_SERIES:
        val, err = await _fred_latest(client, api_key, sid)
        if val is not None:
            results[sid] = {"label": label, "value": val}
            ok_count += 1
        else:
            results[sid] = {"label": label, "error": err}

    if ok_count > 0:
        set_cached("fred_macro", results)
        return ("fred","PASS",None,json.dumps(results))
    else:
        # return FAIL with details
        return ("fred","FAIL",None,json.dumps(results))

# ---------- Orchestrator ----------
async def run_all(as_json: bool=False) -> Tuple[bool, List[Tuple[str,str,Optional[float],Optional[str]]]]:
    cache_hit = get_cached("probe_all")
    if cache_hit:
        return True, cache_hit

    async with httpx.AsyncClient() as client:
        tests = [
            test_yfinance(client),
            test_twelvedata(client),
            test_alpha_vantage(client),
            test_metalprice(client),
            test_goldpricez(client),
            test_fred_macro(client),  # now macro insight, not price
        ]
        results = await asyncio.gather(*tests)

    ok = all(s in ("PASS","SKIP") for _, s, _, _ in results)
    # cache whole bundle for TTL
    set_cached("probe_all", results)
    return ok, results

def print_table(results: List[Tuple[str,str,Optional[float],Optional[str]]]) -> None:
    print("\nProvider Probe Results")
    print("----------------------")
    for name, status, price, note in results:
        shown = _fmt_price(price)
        print(f"{name:<12}: {status:<4}  price: {shown}", end="")
        if note:
            # keep notes short in table view
            short = note if len(note) < 120 else note[:117] + "..."
            print(f"  ({short})")
        else:
            print()
    print("\nLegend: PASS=works, FAIL=error, SKIP=key not set")

async def main() -> int:
    as_json = ("--json" in os.sys.argv)
    ok, results = await run_all(as_json=as_json)
    if as_json:
        # Make FRED note a JSON object when possible
        massaged = []
        for name, status, price, note in results:
            obj = None
            if note:
                try:
                    obj = json.loads(note)
                except Exception:
                    obj = note
            massaged.append({"provider":name,"status":status,"price":price,"note":obj})
        print(json.dumps({"ok":ok,"results":massaged}, indent=2))
    else:
        # print any top-level Yahoo symbol warnings detected earlier in your runs
        print_table(results)
    return 0 if ok else 2

if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
