"""
aggregator.py — GoldMIND Compute Aggregator (updated)
-----------------------------------------------------
Convenience layer to talk to the internal compute service with:
- Safe headers (`X-Internal-Secret`), timeouts, and graceful fallbacks
- In-memory TTL cache with "degraded" mode on compute outages
- Simple wrappers for common calls used by the dashboard/api
- Both sync (requests) and async (httpx) clients

Env:
  COMPUTE_URL               - base URL of compute service (e.g., http://localhost:7001)
  INTERNAL_SHARED_SECRET    - shared secret header value
  COMPUTE_TIMEOUT_SECS      - per-request timeout (default 6.0)
  AGGREGATOR_CACHE_TTL_SECS - cache TTL seconds (default 300)

Usage (sync):
    from aggregator import Aggregator
    agg = Aggregator()
    data, code = agg.predict(symbol="XAUUSD")

Usage (async):
    from aggregator import AsyncAggregator
    agg = AsyncAggregator()
    data, code = await agg.predict(symbol="XAUUSD")
"""

from __future__ import annotations

import os
import json
import time
import threading
from typing import Any, Dict, Optional, Tuple, List

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

COMPUTE_URL = (os.getenv("COMPUTE_URL", "") or "").rstrip("/")
INTERNAL_SHARED_SECRET = os.getenv("INTERNAL_SHARED_SECRET", "")
TIMEOUT_SECS = float(os.getenv("COMPUTE_TIMEOUT_SECS", "6.0"))
FALLBACK_TTL = int(os.getenv("AGGREGATOR_CACHE_TTL_SECS", "300"))  # default 5 minutes

# ----------------------------- Cache -----------------------------

class _TTLCache:
    def __init__(self, ttl: int):
        self.ttl = max(1, int(ttl))
        self._data: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def _key(self, path: str, method: str, body: Optional[dict], params: Optional[dict]) -> str:
        b = "" if body is None else json.dumps(body, sort_keys=True)
        p = "" if params is None else json.dumps(params, sort_keys=True)
        return f"{method.upper()} {path}|{b}|{p}"

    def get(self, path: str, method: str, body: Optional[dict], params: Optional[dict]) -> Optional[Dict[str, Any]]:
        key = self._key(path, method, body, params)
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            exp, value = item
            if time.time() > exp:
                self._data.pop(key, None)
                return None
            return value

    def set(self, path: str, method: str, body: Optional[dict], params: Optional[dict], value: Dict[str, Any]) -> None:
        key = self._key(path, method, body, params)
        with self._lock:
            self._data[key] = (time.time() + self.ttl, value)


_cache = _TTLCache(FALLBACK_TTL)

def _headers() -> Dict[str, str]:
    h = {"Accept": "application/json"}
    if INTERNAL_SHARED_SECRET:
        h["X-Internal-Secret"] = INTERNAL_SHARED_SECRET
    return h

# ----------------------------- Sync client -----------------------------

class Aggregator:
    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        self.base_url = (base_url or COMPUTE_URL or "").rstrip("/")
        self.timeout = float(timeout or TIMEOUT_SECS)

    # Low-level
    def call(self, path: str, method: str = "GET", *, body: Optional[dict] = None, params: Optional[dict] = None) -> Tuple[Dict[str, Any], int]:
        if not self.base_url:
            return {"mode":"unavailable", "reason":"COMPUTE_URL not set"}, 503

        url = f"{self.base_url}/{path.lstrip('/')}"
        cached = _cache.get(path, method, body, params)
        try:
            if method.upper() == "POST":
                r = requests.post(url, headers=_headers(), json=body or {}, params=params or {}, timeout=self.timeout)  # type: ignore[arg-type]
            else:
                r = requests.get(url, headers=_headers(), params=params or {}, timeout=self.timeout)  # type: ignore[arg-type]
            r.raise_for_status()
            data = r.json() if r.content else {}
            _cache.set(path, method, body, params, data)
            return data, r.status_code
        except Exception as e:
            if cached is not None:
                return {"mode":"degraded","reason":f"compute unavailable: {type(e).__name__}","data":cached,"stale":True}, 200
            return {"mode":"unavailable","reason":f"compute unavailable: {type(e).__name__}","data":None}, 503

    # High-level wrappers
    def settings(self) -> Tuple[Dict[str, Any], int]:
        return self.call("settings", "GET")

    def predict(self, *, symbol: str = "XAUUSD", timeframe: str = "1d", lookback: int = 256, extra: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], int]:
        body = {"symbol": symbol, "timeframe": timeframe, "lookback": lookback, "extra": extra or {}}
        return self.call("predict", "POST", body=body)

    def bias(self, *, symbol: str = "XAUUSD", timeframe: str = "1d") -> Tuple[Dict[str, Any], int]:
        return self.call("bias", "GET", params={"symbol": symbol, "timeframe": timeframe})

    def bias_influence(self, *, symbol: str = "XAUUSD") -> Tuple[Dict[str, Any], int]:
        return self.call("bias/influence", "GET", params={"symbol": symbol})

    def feature_importance(self) -> Tuple[Dict[str, Any], int]:
        return self.call("features/importance", "GET")

    def market_series(self, *, symbol: str = "XAUUSD", timeframe: str = "1d") -> Tuple[Dict[str, Any], int]:
        return self.call("market/series", "GET", params={"symbol": symbol, "timeframe": timeframe})


# ----------------------------- Async client -----------------------------

class AsyncAggregator:
    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        self.base_url = (base_url or COMPUTE_URL or "").rstrip("/")
        self.timeout = float(timeout or TIMEOUT_SECS)
        self._client: Optional["httpx.AsyncClient"] = None

    async def _get_client(self) -> "httpx.AsyncClient":
        if httpx is None:
            raise RuntimeError("httpx is required for AsyncAggregator")
        if self._client and not self._client.is_closed:
            return self._client
        self._client = httpx.AsyncClient(base_url=self.base_url or "", timeout=self.timeout, headers=_headers())
        return self._client

    async def aclose(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # Low-level
    async def call(self, path: str, method: str = "GET", *, body: Optional[dict] = None, params: Optional[dict] = None) -> Tuple[Dict[str, Any], int]:
        if not self.base_url:
            return {"mode":"unavailable", "reason":"COMPUTE_URL not set"}, 503

        cli = await self._get_client()
        cached = _cache.get(path, method, body, params)
        try:
            if method.upper() == "POST":
                r = await cli.post(path, json=body or {}, params=params or {})
            else:
                r = await cli.get(path, params=params or {})
            r.raise_for_status()
            data = r.json()
            _cache.set(path, method, body, params, data)
            return data, r.status_code
        except Exception as e:
            if cached is not None:
                return {"mode":"degraded","reason":f"compute unavailable: {type(e).__name__}","data":cached,"stale":True}, 200
            return {"mode":"unavailable","reason":f"compute unavailable: {type(e).__name__}","data":None}, 503

    # High-level wrappers
    async def settings(self) -> Tuple[Dict[str, Any], int]:
        return await self.call("settings", "GET")

    async def predict(self, *, symbol: str = "XAUUSD", timeframe: str = "1d", lookback: int = 256, extra: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], int]:
        body = {"symbol": symbol, "timeframe": timeframe, "lookback": lookback, "extra": extra or {}}
        return await self.call("predict", "POST", body=body)

    async def bias(self, *, symbol: str = "XAUUSD", timeframe: str = "1d") -> Tuple[Dict[str, Any], int]:
        return await self.call("bias", "GET", params={"symbol": symbol, "timeframe": timeframe})

    async def bias_influence(self, *, symbol: str = "XAUUSD") -> Tuple[Dict[str, Any], int]:
        return await self.call("bias/influence", "GET", params={"symbol": symbol})

    async def feature_importance(self) -> Tuple[Dict[str, Any], int]:
        return await self.call("features/importance", "GET")

    async def market_series(self, *, symbol: str = "XAUUSD", timeframe: str = "1d") -> Tuple[Dict[str, Any], int]:
        return await self.call("market/series", "GET", params={"symbol": symbol, "timeframe": timeframe})


# ----------------------------- CLI (debug) -----------------------------

if __name__ == "__main__":
    agg = Aggregator()
    print("Compute URL:", agg.base_url or "(not set)")
    ok, code = agg.settings()
    print("Settings:", code, json.dumps(ok)[:300])
    pred, code = agg.predict(symbol="XAUUSD")
    print("Predict:", code, json.dumps(pred)[:300])
    bias, code = agg.bias(symbol="XAUUSD")
    print("Bias:", code, json.dumps(bias)[:300])
    feats, code = agg.feature_importance()
    print("Features:", code, json.dumps(feats)[:300])
