from __future__ import annotations
import os, time, json
from typing import Any, Dict, Tuple, Optional
import requests

COMPUTE_URL = os.getenv("COMPUTE_URL", "").rstrip("/")
INTERNAL_SHARED_SECRET = os.getenv("INTERNAL_SHARED_SECRET", "")
TIMEOUT_SECS = float(os.getenv("COMPUTE_TIMEOUT_SECS", "4.0"))   # short, fail-fast
FALLBACK_TTL = int(os.getenv("FALLBACK_TTL_SECS", "300"))        # 5 minutes

# simple in-proc TTL cache {key: (expires_at, payload)}
_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

def _cache_key(path: str, method: str, body: Optional[dict]) -> str:
    b = "" if body is None else json.dumps(body, sort_keys=True)
    return f"{method.upper()} {path}|{b}"

def _get_cached(key: str) -> Optional[Dict[str, Any]]:
    now = time.time()
    hit = _cache.get(key)
    if not hit:
        return None
    exp, payload = hit
    if exp < now:
        _cache.pop(key, None)
        return None
    return payload

def _set_cached(key: str, payload: Dict[str, Any]) -> None:
    _cache[key] = (time.time() + FALLBACK_TTL, payload)

def forward_json(path: str, body: Dict[str, Any] | None, method: str = "POST"):
    """
    Forward to compute with secret; on failure, return degraded with cached payload if any.
    Returns (payload, status_code).
    """
    assert COMPUTE_URL, "COMPUTE_URL is not set"
    url = f"{COMPUTE_URL}{path}"
    headers = {
        "Content-Type": "application/json",
        "X-Internal-Secret": INTERNAL_SHARED_SECRET or "",
    }
    key = _cache_key(path, method, body)

    try:
        if method.upper() == "POST":
            r = requests.post(url, headers=headers, json=body, timeout=TIMEOUT_SECS)
        else:
            r = requests.get(url, headers=headers, timeout=TIMEOUT_SECS)
        r.raise_for_status()
        data = r.json() if r.content else {}
        _set_cached(key, data)               # save last-good
        return data, r.status_code
    except Exception as e:
        cached = _get_cached(key)
        if cached is not None:
            return {
                "mode": "degraded",
                "reason": f"compute unavailable: {type(e).__name__}",
                "data": cached,
                "stale": True,
            }, 200
        return {
            "mode": "unavailable",
            "reason": f"compute unavailable: {type(e).__name__}",
            "data": None,
        }, 503
