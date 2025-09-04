"""
goldmind_integration.py — GoldMIND AI API Integration (hardened)
----------------------------------------------------------------
A clean, production-ready async client for the external GoldMIND API, used for
advanced text analysis (e.g., cognitive-bias detection) and model health checks.

Upgrades vs. original:
- Session reuse with aiohttp.ClientSession (connection pooling)
- Robust retry/backoff with jitter and 429/5xx-aware handling
- Tight timeouts, structured error objects, and safe logging
- Convenience wrapper analyze_cognitive_bias(text, ...) returning a stable shape
- Optional circuit-breaker (cooldown after repeated failures)
- Env-driven overrides: GOLDMIND_API_KEY, GOLDMIND_BASE_URL, GOLDMIND_TIMEOUT_SEC

Public surface:
    client = Goldmind(api_key=None, base_url="...")
    await client.analyze_cognitive_bias(text, top_k=5) -> List[Dict]
    await client.search(query, top_k=5, hybrid=True, rerank=True, filters=None) -> List[Dict]
    await client.get_model_status() -> Dict

This module is async-first. If you need sync wrappers in Flask routes,
run with asyncio.run or an event loop helper.
"""

from __future__ import annotations

import os
import json
import time
import math
import asyncio
import logging
import random
from typing import Dict, List, Optional, Any

import aiohttp

logger = logging.getLogger("goldmind.integration")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


class Goldmind:
    """
    Async client for the GoldMIND AI API with connection pooling and retries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        initial_retry_delay: float = 0.5,
        timeout_sec: Optional[float] = None,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_cooldown_sec: int = 30,
    ) -> None:
        self.api_key = api_key or os.getenv("GOLDMIND_API_KEY")
        self.base_url = (base_url or os.getenv("GOLDMIND_BASE_URL") or "https://api.goldmind.ai/v1").rstrip("/")
        self.max_retries = int(max_retries)
        self.initial_retry_delay = float(initial_retry_delay)
        self.timeout_sec = float(timeout_sec or os.getenv("GOLDMIND_TIMEOUT_SEC") or 30)
        self._session: Optional[aiohttp.ClientSession] = None

        # simple circuit breaker
        self.cb_threshold = int(circuit_breaker_threshold)
        self.cb_cooldown = int(circuit_breaker_cooldown_sec)
        self._cb_fail_count = 0
        self._cb_open_until = 0.0

        if not self.api_key:
            logger.critical("❌ GOLDMIND_API_KEY missing — API features disabled.")
        else:
            logger.info("GoldMIND client initialized (base=%s, timeout=%ss).", self.base_url, self.timeout_sec)

    # ---------------- Session lifecycle ----------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session and not self._session.closed:
            return self._session
        headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
            "User-Agent": "GoldMIND-AI-Client/2.0",
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout_sec, connect=min(10.0, self.timeout_sec))
        self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session

    async def aclose(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ---------------- Core request with retry/backoff ----------------

    async def _request(self, endpoint: str, *, method: str = "POST", payload: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.api_key:
            return {"error": "GoldMIND API key is missing.", "source": "client_error"}

        # circuit breaker
        now = time.time()
        if now < self._cb_open_until:
            return {"error": "circuit_open", "retry_after_sec": round(self._cb_open_until - now, 2), "source": "client_backoff"}

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        attempt = 0
        last_error: Optional[str] = None
        session = await self._get_session()

        while True:
            try:
                attempt += 1
                if method.upper() == "POST":
                    async with session.post(url, json=payload) as resp:
                        data = await self._parse_response(resp)
                elif method.upper() == "GET":
                    async with session.get(url, params=params) as resp:
                        data = await self._parse_response(resp)
                else:
                    return {"error": f"unsupported_method:{method}", "source": "client_error"}

                # successful reply resets breaker
                self._cb_fail_count = 0
                return data

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                status = getattr(e, "status", None)
                last_error = f"{type(e).__name__}:{getattr(e, 'message', str(e))}"
                retryable = (status is None) or (status >= 500) or (status == 429)
                if not retryable or attempt > self.max_retries + 1:
                    self._note_failure()
                    logger.error("GoldMIND request failed (final): %s", last_error)
                    return {"error": f"network:{last_error}", "source": "goldmind_api_failure"}

                delay = self._backoff_delay(attempt, status)
                logger.warning("GoldMIND request error (attempt %d/%d): %s → retrying in %.2fs", attempt, self.max_retries + 1, last_error, delay)
                await asyncio.sleep(delay)

            except Exception as e:
                self._note_failure()
                logger.exception("GoldMIND unexpected error: %s", e)
                return {"error": f"unexpected:{e}", "source": "goldmind_api_failure"}

    async def _parse_response(self, resp: aiohttp.ClientResponse) -> Dict[str, Any]:
        # Observe server-side rate limiting hints
        retry_after = resp.headers.get("Retry-After")
        limit_info = {k.lower(): v for k, v in resp.headers.items() if k.lower().startswith("x-ratelimit")}

        text = await resp.text()
        try:
            data = json.loads(text) if text else {}
        except Exception:
            data = {"raw": text}

        if 200 <= resp.status < 300:
            if isinstance(data, dict):
                data.setdefault("_ratelimit", limit_info)
            return data

        # non-2xx
        err = {
            "error": f"http:{resp.status}",
            "details": data,
            "retry_after": float(retry_after) if retry_after and retry_after.isdigit() else None,
            "source": "goldmind_api_error",
        }
        raise aiohttp.ClientResponseError(request_info=resp.request_info, history=resp.history, status=resp.status, message=json.dumps(err)[:200])

    def _backoff_delay(self, attempt: int, status: Optional[int]) -> float:
        base = self.initial_retry_delay * (2 ** (attempt - 1))
        # jitter: 0.8..1.2x
        jitter = random.uniform(0.8, 1.2)
        # if 429, honor minimal cool-off
        if status == 429:
            base = max(base, 1.5)
        return min(10.0, base * jitter)

    def _note_failure(self) -> None:
        self._cb_fail_count += 1
        if self._cb_fail_count >= self.cb_threshold:
            self._cb_open_until = time.time() + self.cb_cooldown
            logger.error("GoldMIND circuit breaker OPEN for %ss after %d consecutive failures.", self.cb_cooldown, self._cb_fail_count)
            self._cb_fail_count = 0

    # ---------------- High level APIs ----------------

    async def search(self, query: str, top_k: int = 5, hybrid: bool = True, rerank: bool = True, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        General analysis endpoint used by CognitiveBiasDetector and others.
        """
        payload = {
            "text_to_analyze": query,
            "analysis_options": {"top_k": int(top_k), "hybrid_search": bool(hybrid), "rerank_results": bool(rerank), "filters": filters or {}},
        }
        data = await self._request("analyze/text", method="POST", payload=payload)
        if "error" in data:
            logger.error("GoldMIND search failed: %s", data.get("error"))
            return [{"content": f"Error: {data.get('error')}", "source": data.get("source", "unknown")}]
        results = data.get("results")
        if isinstance(results, list):
            return results
        return [{"content": "Error: Invalid API response structure.", "source": "goldmind_api_error"}]

    async def analyze_cognitive_bias(self, text: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Convenience wrapper for cognitive-bias analysis.
        Returns:
            [{"bias": "...", "explanation": "...", "severity": 0..1, "source": "goldmind"}] on success,
            [{"content": "Error: ...", "source": "..."}] on failure
        """
        results = await self.search(text, top_k=top_k, hybrid=True, rerank=True, filters={"analysis_type": "cognitive_bias"})
        # normalize a compact shape if GoldMIND returns richer objects
        out: List[Dict[str, Any]] = []
        for r in results or []:
            if "error" in r.get("content", "").lower():
                out.append(r)
                continue
            out.append({
                "bias": r.get("bias") or r.get("label") or r.get("title") or "unknown",
                "explanation": r.get("explanation") or r.get("content") or "",
                "severity": float(r.get("severity", 0.0)) if isinstance(r.get("severity", None), (int, float)) else 0.0,
                "source": "goldmind"
            })
        return out

    async def get_model_status(self) -> Dict[str, Any]:
        data = await self._request("status/models", method="GET", params=None)
        if "error" in data:
            return {"status": "offline", "error": data.get("error"), "source": data.get("source")}
        return data


# ---------------- Demo ----------------
if __name__ == "__main__":
    async def _demo() -> None:
        os.environ.setdefault("GOLDMIND_API_KEY", "YOUR_ACTUAL_GOLDMIND_API_KEY_HERE")
        cli = Goldmind()
        print("Model status:", await cli.get_model_status())
        res = await cli.analyze_cognitive_bias("I'm certain this will work — I've invested too much to quit.", top_k=3)
        print("Bias analysis:", json.dumps(res, indent=2))
        await cli.aclose()

    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception as e:
        logger.exception("goldmind_integration demo failed: %s", e)
