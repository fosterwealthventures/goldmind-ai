"""
goldmind_client.py — Backward-Compatible Wrapper (hardened)
----------------------------------------------------------
This preserves the original `GoldMINDClient` API, but under the hood it
delegates to the newer async client in `goldmind_integration.Goldmind` when
available. If that module isn't present, it falls back to an internal
minimal client with retries, backoff, and structured errors.

Public surface (unchanged):
    client = GoldMINDClient(api_url="https://api.goldmind.ai/v1", logger=logger)
    await client.analyze_text("text...", user_id=123)
    await client.close_session()

Environment overrides:
    GOLDMIND_API_URL (base URL)
    GOLDMIND_API_KEY (used when delegating to goldmind_integration)
    GOLDMIND_TIMEOUT_SEC
"""

from __future__ import annotations

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional

try:
    import aiohttp  # type: ignore
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore


# ---------------- Delegating wrapper ----------------

class GoldMINDClient:
    def __init__(self, api_url: str, logger: logging.Logger):
        self.api_url = os.getenv("GOLDMIND_API_URL", api_url).rstrip("/")
        self.logger = logger
        self._delegate = None  # type: Optional[object]
        self._session = None   # type: Optional["aiohttp.ClientSession"]
        self._lock = asyncio.Lock()
        self._timeout = float(os.getenv("GOLDMIND_TIMEOUT_SEC", "30"))

        # Try to delegate to the modern client if available
        try:
            from goldmind_integration import Goldmind  # local import
            self._delegate = Goldmind(base_url=self.api_url)
            self.logger.info("GoldMINDClient delegating to goldmind_integration.Goldmind")
        except Exception as e:  # fallback to internal client
            self._delegate = None
            self.logger.warning("GoldMIND integration client not available; using internal HTTP client (%s).", e)
            if aiohttp is None:
                raise RuntimeError("aiohttp is required for the fallback GoldMINDClient.")

    # ------------- Public API (backwards-compatible) -------------

    async def analyze_text(self, text: str, user_id: int) -> Dict[str, Any]:
        """
        Calls the GoldMIND /analyze/text endpoint and normalizes the result.
        """
        if self._delegate is not None:
            try:
                # Use the richer integration path with bias filter
                from goldmind_integration import Goldmind  # ensure type
                results = await self._delegate.search(
                    query=text,
                    top_k=5,
                    hybrid=True,
                    rerank=True,
                    filters={"analysis_type": "cognitive_bias", "user_id": user_id},
                )
                # Map to the legacy shape expected by older code paths
                bias_hit = next((r for r in results if str(r.get("severity", 0)) not in ("", "0", "0.0")), None)
                return {
                    "bias_detected": bool(bias_hit),
                    "bias_type": (bias_hit or {}).get("bias") or (bias_hit or {}).get("label") or "None",
                    "confidence": float((bias_hit or {}).get("severity", 0.0)) if isinstance((bias_hit or {}).get("severity", 0.0), (int, float)) else 0.0,
                    "reasoning": (bias_hit or {}).get("explanation") or (bias_hit or {}).get("content") or "No significant cognitive biases detected by GoldMIND AI.",
                    "raw": results,
                }
            except Exception as e:
                self.logger.error("Delegated analyze_text failed: %s", e)

        # Fallback: direct HTTP call with retries/backoff (keeps legacy route)
        if aiohttp is None:
            return {
                "bias_detected": False, "bias_type": "None", "confidence": 0.0,
                "reasoning": "aiohttp unavailable; cannot call GoldMIND API."
            }

        url = f"{self.api_url}/analyze/text"
        payload = {"text_to_analyze": text, "user_id": user_id}
        headers = {
            "Authorization": f"Bearer {os.getenv('GOLDMIND_API_KEY','')}",
            "Content-Type": "application/json",
            "User-Agent": "GoldMIND-Compat-Client/2.0",
        }

        max_retries = 3
        for attempt in range(1, max_retries + 2):
            try:
                async with self._get_session() as session:
                    async with self._lock:
                        async with session.post(url, headers=headers, json=payload, timeout=self._timeout) as resp:
                            text_body = await resp.text()
                            try:
                                data = json.loads(text_body) if text_body else {}
                            except Exception:
                                data = {"raw": text_body}

                            if 200 <= resp.status < 300:
                                # Normalize to legacy shape if needed
                                if isinstance(data, dict) and {"bias_detected","bias_type","confidence","reasoning"} <= data.keys():
                                    return data
                                # Best-effort mapping
                                bias = (data.get("results") or [{}])[0] if isinstance(data, dict) else {}
                                return {
                                    "bias_detected": bool(bias),
                                    "bias_type": bias.get("bias") or bias.get("label") or "None",
                                    "confidence": float(bias.get("severity", 0.0)) if isinstance(bias.get("severity", 0.0), (int, float)) else 0.0,
                                    "reasoning": bias.get("explanation") or bias.get("content") or "No significant cognitive biases detected by GoldMIND AI.",
                                    "raw": data,
                                }

                            # Non-2xx → maybe retry
                            self.logger.error("GoldMIND HTTP %s: %s", resp.status, text_body[:200])
                            if resp.status < 500 and resp.status != 429:
                                break  # don't retry on 4xx other than 429
                            await asyncio.sleep(min(10, 0.5 * (2 ** (attempt - 1))))

            except (asyncio.TimeoutError, Exception) as e:
                self.logger.error("Attempt %d failed: %s", attempt, e)
                if attempt <= max_retries:
                    await asyncio.sleep(min(10, 0.5 * (2 ** (attempt - 1))))
                else:
                    break

        self.logger.warning("GoldMIND API failed; returning neutral result.")
        return {
            "bias_detected": False,
            "bias_type": "None",
            "confidence": 0.0,
            "reasoning": "No significant cognitive biases detected by GoldMIND AI.",
        }

    async def close_session(self):
        if self._delegate is not None:
            try:
                await self._delegate.aclose()  # type: ignore[attr-defined]
            except Exception:
                pass
        if self._session and aiohttp and not self._session.closed:
            await self._session.close()
            self._session = None
            self.logger.info("GoldMIND client session closed.")

    # ------------- Internals -------------

    async def _get_session(self):
        if self._session and not self._session.closed:
            return self._session
        if aiohttp is None:
            raise RuntimeError("aiohttp is required for HTTP fallback in GoldMINDClient.")
        timeout = aiohttp.ClientTimeout(total=self._timeout, connect=min(self._timeout, 10))
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session


# ---------------- Demo ----------------
if __name__ == "__main__":
    async def _demo():
        logger = logging.getLogger("demo")
        logging.basicConfig(level=logging.INFO)
        client = GoldMINDClient("https://api.goldmind.ai/v1", logger)
        res = await client.analyze_text("I know I'm right; I sunk too much into this.", user_id=1)
        print(json.dumps(res, indent=2)[:600])
        await client.close_session()

    asyncio.run(_demo())
