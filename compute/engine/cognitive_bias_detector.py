"""
Cognitive Bias Detection components for GoldMIND Compute.

This module contains:
  - CognitiveBiasDetector: high-level wrapper that turns free-form user input
    into a normalized bias report, backed by the GoldMIND API.
  - GoldMINDClient: lightweight async client with retries, timeouts,
    and an async context manager for clean resource handling.

Public API (stable):
  class CognitiveBiasDetector:
      async def analyze_input(self, user_input: str, user_id: str) -> Dict[str, Any]

  class GoldMINDClient:
      async def analyze_text(self, text: str, user_id: str) -> Optional[Dict[str, Any]]

Drop-in compatible with the previous version, now with better error handling,
input validation, and configuration via environment variables.

Env (optional):
  GOLDMIND_API_URL        - Base URL for the GoldMIND API (default: http://localhost:8080)
  GOLDMIND_API_KEY        - If set, sent as 'Authorization: Bearer <key>'
  GOLDMIND_API_TIMEOUT    - Per-request timeout seconds (default: 20)
  GOLDMIND_API_RETRIES    - Number of attempts per call (default: 3)
"""

from __future__ import annotations

import os
import asyncio
import logging
from typing import Any, Dict, Optional

import aiohttp

# ----------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------

def _coerce_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y"}
    return False

def _coerce_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _normalize_confidence(val: Any) -> float:
    """Accept 0..1 or 0..100 and return 0..1."""
    c = _coerce_float(val, 0.0)
    if c > 1.0:
        c = max(0.0, min(100.0, c)) / 100.0
    return max(0.0, min(1.0, c))

# ----------------------------------------------------------------------------
# GoldMIND Client
# ----------------------------------------------------------------------------

class GoldMINDClient:
    """Async client for the GoldMIND API with retry + context manager support."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        timeout_sec: Optional[int] = None,
        retries: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.api_url = (api_url or os.getenv("GOLDMIND_API_URL", "http://localhost:8080")).rstrip("/")
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.timeout_sec = int(timeout_sec or os.getenv("GOLDMIND_API_TIMEOUT", 20))
        self.retries = int(retries or os.getenv("GOLDMIND_API_RETRIES", 3))
        self.api_key = api_key or os.getenv("GOLDMIND_API_KEY", "")
        self._session: Optional[aiohttp.ClientSession] = None
        self.logger.info("GoldMIND API client configured: url=%s, timeout=%ss, retries=%s", self.api_url, self.timeout_sec, self.retries)

    # --- async context manager ---
    async def __aenter__(self) -> "GoldMINDClient":
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self.logger.debug("GoldMIND client session opened")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # Lazy-create if used outside context manager
            self._session = aiohttp.ClientSession()
            self.logger.debug("GoldMIND client session lazily opened")
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self.logger.info("GoldMIND API client session closed.")

    # --- internal HTTP call with retries ---
    async def _call_api(self, method: str, endpoint: str, json_body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Simple exponential backoff with jitter
        for attempt in range(1, self.retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
                self.logger.info("GoldMIND %s %s attempt=%s", method, url, attempt)
                async with self.session.request(method.upper(), url, json=json_body, headers=headers, timeout=timeout) as resp:
                    text_ct = resp.headers.get("content-type", "")
                    if resp.status >= 400:
                        body = await resp.text()
                        self.logger.error("GoldMIND %s %s failed: HTTP %s, body=%s", method, url, resp.status, body[:500])
                        # retry on 5xx / timeouts; not on 4xx except 429
                        if resp.status >= 500 or resp.status == 429:
                            raise aiohttp.ClientResponseError(resp.request_info, resp.history, status=resp.status)
                        return None
                    if "application/json" in text_ct:
                        return await resp.json()
                    # Try to parse non-json (rare)
                    try:
                        data = await resp.json()
                        return data
                    except Exception:
                        raw = await resp.text()
                        return {"raw": raw}
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                self.logger.warning("Attempt %s error: %s", attempt, e)
                if attempt >= self.retries:
                    return None
                await asyncio.sleep((2 ** (attempt - 1)) * 0.3)
            except Exception as e:
                self.logger.error("Unexpected error calling GoldMIND %s %s: %s", method, url, e, exc_info=True)
                return None
        return None

    # --- public API ---
    async def analyze_text(self, text: str, user_id: str) -> Optional[Dict[str, Any]]:
        payload = {"text": str(text or "").strip(), "user_id": str(user_id or "").strip()}
        return await self._call_api("POST", "analyze/text", payload)

# ----------------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------------

class CognitiveBiasDetector:
    """Analyzes user input for cognitive biases using the GoldMIND API."""

    def __init__(self, goldmind_client: GoldMINDClient) -> None:
        self.goldmind_client = goldmind_client
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("CognitiveBiasDetector initialized with GoldMIND client.")

    async def analyze_input(self, user_input: str, user_id: str) -> Dict[str, Any]:
        """Return a normalized bias report for the dashboard/compute layers.

        Returns shape:
        {
          "bias_detected": bool,
          "bias_type": "None|Confirmation Bias|...|Error|Unknown",
          "confidence": 0.0..1.0,
          "reasoning": str
        }
        """
        text = (user_input or "").strip()
        uid  = (user_id or "").strip()
        if not text:
            return {
                "bias_detected": False,
                "bias_type": "None",
                "confidence": 0.0,
                "reasoning": "No text provided.",
            }

        # Default no-bias report
        report: Dict[str, Any] = {
            "bias_detected": False,
            "bias_type": "None",
            "confidence": 0.0,
            "reasoning": "No significant cognitive biases detected by GoldMIND AI.",
        }

        try:
            self.logger.info("Submitting user input for GoldMIND AI bias analysis: %r", text[:200])
            data = await self.goldmind_client.analyze_text(text=text, user_id=uid)
            if not isinstance(data, dict):
                self.logger.warning("GoldMIND API returned invalid response; using default no-bias report.")
                return report

            # Typical response fields we support (flexible naming)
            bias_detected = _coerce_bool(data.get("bias_detected") or data.get("detected") or data.get("has_bias"))
            bias_type     = (data.get("bias_type") or data.get("bias") or data.get("type") or "Unknown")
            confidence    = _normalize_confidence(data.get("confidence"))
            reasoning     = data.get("reasoning") or data.get("explanation") or report["reasoning"]

            report.update({
                "bias_detected": bool(bias_detected),
                "bias_type": str(bias_type),
                "confidence": float(confidence),
                "reasoning": str(reasoning),
            })
            self.logger.info("Bias analysis result: %s", report)
        except Exception as e:
            self.logger.error("Error during cognitive bias analysis", exc_info=True)
            report.update({
                "bias_detected": False,
                "bias_type": "Error",
                "confidence": 0.0,
                "reasoning": f"Error during bias detection: {e}",
            })
        return report
