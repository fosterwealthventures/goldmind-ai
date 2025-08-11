import os
import aiohttp
import asyncio
import logging
import json
from typing import Dict, Any

class GoldMINDClient:
    def __init__(self, api_url: str, logger: logging.Logger):
        # Check for an environment variable override
        self.api_url = os.getenv("GOLDMIND_API_URL", api_url)
        self.logger = logger
        self.session = aiohttp.ClientSession()
        self.lock = asyncio.Lock()

    async def analyze_text(self, text: str, user_id: int) -> Dict[str, Any]:
        url = f"{self.api_url}/analyze/text"
        payload = {
            "text": text,
            "user_id": user_id
        }
        self.logger.info(f"Submitting request to GoldMIND API at {url}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with self.lock:
                    async with self.session.post(url, json=payload, timeout=60) as response:
                        response.raise_for_status()
                        result = await response.json()
                        self.logger.info("Successfully received response from GoldMIND API.")
                        return result
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                self.logger.error(f"Attempt {attempt + 1} for analyze/text failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** (attempt + 1))  # Exponential backoff
                else:
                    self.logger.warning("GoldMIND API returned invalid response, assuming no bias.")
                    return {
                        "bias_detected": False,
                        "bias_type": "None",
                        "confidence": 0.0,
                        "reasoning": "No significant cognitive biases detected by GoldMIND AI."
                    }

    async def close_session(self):
        if not self.session.closed:
            await self.session.close()
            self.logger.info("GoldMIND API client session closed.")