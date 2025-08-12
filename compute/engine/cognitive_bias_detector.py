# cognitive_bias_detector.py
import logging
from typing import Any, Dict
from engine.goldmind_client import GoldMINDClient

class CognitiveBiasDetector:
    """
    Component responsible for analyzing user input for cognitive biases
    using the GoldMIND AI service.
    """
    def __init__(self, goldmind_client: GoldMINDClient):
        """
        Initializes the CognitiveBiasDetector.

        Args:
            goldmind_client (GoldMINDClient): An instance of the GoldMIND API client.
        """
        self.goldmind_client = goldmind_client
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("CognitiveBiasDetector initialized with GoldMIND client.")

    async def analyze_input(self, user_input: str, user_id: str) -> Dict[str, Any]:
        """
        Analyzes the user's natural language query for cognitive biases by
        calling the GoldMIND AI service.

        Args:
            user_input (str): The user's query text.
            user_id (str): The ID of the user.

        Returns:
            dict: A report on detected biases, including type, confidence, and reasoning.
        """
        self.logger.info(f"Submitting user input for GoldMIND AI bias analysis: '{user_input}'")
        # Default to no bias
        bias_report: Dict[str, Any] = {
            "bias_detected": False,
            "bias_type": "None",
            "confidence": 0.0,
            "reasoning": "No significant cognitive biases detected by GoldMIND AI."
        }
        try:
            analysis_data = await self.goldmind_client.analyze_text(text=user_input, user_id=user_id)
            # If the client returned None or non-dict, treat as no bias
            if not isinstance(analysis_data, dict):
                self.logger.warning("GoldMIND API returned invalid response, assuming no bias.")
                return bias_report
            # Update report from API response
            bias_report.update({
                "bias_detected": analysis_data.get("bias_detected", False),
                "bias_type":     analysis_data.get("bias_type", "Unknown"),
                "confidence":    analysis_data.get("confidence", 0.0),
                "reasoning":     analysis_data.get("reasoning", bias_report["reasoning"])
            })
            self.logger.info(f"Bias analysis result: {bias_report}")
        except Exception as e:
            self.logger.error("Error during cognitive bias analysis", exc_info=True)
            bias_report.update({
                "bias_detected": False,
                "bias_type":     "Error",
                "confidence":    0.0,
                "reasoning":     f"Error during bias detection: {e}"
            })
        return bias_report


# goldmind_client.py
import asyncio
import logging
from typing import Optional, Dict, Any
import aiohttp

class GoldMINDClient:
    """
    Client for interacting with the GoldMIND AI API, with retry logic
    and a persistent aiohttp session.
    """
    def __init__(self, api_url: str, logger: logging.Logger = None):
        self.api_url = api_url.rstrip('/')
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.session = aiohttp.ClientSession()
        self.logger.info("GoldMIND API client initialized.")

    async def close(self):
        """Closes the internal aiohttp session."""
        await self.session.close()
        self.logger.info("GoldMIND API client session closed.")

    async def _call_api(
        self,
        endpoint: str,
        data: Dict[str, Any],
        timeout: int = 20
    ) -> Optional[Dict[str, Any]]:
        """
        Generic method to call the GoldMIND API with simple retry logic.
        Returns JSON dict on success, or None on failure.
        """
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        self.logger.info(f"Submitting request to GoldMIND API at {url}")
        for attempt in range(1, 4):
            try:
                async with self.session.post(url, json=data, timeout=timeout) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                self.logger.error(f"Attempt {attempt} for {endpoint} failed: {e}")
                if attempt < 3:
                    await asyncio.sleep((2 ** (attempt - 1)) * 0.3)
                    continue
                return None
            except Exception as e:
                self.logger.error(f"Unexpected error during API call to {endpoint}: {e}", exc_info=True)
                return None

    async def analyze_text(self, text: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Sends user text to the GoldMIND AI for bias analysis.
        """
        payload = {"text": text, "user_id": user_id}
        return await self._call_api("analyze/text", payload)
