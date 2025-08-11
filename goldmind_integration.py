"""
goldmind_integration.py

GoldMIND AI API Integration
===========================

This module provides a client for interacting with the external GoldMIND AI service,
specifically for advanced text analysis such as cognitive bias detection.
"""

import os
import json
import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
import time # For exponential backoff

logger = logging.getLogger(__name__)

class Goldmind:
    """
    Client for the GoldMIND AI API.
    Handles authentication, asynchronous requests, and response parsing
    for AI-enhanced text analysis services.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.goldmind.ai/v1"):
        """
        Initializes the Goldmind API client.
        :param api_key: Your GoldMIND AI API key. If None, it will attempt to load from GOLDMIND_API_KEY environment variable.
        :param base_url: The base URL for the GoldMIND AI API.
        """
        self.api_key = api_key or os.getenv('GOLDMIND_API_KEY')
        self.base_url = base_url
        self.max_retries = 2 # Max retries for API calls
        self.initial_retry_delay = 0.5 # seconds

        if not self.api_key:
            logger.critical("❌ GoldMIND API key not provided. AI-enhanced features will not work.")
        else:
            logger.info("GoldMIND API client initialized.")
            logger.debug(f"GoldMIND API Base URL: {self.base_url}")

    async def _make_request(self, endpoint: str, method: str = 'POST', data: Optional[Dict] = None, retries: int = 0) -> Dict:
        """
        Internal helper to make an asynchronous HTTP request to the GoldMIND API with exponential backoff.
        :param endpoint: The API endpoint (e.g., "analyze/text").
        :param method: HTTP method ('POST', 'GET').
        :param data: JSON payload for POST requests or params for GET requests.
        :param retries: Current retry attempt count.
        :return: JSON response from the API or an error dictionary.
        """
        if not self.api_key:
            return {"error": "GoldMIND API key is missing.", "source": "client_error"}

        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "GoldMIND-AI-Client/1.0"
        }

        timeout = aiohttp.ClientTimeout(total=30, connect=10)

        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                logger.debug(f"Making {method} request to {url} (attempt {retries + 1}/{self.max_retries + 1})")
                if method == 'POST':
                    async with session.post(url, json=data) as response:
                        response.raise_for_status()
                        return await response.json()
                elif method == 'GET':
                    async with session.get(url, params=data) as response:
                        response.raise_for_status()
                        return await response.json()
                else:
                    return {"error": f"Unsupported HTTP method: {method}", "source": "client_error"}

        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            error_message = f"Network error: {str(e)}"
            if isinstance(e, aiohttp.ClientResponseError):
                error_message = f"HTTP error: Status {e.status}, Message: {e.message}"
            elif isinstance(e, json.JSONDecodeError):
                error_message = f"Invalid JSON response: {e}"
            
            logger.error(f"GoldMIND API request failed for {endpoint}: {error_message}", exc_info=False) # No exc_info for retries

            if retries < self.max_retries:
                delay = self.initial_retry_delay * (2 ** retries)
                logger.warning(f"Retrying GoldMIND API request to {endpoint} in {delay:.2f}s (attempt {retries + 1}/{self.max_retries})...")
                await asyncio.sleep(delay)
                return await self._make_request(endpoint, method, data, retries + 1)
            else:
                logger.error(f"Network failure for {url} after {self.max_retries} retries")
                return {"error": f"Network error: {error_message}", "source": "goldmind_api_failure"}
        except Exception as e:
            logger.error(f"An unexpected error occurred during GoldMIND API request to {endpoint}: {e}", exc_info=True)
            return {"error": f"Unexpected error: {str(e)}", "source": "goldmind_api_failure"}

    async def search(self, query: str, top_k: int = 5, hybrid: bool = True, rerank: bool = True, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Performs a search/analysis query against the GoldMIND AI service.
        This method is designed to be called by the CognitiveBiasDetector.
        
        :param query: The text query or prompt for analysis.
        :param top_k: Number of top results to return.
        :param hybrid: Whether to use hybrid search (if supported by GoldMIND API).
        :param rerank: Whether to rerank results (if supported).
        :param filters: Optional filters for the search (e.g., {"analysis_type": "cognitive_bias"}).
        :return: A list of analysis results from the GoldMIND API, or a list with an error dictionary.
        """
        endpoint = "analyze/text"
        payload = {
            "text_to_analyze": query,
            "analysis_options": {
                "top_k": top_k,
                "hybrid_search": hybrid,
                "rerank_results": rerank,
                "filters": filters or {}
            }
        }
        
        response_data = await self._make_request(endpoint, method='POST', data=payload)
        
        if "error" in response_data:
            logger.error(f"GoldMIND API search failed: {response_data['error']}")
            return [{"content": f"Error: GoldMIND API call failed - {response_data['error']}", "source": response_data['source']}]
        
        if "results" in response_data and isinstance(response_data["results"], list):
            return response_data["results"]
        else:
            logger.warning(f"GoldMIND API response missing 'results' key or is not a list: {response_data}")
            return [{"content": "Error: Invalid API response structure.", "source": "goldmind_api_error"}]

    async def get_model_status(self) -> Dict:
        """Gets the status of the GoldMIND AI models."""
        endpoint = "status/models"
        response_data = await self._make_request(endpoint, method='GET')
        
        if "error" in response_data:
            logger.error(f"Failed to get GoldMIND AI model status: {response_data['error']}")
            return {"status": "offline", "error": response_data['error'], "source": response_data['source']}
        
        return response_data

# --- Standalone Testing Example ---
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Set a dummy API key for testing purposes (replace with your actual key for real tests)
    # For local testing without a real API, the mock client in cognitive_bias_detector.py is used.
    # If you intend to test against a real GoldMIND API, ensure this key is valid and the API is accessible.
    os.environ['GOLDMIND_API_KEY'] = 'YOUR_ACTUAL_GOLDMIND_API_KEY_HERE' 

    goldmind_client = Goldmind()

    print("\n--- Testing GoldMIND API Client (Search/Analyze) ---")
    test_text_1 = "I am fully convinced this strategy will work, despite recent setbacks. I've invested too much to quit now."
    test_text_2 = "Analyzing market trends and historical data suggests a neutral outlook for the coming week."

    print(f"\nAnalyzing text 1: '{test_text_1}'")
    results_1 = await goldmind_client.search(
        query=test_text_1,
        filters={"analysis_type": "cognitive_bias"}
    )
    print(json.dumps(results_1, indent=2))

    print(f"\nAnalyzing text 2: '{test_text_2}'")
    results_2 = await goldmind_client.search(
        query=test_text_2,
        filters={"analysis_type": "cognitive_bias"}
    )
    print(json.dumps(results_2, indent=2))

    print("\n--- Testing GoldMIND API Client (Model Status) ---")
    model_status = await goldmind_client.get_model_status()
    print(json.dumps(model_status, indent=2))

    print("\nGoldMIND API Integration testing complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("GoldMIND Integration example interrupted by user.")
    except Exception as e:
        logger.critical(f"GoldMIND Integration example failed: {e}", exc_info=True)
        exit(1)