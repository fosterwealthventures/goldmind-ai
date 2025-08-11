import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class TextRequest(BaseModel):
    text: str
    user_id: int

@app.post("/analyze/text")
async def analyze_text(request: Request, data: TextRequest):
    """
    Mock endpoint to simulate the GoldMIND AI bias detection API.
    It returns a hardcoded, no-bias response.
    """
    logger.info(f"Mock API received request for user_id {data.user_id} with text: '{data.text}'")

    mock_response = {
        "analysis_id": "mock-12345",
        "bias_detected": False,
        "bias_type": "None",
        "confidence": 0.95,
        "reasoning": "Mock analysis indicates no cognitive bias."
    }
    return mock_response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)