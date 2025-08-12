"""
lstm_temporal_analysis.py

Enhanced LSTM Temporal Analysis Engine for GoldMIND AI.
Reads the ticker and parameters from `config.json`, processes sliding-window data, and generates predictive insights.
"""

import logging
import asyncio
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Load global config.json if present
_log = logging.getLogger(__name__)
_global_cfg: Dict[str, Any] = {}
_cfg_file = Path(__file__).parent / "config.json"
if _cfg_file.exists():
    try:
        _global_cfg = json.loads(_cfg_file.read_text())
        _log.info(f"Loaded config from {_cfg_file}")
    except Exception as e:
        _log.warning(f"Failed to load global config: {e}")

# Attempt real FinancialDataFramework import; fallback to stub
try:
    from financial_data_framework import FinancialDataFramework
except ImportError:
    _log.critical("❌ Could not import FinancialDataFramework; using stub.")
    class FinancialDataFramework:
        async def get_processed_data(self, ticker: str, days_back: int) -> pd.DataFrame:
            _log.warning("Stub get_processed_data: returning empty DataFrame.")
            return pd.DataFrame(columns=['Close'])

class LSTMTemporalAnalysis:
    """
    LSTM-based temporal analysis pipeline.
    """
    def __init__(
        self,
        financial_data_framework: FinancialDataFramework,
        config: Optional[Dict[str, Any]] = None
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.financial_data_framework = financial_data_framework
        # Merge global config with passed config (passed overrides global)
        cfg = {**_global_cfg, **(config or {})}

        # Determine ticker: 'symbol' or 'ticker' key, default to 'GLD'
        self.ticker: str = str(cfg.get("symbol") or cfg.get("ticker") or "GLD")
        # Sequence length from 'lstm_window' or 'sequence_length'
        self.sequence_length: int = int(cfg.get("lstm_window", cfg.get("sequence_length", 60)))
        self.features: list = cfg.get("features", ["Close"])
        self.model_path: Optional[str] = cfg.get("model_path")
        # Optional buffer days for lookback
        self.buffer_days: int = int(cfg.get("buffer_days", 10))

        self.model = self._load_model()
        self.logger.info(
            f"Initialized LSTMTemporalAnalysis for '{self.ticker}' "
            f"(seq_len={self.sequence_length}, features={self.features})"
        )

    def _load_model(self):
        if self.model_path:
            try:
                from tensorflow.keras.models import load_model
                model = load_model(self.model_path)
                self.logger.info(f"✅ Loaded LSTM model from '{self.model_path}'")
                return model
            except Exception as e:
                self.logger.error(f"Error loading LSTM model at '{self.model_path}': {e}")
        # Fallback mock model
        self.logger.warning("⚠️ Using mock LSTM model. Predictions will be random.")
        class MockModel:
            def predict(self, data: np.ndarray) -> np.ndarray:
                return np.random.uniform(-1, 1, size=(data.shape[0], 1))
        return MockModel()

    def _preprocess_data(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        arr = data[self.features].values
        if arr.shape[0] < self.sequence_length:
            self.logger.error(
                f"Insufficient data: required {self.sequence_length}, found {arr.shape[0]}"
            )
            return None
        # Take last `sequence_length` rows
        window = arr[-self.sequence_length :]
        self.logger.debug(f"Preprocessed window shape: {window.shape}")
        return window.reshape((1, self.sequence_length, len(self.features)))

    async def generate_prediction(
        self,
        ticker: Optional[str] = None,
        days: Optional[int] = None
    ) -> Dict[str, Any]:
        # Use override ticker or default
        symbol = str(ticker or self.ticker)
        # Lookback period
        lookback = int(days or (self.sequence_length + self.buffer_days))
        self.logger.info(f"Generating prediction for {symbol} over {lookback} days.")

        # Fetch data
        try:
            df = await self.financial_data_framework.get_processed_data(symbol, lookback)
        except Exception as e:
            self.logger.error(f"Data fetch failed for {symbol}: {e}")
            return {
                "action": "HOLD", "confidence": 0.0,
                "reasoning": "Data fetch error.",
                "position_size": 0.0, "risk_score": 0.0
            }

        if df.empty:
            return {
                "action": "HOLD", "confidence": 0.0,
                "reasoning": "No data available.",
                "position_size": 0.0, "risk_score": 0.0
            }

        arr = self._preprocess_data(df)
        if arr is None:
            return {
                "action": "HOLD", "confidence": 0.0,
                "reasoning": "Insufficient data.",
                "position_size": 0.0, "risk_score": 0.0
            }

        # Model prediction
        raw = float(self.model.predict(arr)[0][0])
        if raw > 0.5:
            action = "BUY"
            conf = raw
        elif raw < -0.5:
            action = "SELL"
            conf = abs(raw)
        else:
            action = "HOLD"
            conf = 0.5 + abs(raw)

        pos_size = min(0.5, conf * 0.7)
        risk = round(1.0 - conf, 4)
        self.logger.info(f"LSTM result for {symbol}: action={action}, confidence={conf:.2f}")

        return {
            "action": action,
            "confidence": conf,
            "reasoning": f"Temporal LSTM analysis indicates a {action} trend.",
            "position_size": pos_size,
            "risk_score": risk
        }

# --- Standalone test harness ---
if __name__ == "__main__":
    import json
    async def _test():
        logging.basicConfig(level=logging.INFO)
        class MockFDF:
            async def get_processed_data(self, ticker, days_back):
                dates = pd.date_range(end=pd.Timestamp.now(), periods=days_back)
                data = np.random.uniform(150, 200, days_back)
                return pd.DataFrame({"Close": data}, index=dates)
        engine = LSTMTemporalAnalysis(MockFDF(), config={})
        print(json.dumps(await engine.generate_prediction(), indent=2))
    asyncio.run(_test())
