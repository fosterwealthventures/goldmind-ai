"""
bias_aware_dual_system_integration.py

Enhanced Bias-Aware Dual System Integration for GoldMIND AI.
Combines bias detection, analytical and LSTM pathways, and conflict resolution.
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import asdict, dataclass

import pandas as pd
import numpy as np

# Attempt to import real components; fallback to stubs
try:
    from cognitive_bias_detector import CognitiveBiasDetector
    from dual_system_conflict_resolution import (
        DualSystemConflictResolver,
        PathwayRecommendation,
        PathwayType
    )
    from lstm_temporal_analysis import LSTMTemporalAnalysis
    from goldmind_client import GoldMINDClient as Goldmind
    from financial_data_framework import FinancialDataFramework
except ImportError as e:
    logging.critical(f"❌ Dependency import failed: {e}")
    
    @dataclass
    class PathwayRecommendation:
        pathway_type: str
        action: str
        confidence: float
        target_price: float
        stop_loss: float
        position_size: float
        reasoning: str
        technical_indicators: dict
        risk_score: float
        timestamp: datetime
    
    @dataclass
    class FinalRecommendation:
        final_action: str
        final_confidence: float
        resolution_method: str
        resolved_reasoning: str
        bias_impact: Optional[str]
    
    class CognitiveBiasDetector:
        def __init__(self, client: Any, config: Dict[str, Any] = None): pass
        async def analyze_input(self, user_input: str) -> Dict[str, Any]:
            return {"bias_detected": False, "bias_type": "None", "confidence": 0.0, "reasoning": "Mock"}
    class DualSystemConflictResolver:
        def __init__(self, config: Dict[str, Any] = None): pass
        def resolve_recommendations(self, *args, **kwargs) -> FinalRecommendation:
            return FinalRecommendation(
                final_action="HOLD",
                final_confidence=0.5,
                resolution_method="mock",
                resolved_reasoning="Mock resolution",
                bias_impact=None
            )
    class PathwayType:
        ANALYTICAL = type("P", (), {"value":"analytical"})
        LSTM_TEMPORAL = type("P", (), {"value":"lstm_temporal"})
    class LSTMTemporalAnalysis:
        def __init__(self, fdf: Any, config: Dict[str, Any] = None): pass
        async def generate_prediction(self, ticker: str, days: int) -> Dict[str, Any]:
            return {"action":"HOLD","confidence":0.5,"reasoning":"mock","position_size":0.0,"risk_score":0.0}
    class Goldmind:
        def __init__(self, api_key: str = ""): pass
        async def close(self): pass
    class FinancialDataFramework:
        async def get_processed_data(self, ticker: str, days_back: int) -> pd.DataFrame:
            return pd.DataFrame()

logger = logging.getLogger(__name__)

class UltimateBiasAwareManager:
    def __init__(
        self,
        config: Dict[str, Any],
        financial_data_framework: FinancialDataFramework,
        lstm_temporal_analyzer: LSTMTemporalAnalysis,
        cognitive_bias_detector: CognitiveBiasDetector,
        conflict_resolver: DualSystemConflictResolver
    ):
        self.config = config
        self.financial_data_framework = financial_data_framework
        self.lstm_temporal_analyzer = lstm_temporal_analyzer
        self.bias_detector = cognitive_bias_detector
        self.conflict_resolver = conflict_resolver
        self.user_bias_history: Dict[int, List[Dict[str, Any]]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("UltimateBiasAwareManager initialized.")

    async def generate_bias_aware_recommendation(
        self,
        user_id: int,
        symbol: str,
        user_input: str,
        user_context: Dict[str, Any]
    ) -> dict:
        """
        Generates a final, bias-aware recommendation based on dual-pathway analysis.
        """
        self.logger.info(f"Starting bias-aware recommendation for '{symbol}' (user {user_id}).")

        # 1) Get market data
        # FIX: Added a default value to prevent crash if key is missing
        lookback_days = self.config.get("lookback_days", 100)
        market_data = await self.financial_data_framework.get_processed_data(symbol, lookback_days)
        
        if market_data.empty:
            self.logger.error(f"Cannot generate recommendation due to insufficient market data for {symbol}.")
            return self._generate_error_report("Insufficient market data.", symbol, user_id, user_input, user_context)

        last_price = market_data['Close'].iloc[-1]

        # 2) Bias detection
        bias_report = await self.bias_detector.analyze_input(user_input, user_id=user_id)
        self.logger.info(f"Bias detected: {bias_report.get('bias_detected')}")
        self.user_bias_history.setdefault(user_id, []).append(bias_report)

        # 3) Analytical pathway (mock)
        analytical = PathwayRecommendation(
            pathway_type=PathwayType.ANALYTICAL.value,
            action="BUY",
            confidence=float(np.random.uniform(0.5, 0.7)),
            target_price=last_price * 1.05,
            stop_loss=last_price * 0.95,
            position_size=float(np.random.uniform(0.1, 0.3)),
            reasoning="Mock analytical momentum.",
            technical_indicators={"RSI": 65, "MACD": "Bullish Crossover"},
            risk_score=0.4,
            timestamp=datetime.utcnow()
        )

        # 4) LSTM pathway
        sequence_length = self.config.get("ml_models", {}).get("sequence_length", 60)
        lstm_pred = await self.lstm_temporal_analyzer.generate_prediction(
            ticker=symbol,
            days=sequence_length + self.config.get("buffer_days", 10)
        )
        lstm = PathwayRecommendation(
            pathway_type=PathwayType.LSTM_TEMPORAL.value,
            action=lstm_pred['action'],
            confidence=float(lstm_pred['confidence']),
            target_price=last_price * 1.08,
            stop_loss=last_price * 0.92,
            position_size=float(lstm_pred.get('position_size', 0.0)),
            reasoning=lstm_pred.get('reasoning', ''),
            technical_indicators={},
            risk_score=float(lstm_pred.get('risk_score', 0.0)),
            timestamp=datetime.utcnow()
        )

        # 5) Conflict resolution
        final = self.conflict_resolver.resolve_recommendations(
            analytical_rec=analytical,
            lstm_rec=lstm,
            user_bias_report=bias_report,
            user_context=user_context
        )

        # 6) Assemble output
        output = {
            "user_id": user_id,
            "symbol": symbol,
            "user_input": user_input,
            "final_recommendation": asdict(final),
            "pathway_recommendations": {
                "analytical": asdict(analytical),
                "lstm_temporal": asdict(lstm)
            },
            "bias_analysis_report": bias_report
        }
        self.logger.info(
            f"Final action={output['final_recommendation']['final_action']} "
            f"with confidence={output['final_recommendation']['final_confidence']:.2f}"
        )
        return output

    def _generate_error_report(self, reason: str, symbol: str, user_id: int, user_input: str, user_context: Dict[str, Any]) -> dict:
        """Helper to create a standardized error report."""
        return {
            "user_id": user_id,
            "symbol": symbol,
            "user_input": user_input,
            "final_recommendation": {
                "final_action": "HOLD",
                "final_confidence": 0.0,
                "resolution_method": "Error handling",
                "resolved_reasoning": reason,
                "bias_impact": None
            },
            "pathway_recommendations": {},
            "bias_analysis_report": {}
        }
    
    def generate_bias_report(self, user_id: int) -> List[Dict[str, Any]]:
        return self.user_bias_history.get(user_id, [])

# --- Standalone Test Harness ---
if __name__ == "__main__":
    async def _test():
        logging.basicConfig(level=logging.INFO)
        # Mocks
        class MockFDF:
            async def get_processed_data(self, ticker, days_back):
                dates = pd.date_range(end=datetime.now(), periods=days_back)
                df = pd.DataFrame({
                    'Close': np.linspace(100, 110, days_back),
                    'Volume': np.random.randint(100000, 200000, days_back)
                }, index=dates)
                return df
        config = {
            'market_data': {'gold_ohlcv_symbol': 'XAU/USD'},
            'ml_models': {'sequence_length': 60},
            'lookback_days': 70
        }
        fdf = MockFDF()
        lstm = LSTMTemporalAnalysis(fdf, config)
        bias_det = CognitiveBiasDetector(Goldmind(''), config)
        resolver = DualSystemConflictResolver(config)
        mgr = UltimateBiasAwareManager(config, fdf, lstm, bias_det, resolver)
        
        user_id = 1
        symbol = 'XAU/USD'
        user_input = 'Test gold bias?'
        user_context = {'user_id': user_id, 'symbol': symbol}
        
        out = await mgr.generate_bias_aware_recommendation(
            user_id=user_id,
            symbol=symbol,
            user_input=user_input,
            user_context=user_context
        )
        print(json.dumps(out, indent=2, default=str))
    asyncio.run(_test())