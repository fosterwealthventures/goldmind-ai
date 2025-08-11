"""
dual_system_conflict_resolution.py

Core logic for GoldMIND AI's dual-system conflict resolution.
Defines the recommendation pathways and the strategy for resolving conflicts
between them, especially when cognitive bias is detected.
"""

import logging
from enum import Enum
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional
from datetime import datetime, timedelta # CORRECTED: Import datetime and timedelta

logger = logging.getLogger(__name__)

class PathwayType(Enum):
    """Enumeration of the different recommendation pathways."""
    ANALYTICAL = "analytical"
    LSTM_TEMPORAL = "lstm_temporal"
    QUANTITATIVE = "quantitative"
    SENTIMENT = "sentiment"
    
@dataclass
class PathwayRecommendation:
    """A data class to hold the output from a single recommendation pathway."""
    pathway_type: str
    action: str
    confidence: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    position_size: Optional[float]
    reasoning: str
    technical_indicators: Dict[str, Any]
    risk_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ConflictResolution:
    """A data class to hold the final, resolved recommendation."""
    final_action: str
    final_confidence: float
    resolution_method: str
    resolved_reasoning: str
    bias_impact: Optional[Dict] = None

class DualSystemConflictResolver:
    """
    Resolves conflicts and synthesizes recommendations from multiple pathways.
    It takes into account user cognitive biases to adjust the final output.
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.resolution_threshold = self.config.get("resolution_threshold", 0.1)
        self.bias_mitigation_factor = self.config.get("bias_mitigation_factor", 0.5)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("DualSystemConflictResolver initialized.")

    def resolve_recommendations(self, 
                                analytical_rec: PathwayRecommendation, 
                                lstm_rec: PathwayRecommendation, 
                                user_bias_report: Dict, 
                                user_context: Dict) -> ConflictResolution:
        """
        Synthesizes the final recommendation based on the outputs of the dual systems and user bias.
        
        :param analytical_rec: Recommendation from the analytical pathway.
        :param lstm_rec: Recommendation from the LSTM pathway.
        :param user_bias_report: The report from the cognitive bias detector.
        :param user_context: User-specific context data.
        :return: A ConflictResolution data class with the final decision.
        """
        self.logger.info("Starting conflict resolution process.")
        
        # Check for user bias and its confidence
        bias_detected = user_bias_report.get("bias_detected", False)
        bias_confidence = user_bias_report.get("confidence", 0.0)

        # Simple conflict detection logic
        if analytical_rec.action != lstm_rec.action:
            self.logger.warning(f"Conflict detected! Analytical: {analytical_rec.action}, LSTM: {lstm_rec.action}")
            if bias_detected and bias_confidence > 0.6:
                # If a strong bias is detected, we'll favor the pathway that
                # contradicts the user's likely biased inclination.
                # Here, we assume the LSTM is more objective.
                resolved_action = lstm_rec.action
                resolved_confidence = lstm_rec.confidence * (1 - self.bias_mitigation_factor)
                resolution_method = f"Bias-aware override (favored {lstm_rec.pathway_type})"
                resolved_reasoning = f"Conflicting recommendations detected. Overriding to {lstm_rec.pathway_type} recommendation due to detected cognitive bias in user input: {user_bias_report.get('bias_type')}."
                
                return ConflictResolution(
                    final_action=resolved_action,
                    final_confidence=float(resolved_confidence),
                    resolution_method=resolution_method,
                    resolved_reasoning=resolved_reasoning,
                    bias_impact=user_bias_report
                )
            else:
                # Simple confidence-based resolution if no bias or weak bias
                if analytical_rec.confidence > lstm_rec.confidence:
                    resolved_action = analytical_rec.action
                    resolved_confidence = analytical_rec.confidence
                    resolution_method = "Confidence-based resolution"
                    resolved_reasoning = f"Conflicting recommendations detected. Final recommendation is based on the higher confidence score from the {analytical_rec.pathway_type} system."
                else:
                    resolved_action = lstm_rec.action
                    resolved_confidence = lstm_rec.confidence
                    resolution_method = "Confidence-based resolution"
                    resolved_reasoning = f"Conflicting recommendations detected. Final recommendation is based on the higher confidence score from the {lstm_rec.pathway_type} system."

                return ConflictResolution(
                    final_action=resolved_action,
                    final_confidence=float(resolved_confidence),
                    resolution_method=resolution_method,
                    resolved_reasoning=resolved_reasoning
                )
        else:
            # If both systems agree
            self.logger.info("Dual systems are in agreement.")
            resolved_action = analytical_rec.action # They are the same, so pick one
            # Use a weighted average of confidence, giving a boost for agreement
            resolved_confidence = (analytical_rec.confidence + lstm_rec.confidence) / 2.0
            resolved_confidence = min(1.0, resolved_confidence * 1.1) # Boost for agreement
            resolution_method = "Consensus"
            resolved_reasoning = f"Both dual systems are in consensus on the '{resolved_action}' action. Confidence is boosted for this agreement."
            
            # If bias is detected but there's no conflict, we still log it but don't change the action
            bias_impact = user_bias_report if bias_detected else None

            return ConflictResolution(
                final_action=resolved_action,
                final_confidence=float(resolved_confidence),
                resolution_method=resolution_method,
                resolved_reasoning=resolved_reasoning,
                bias_impact=bias_impact
            )