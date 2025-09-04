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
            )"""
dual_system_conflict_resolution.py — GoldMIND AI
------------------------------------------------
Resolves conflicts between the Analytical Pathway (rules/technicals) and
the LSTM Temporal Analysis pathway, blending in Market Regime and (optionally)
AdaptiveLearningSystem weights to produce a single, dashboard-ready decision.

Design goals:
- Deterministic, explainable tie‑break rules with weightable blending
- Works even if some inputs are missing (graceful degradation)
- Clean shapes that match your new dashboard
- Optional bias-aware nudge hook (e.g., integrate CognitiveBiasDetector upstream)

Public surface:
    resolver = DualSystemConflictResolver(config)
    unified = resolver.resolve(
        analytical={"action":"BUY","confidence":0.8,"target_price":...,"reasoning":"..."},
        lstm={"action":"SELL","confidence":0.62,"target_price":...,"reasoning":"..."},
        regime={"regime":"trending","sentiment":"bullish","confidence":0.66},
        adaptive_weights={"ANALYTICAL":0.55,"LSTM_TEMPORAL":0.45},
    )
    card = DualSystemConflictResolver.to_recommendation_card(unified, symbol="XAU/USD")
    summary = DualSystemConflictResolver.to_summary_block(unified, symbol="XAU/USD")

All inputs are ordinary dicts (no hard imports). Keys used (best-effort):
- recommendation dicts: {"action","confidence","target_price","stop_loss","reasoning","risk_score","position_size"}
- regime dict: {"regime","sentiment","confidence"}
- adaptive_weights dict: e.g. {"ANALYTICAL":0.6,"LSTM_TEMPORAL":0.4}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

log = logging.getLogger("goldmind.dual_resolver")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

@dataclass
class ResolverConfig:
    # Base weights if AdaptiveLearningSystem not provided
    w_analytical: float = 0.5
    w_lstm: float = 0.5
    # Confidence floor/ceil
    min_confidence: float = 0.5
    max_confidence: float = 0.95
    # Extra nudges
    regime_trend_boost: float = 0.08      # boost when regime sentiment aligns
    regime_sideways_penalty: float = 0.05 # reduce confidence when sideways/uncertain
    max_target_deviation: float = 0.12    # cap relative divergence between targets
    # Final safety clamps
    position_size_default: float = 0.15
    position_size_max: float = 0.35
    position_size_min: float = 0.05


# ---------------------------------------------------------
# Resolver
# ---------------------------------------------------------

class DualSystemConflictResolver:
    def __init__(self, config: Dict[str, Any] | None = None):
        self.cfg = ResolverConfig(**(config or {}))

    # -------------- Public API --------------

    def resolve(
        self,
        analytical: Optional[Dict[str, Any]] = None,
        lstm: Optional[Dict[str, Any]] = None,
        regime: Optional[Dict[str, Any]] = None,
        adaptive_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Blend/resolve pathway outputs into a unified decision.
        """
        # Normalize present inputs
        a = self._norm_rec(analytical)
        l = self._norm_rec(lstm)
        r = self._norm_regime(regime)
        w = self._norm_weights(adaptive_weights)

        # If only one pathway is present, return it (with regime adjustments)
        if a and not l:
            out = self._apply_regime(a.copy(), r, origin="analytical_only")
            out["sources"] = {"analytical": a, "lstm": None, "regime": r, "weights": w}
            return out
        if l and not a:
            out = self._apply_regime(l.copy(), r, origin="lstm_only")
            out["sources"] = {"analytical": None, "lstm": l, "regime": r, "weights": w}
            return out
        if not a and not l:
            return self._empty("No pathway inputs available.", r, w)

        # Core blending: compute signed scores for BUY/SELL, HOLD as 0
        score_a = self._signed_score(a)
        score_l = self._signed_score(l)

        # Weighted sum using adaptive or default weights
        wa = float(w.get("ANALYTICAL", self.cfg.w_analytical))
        wl = float(w.get("LSTM_TEMPORAL", self.cfg.w_lstm))
        if wa + wl <= 0:
            wa, wl = 0.5, 0.5
        # normalize
        s = wa + wl
        wa, wl = wa / s, wl / s

        blended_score = wa * score_a + wl * score_l

        decision, base_conf = self._decision_from_score(blended_score)

        # Build recommendation scaffold
        rec = {
            "action": decision,
            "confidence": base_conf,
            "target_price": self._blend_numeric(a.get("target_price"), l.get("target_price")),
            "stop_loss": self._blend_numeric(a.get("stop_loss"), l.get("stop_loss")),
            "position_size": self.cfg.position_size_default,
            "reasoning": "",
            "risk_score": self._blend_numeric(a.get("risk_score"), l.get("risk_score"), default=0.5),
        }

        reasons: List[str] = []
        reasons.append(f"Weighted blend: analytical={wa:.2f}*{score_a:+.2f}, lstm={wl:.2f}*{score_l:+.2f} → {blended_score:+.2f}")
        reasons.append(f"Initial action={decision} (confidence {base_conf:.2f}).")

        # Regime-aware adjustments
        rec = self._apply_regime(rec, r, reasons)

        # Cap extreme target divergence for safety
        rec["target_price"] = self._cap_target_deviation(rec["target_price"], a.get("target_price"), l.get("target_price"))
        # Clamp confidence & position size
        rec["confidence"] = float(max(self.cfg.min_confidence, min(self.cfg.max_confidence, rec["confidence"])))
        rec["position_size"] = float(min(self.cfg.position_size_max, max(self.cfg.position_size_min, rec["position_size"])))

        rec["reasoning"] = " ".join(reasons).strip()
        rec["sources"] = {"analytical": a, "lstm": l, "regime": r, "weights": {"ANALYTICAL": wa, "LSTM_TEMPORAL": wl}}
        return rec

    # -------------- Dashboard helpers --------------

    @staticmethod
    def to_recommendation_card(rec: Dict[str, Any], symbol: str = "") -> Dict[str, Any]:
        return {
            "action": str(rec.get("action","HOLD")).upper(),
            "confidence": float(rec.get("confidence", 0.5)),
            "entry": None,
            "target": float(rec.get("target_price", 0.0) or 0.0),
            "note": rec.get("reasoning",""),
            "symbol": symbol,
        }

    @staticmethod
    def to_summary_block(rec: Dict[str, Any], symbol: str = "") -> Dict[str, Any]:
        action = str(rec.get("action","HOLD")).upper()
        conf = float(rec.get("confidence", 0.5))
        risk = float(rec.get("risk_score", 0.5))
        regime_hint = "Trend" if action in {"BUY","SELL"} and conf >= 0.65 else "Neutral"
        vol = "High" if risk >= 0.7 else "Moderate" if risk >= 0.4 else "Low"
        return {"regime": regime_hint, "volatility": vol, "note": rec.get("reasoning",""), "symbol": symbol}

    # -------------- Internals --------------

    def _norm_rec(self, x: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not x or not isinstance(x, dict):
            return None
        y = {**x}
        y["action"] = str(y.get("action","HOLD")).upper()
        y["confidence"] = float(y.get("confidence", 0.5))
        return y

    def _norm_regime(self, r: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not r or not isinstance(r, dict):
            return {"regime":"unknown","sentiment":"unknown","confidence":0.0}
        return {"regime": str(r.get("regime","unknown")).lower(),
                "sentiment": str(r.get("sentiment","unknown")).lower(),
                "confidence": float(r.get("confidence", 0.0))}

    def _norm_weights(self, w: Optional[Dict[str, float]]) -> Dict[str, float]:
        if not w or not isinstance(w, dict):
            return {"ANALYTICAL": self.cfg.w_analytical, "LSTM_TEMPORAL": self.cfg.w_lstm}
        return {"ANALYTICAL": float(w.get("ANALYTICAL", self.cfg.w_analytical)),
                "LSTM_TEMPORAL": float(w.get("LSTM_TEMPORAL", self.cfg.w_lstm))}

    def _signed_score(self, rec: Dict[str, Any]) -> float:
        m = {"BUY": +1.0, "SELL": -1.0, "HOLD": 0.0}
        return float(m.get(rec.get("action","HOLD"), 0.0)) * float(rec.get("confidence", 0.5))

    def _decision_from_score(self, s: float) -> tuple[str, float]:
        # Soft thresholds around zero to reduce flip-flops
        if s > +0.08:
            return "BUY", min(0.9, 0.6 + s)
        if s < -0.08:
            return "SELL", min(0.9, 0.6 + abs(s))
        return "HOLD", 0.55

    def _apply_regime(self, rec: Dict[str, Any], regime: Dict[str, Any], reasons: Optional[List[str]] = None, origin: Optional[str] = None) -> Dict[str, Any]:
        reasons = reasons if reasons is not None else []
        if origin:
            reasons.append(f"Single pathway used: {origin}.")

        sentiment = regime.get("sentiment","unknown")
        r_conf = float(regime.get("confidence", 0.0))

        # Boost when aligned with clear trend; reduce when sideways/uncertain
        if regime.get("regime") == "trending" and sentiment in {"bullish","bearish"}:
            aligned = ((rec.get("action") == "BUY" and sentiment == "bullish") or
                       (rec.get("action") == "SELL" and sentiment == "bearish"))
            if aligned:
                bump = self.cfg.regime_trend_boost * (0.5 + r_conf)
                rec["confidence"] = float(rec.get("confidence", 0.5) + bump)
                reasons.append(f"Regime alignment boost (+{bump:.2f}).")
        elif regime.get("regime") in {"sideways","volatile"} or sentiment in {"neutral","uncertain"}:
            rec["confidence"] = float(rec.get("confidence", 0.5) - self.cfg.regime_sideways_penalty * (0.5 + r_conf))
            reasons.append("Confidence reduced due to sideways/uncertain regime.")

        # Adjust position sizing slightly with regime
        if regime.get("regime") == "volatile":
            rec["position_size"] = float(max(self.cfg.position_size_min, rec.get("position_size", self.cfg.position_size_default) * 0.8))
        elif regime.get("regime") == "trending":
            rec["position_size"] = float(min(self.cfg.position_size_max, rec.get("position_size", self.cfg.position_size_default) * 1.1))

        return rec

    def _blend_numeric(self, a: Optional[float], b: Optional[float], default: Optional[float] = None) -> Optional[float]:
        vals = [v for v in [a, b] if isinstance(v, (int, float))]
        if not vals:
            return default
        return float(sum(vals) / len(vals))

    def _cap_target_deviation(self, blended: Optional[float], a: Optional[float], b: Optional[float]) -> Optional[float]:
        if blended is None or not isinstance(blended, (int,float)) or not all(isinstance(x,(int,float)) for x in (a,b) if x is not None):
            return blended
        # If both targets exist and diverge excessively, pull towards current blended
        try:
            arr = [x for x in (a,b) if isinstance(x,(int,float))]
            if not arr or blended == 0:
                return blended
            avg = sum(arr)/len(arr)
            dev = abs(blended - avg) / max(1e-9, avg)
            if dev > self.cfg.max_target_deviation:
                return avg + (blended - avg) * (self.cfg.max_target_deviation / dev)
        except Exception:
            pass
        return blended

    def _empty(self, msg: str, regime: Dict[str, Any], w: Dict[str, float]) -> Dict[str, Any]:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "target_price": None,
            "stop_loss": None,
            "position_size": self.cfg.position_size_default,
            "reasoning": msg,
            "risk_score": 0.5,
            "sources": {"analytical": None, "lstm": None, "regime": regime, "weights": w},
        }


# ---------------- Demo ----------------
if __name__ == "__main__":
    analytical = {"action": "BUY", "confidence": 0.78, "target_price": 110.0, "stop_loss": 101.0, "reasoning": "Golden Cross."}
    lstm = {"action": "SELL", "confidence": 0.62, "target_price": 104.0, "stop_loss": 112.0, "reasoning": "Downturn predicted."}
    regime = {"regime": "trending", "sentiment": "bullish", "confidence": 0.65}
    weights = {"ANALYTICAL": 0.55, "LSTM_TEMPORAL": 0.45}

    resolver = DualSystemConflictResolver({})
    unified = resolver.resolve(analytical, lstm, regime, weights)
    print("Unified:", unified)
    print("Card:", DualSystemConflictResolver.to_recommendation_card(unified, "XAU/USD"))
    print("Summary:", DualSystemConflictResolver.to_summary_block(unified, "XAU/USD"))
