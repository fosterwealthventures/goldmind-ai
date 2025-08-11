# app/models.py
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

@dataclass
class TraceStep:
    ts: str                 # ISO8601 string
    type: str               # "input" | "feature_importance" | "bias_check" | "decision" | ...
    detail: str

@dataclass
class Decision:
    action: str             # "BUY" | "SELL" | "HOLD" | etc.
    entry: float
    target: float
    pips: int

@dataclass
class Trace:
    id: str
    created_at: str         # ISO8601 string
    symbol: str             # e.g., "XAUUSD"
    decision: Optional[Decision]
    steps: List[TraceStep]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # dataclasses nested also get dict-ified; that’s fine for Firestore/JSON
        return d

def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
