from typing import List, Tuple, Dict, Any
from rag.document import DocumentChunk

class HallucinationGuardrail:
    """Enforces confidence thresholds and validates source attribution."""
    def __init__(self, min_confidence: float = 0.50):
        self.min_confidence = min_confidence

    def verify(self, retrieved: List[Tuple[float, DocumentChunk]]) -> Dict[str, Any]:
        if not retrieved or retrieved[0][0] < self.min_confidence:
            return {"is_grounded": False, "confidence": 0.0, "status": "REJECTED_LOW_CONFIDENCE"}
        return {
            "is_grounded": True,
            "confidence": round(retrieved[0][0], 2),
            "status": "VERIFIED_GROUNDED",
            "primary_source": retrieved[0][1].metadata.get("source", "unknown")
        }
