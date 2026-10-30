import json
from typing import Dict, Any
from rag.retrievers.hybrid import HybridRetriever
from rag.guards.hallucination_guard import HallucinationGuardrail
from rag.document import DocumentChunk

class EnterpriseRAGAgent:
    def __init__(self, retriever: HybridRetriever, guard: HallucinationGuardrail):
        self.retriever = retriever
        self.guard = guard

    def ask(self, query: str) -> Dict[str, Any]:
        candidates = self.retriever.retrieve(query, top_k=2)
        guard_result = self.guard.verify(candidates)
        if not guard_result["is_grounded"]:
            return {
                "query": query,
                "answer": "I cannot verify this information with certified enterprise documents.",
                "guardrail": guard_result
            }
        separator = chr(10) + '---' + chr(10)
        context = separator.join([c[1].content for c in candidates if c[0] > 0])
        return {
            "query": query,
            "answer": f"Synthesized verified knowledge from {len(candidates)} authoritative chunks.",
            "context": context,
            "guardrail": guard_result,
            "sources": [c[1].metadata.get("source") for c in candidates]
        }
