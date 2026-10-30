from rag.document import DocumentChunk
from rag.retrievers.hybrid import HybridRetriever
from rag.guards.hallucination_guard import HallucinationGuardrail
from rag.agent import EnterpriseRAGAgent
import json

def create_agent() -> EnterpriseRAGAgent:
    retriever = HybridRetriever()
    retriever.add_documents([
        DocumentChunk("1", "FastAPI async IO handles microsecond financial orders.", {"source": "fastapi_perf.pdf"}),
        DocumentChunk("2", "PostgreSQL partitioned tables scale multi-tenant telemetry.", {"source": "postgres_scale.pdf"}),
        DocumentChunk("3", "Redis caching achieves sub-millisecond response for order book depth.", {"source": "redis_depth.pdf"}),
    ])
    guard = HallucinationGuardrail(min_confidence=0.30)
    return EnterpriseRAGAgent(retriever, guard)

if __name__ == '__main__':
    agent = create_agent()
    res = agent.ask("FastAPI async orders")
    print(json.dumps(res, indent=2))
