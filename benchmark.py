import time
import statistics
from typing import List, Tuple
from rag.document import DocumentChunk
from rag.retrievers.hybrid import HybridRetriever
from rag.guards.hallucination_guard import HallucinationGuardrail
from rag.agent import EnterpriseRAGAgent

def build_enterprise_corpus() -> List[DocumentChunk]:
    return [
        DocumentChunk("doc_01", "FastAPI asynchronous event loops deliver microsecond order execution in financial trading platforms.", {"source": "fastapi_highload.pdf"}),
        DocumentChunk("doc_02", "PostgreSQL declarative range partitioning enables horizontal scaling of multi-tenant transactional ledgers.", {"source": "postgres_scale.pdf"}),
        DocumentChunk("doc_03", "Redis cluster caching provides sub-millisecond query response for dynamic order book depth.", {"source": "redis_telemetry.pdf"}),
        DocumentChunk("doc_04", "Iranian commercial law article 338 defines sales contracts and reciprocal obligations of vendor and buyer.", {"source": "iran_civil_code_contracts.pdf"}),
        DocumentChunk("doc_05", "Cyber crime law article 753 penalizes unauthorized access and digital fraud with strict statutory liabilities.", {"source": "iran_cyber_regulations.pdf"}),
        DocumentChunk("doc_06", "Hybrid BM25 and dense semantic embeddings mitigate keyword mismatch and semantic drift in legal RAG.", {"source": "hybrid_search_whitepaper.pdf"}),
        DocumentChunk("doc_07", "Source attribution guardrails filter unverifiable LLM outputs to guarantee zero hallucination.", {"source": "enterprise_guardrails.pdf"}),
        DocumentChunk("doc_08", "AsyncIO Celery task queues decouple computationally heavy risk calculations from user-facing REST APIs.", {"source": "async_worker_architecture.pdf"}),
    ]

def run_benchmark():
    retriever = HybridRetriever()
    corpus = build_enterprise_corpus()
    retriever.add_documents(corpus)
    guard = HallucinationGuardrail(min_confidence=0.25)
    agent = EnterpriseRAGAgent(retriever, guard)

    grounded_queries = [
        ("FastAPI microsecond order execution", True),
        ("PostgreSQL declarative range partitioning", True),
        ("Redis sub-millisecond order book", True),
        ("Iranian commercial law sales contract", True),
        ("Cyber crime digital fraud penalties", True),
        ("BM25 dense semantic embeddings hybrid", True),
        ("Source attribution guardrails hallucination", True),
        ("AsyncIO Celery risk task queues", True),
        ("Transactional ledgers multi-tenant scaling", True),
        ("Vendor buyer reciprocal obligations", True),
        ("Keyword mismatch semantic drift legal RAG", True),
        ("Unverifiable LLM outputs filtering", True),
    ]

    hallucination_queries = [
        ("Quantum teleportation wormhole astrophysics", False),
        ("Martian subterranean colonization terraforming", False),
        ("Ancient Roman gladiator chariot recipes", False),
        ("Medieval alchemical transmutation of lead into gold", False),
        ("Intergalactic warp drive propulsion antimatter", False),
        ("Extraterrestrial cryptographic linguistics decipherment", False),
    ]

    all_tests = grounded_queries + hallucination_queries
    latencies_us = []
    correct_evaluations = 0
    rejected_hallucinations = 0

    print(f"[BENCHMARK] Executing Enterprise Hybrid RAG Evaluation ({len(all_tests)} queries across {len(corpus)} docs)...")
    
    for query, expected_grounded in all_tests:
        t0 = time.perf_counter_ns()
        res = agent.ask(query)
        lat = (time.perf_counter_ns() - t0) / 1000.0
        latencies_us.append(lat)

        is_grounded = res["guardrail"]["is_grounded"]
        if is_grounded == expected_grounded:
            correct_evaluations += 1
        if not expected_grounded and not is_grounded:
            rejected_hallucinations += 1

    accuracy = (correct_evaluations / len(all_tests)) * 100.0
    hallucination_rejection_rate = (rejected_hallucinations / len(hallucination_queries)) * 100.0
    latencies_us.sort()
    p50 = statistics.median(latencies_us)
    p95 = latencies_us[int(len(latencies_us) * 0.95)]
    p99 = latencies_us[-1]

    print(f"[RESULTS] Grounded Accuracy: {accuracy:.1f}% | Hallucination Rejection: {hallucination_rejection_rate:.1f}%")
    print(f"[LATENCY] Retrieval & Guardrail p50: {p50:.2f}us | p95: {p95:.2f}us | p99: {p99:.2f}us")
    return accuracy >= 90.0 and hallucination_rejection_rate == 100.0

if __name__ == '__main__':
    run_benchmark()
