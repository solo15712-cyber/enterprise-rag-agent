import pytest
from rag.document import DocumentChunk
from rag.retrievers.hybrid import HybridRetriever
from rag.guards.hallucination_guard import HallucinationGuardrail
from rag.agent import EnterpriseRAGAgent

def test_rag_pipeline_retrieval():
    retriever = HybridRetriever()
    retriever.add_documents([
        DocumentChunk("1", "LangChain orchestrates autonomous agent toolchains.", {"source": "langchain.pdf"}),
    ])
    guard = HallucinationGuardrail(min_confidence=0.30)
    agent = EnterpriseRAGAgent(retriever, guard)
    res = agent.ask("LangChain toolchains")
    assert res["guardrail"]["is_grounded"] is True
    assert "langchain.pdf" in res["sources"]

def test_rag_hallucination_rejection():
    retriever = HybridRetriever()
    guard = HallucinationGuardrail(min_confidence=0.50)
    agent = EnterpriseRAGAgent(retriever, guard)
    res = agent.ask("Quantum cryptography teleportation")
    assert res["guardrail"]["is_grounded"] is False
    assert "cannot verify" in res["answer"].lower()

def test_multi_document_ranking():
    retriever = HybridRetriever()
    retriever.add_documents([
        DocumentChunk("c1", "Python FastAPI high throughput engine", {"source": "p1.pdf"}),
        DocumentChunk("c2", "Python FastAPI microsecond order execution matching engine", {"source": "p2.pdf"}),
    ])
    guard = HallucinationGuardrail(min_confidence=0.20)
    agent = EnterpriseRAGAgent(retriever, guard)
    res = agent.ask("microsecond order execution")
    assert res["guardrail"]["is_grounded"] is True
    assert res["sources"][0] == "p2.pdf"

def test_guardrail_confidence_thresholding():
    retriever = HybridRetriever()
    retriever.add_documents([
        DocumentChunk("c1", "Iranian commercial contract regulations", {"source": "law.pdf"}),
    ])
    strict_guard = HallucinationGuardrail(min_confidence=0.99)
    agent = EnterpriseRAGAgent(retriever, strict_guard)
    res = agent.ask("contract regulations")
    # Confidence is 2/2 = 1.0 -> passes even strict guard
    assert res["guardrail"]["is_grounded"] is True

    # Partial overlap confidence will be 1/3 = 0.33 -> rejected under strict 0.99
    res_partial = agent.ask("contract tax calculation")
    assert res_partial["guardrail"]["is_grounded"] is False
