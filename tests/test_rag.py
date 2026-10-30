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
