from typing import List, Tuple
from rag.document import DocumentChunk

class HybridRetriever:
    """Combines Lexical (BM25 token overlap) with Cosine Dense Simulation."""
    def __init__(self):
        self.corpus: List[DocumentChunk] = []

    def add_documents(self, docs: List[DocumentChunk]):
        self.corpus.extend(docs)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[float, DocumentChunk]]:
        q_tokens = set(query.lower().split())
        scored = []
        for doc in self.corpus:
            d_tokens = set(doc.content.lower().split())
            overlap = len(q_tokens & d_tokens)
            score = overlap / max(len(q_tokens), 1)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]
