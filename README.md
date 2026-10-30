# Enterprise RAG & Autonomous Agent System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![RAG](https://img.shields.io/badge/RAG-Hybrid%20Retrieval-orange.svg)]()
[![Guardrails](https://img.shields.io/badge/Hallucination-Shielded-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deterministic enterprise document intelligence pipeline with hybrid retrieval (Lexical BM25 + Dense Vector Semantic) and real-time hallucination guardrails.

## Architecture

```
[User Query] 
      |
      v
[Hybrid Retriever] ---> (Dense Vector + Lexical BM25)
      |
      v
[Hallucination Guardrail] ---> (Confidence & Source Attestation)
      |
      +---> [Approved Grounded Response] (Source Citations)
      +---> [Rejection Fallback] (Prevents False Information)
```

## Verification & Tests

```bash
pytest tests/
python main.py
```

## Author
**Alireza Ghadiri** - AI Systems & LLM Integrations Engineer.
