"""
reranker — CodeLens Module 5
=============================
Responsible for:
  - Taking top-K RetrievalResult candidates from the retriever (M4)
  - Scoring each (query_text, code) pair with a cross-encoder model
  - Returning a re-ranked, truncated list with rerank_score set
  - Cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2

Public API:
  CrossEncoderReranker     — wraps the HuggingFace cross-encoder
  rerank()                 — convenience: rerank candidates, return top-N
"""

from .cross_encoder import CrossEncoderReranker
from .rerank import rerank

__all__ = [
    "CrossEncoderReranker",
    "rerank",
]