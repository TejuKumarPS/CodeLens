"""
retriever — CodeLens Module 4
==============================
Responsible for:
  - Taking an EmbeddedQuery and returning top-K RetrievalResult objects
  - Neural retrieval: ANN search via CodeLensIndex (ChromaDB/HNSW)
  - Keyword baseline: BM25 over tokenised function code
  - Unified retrieve() function consumed by the reranker (M5)

Public API:
  RetrievalResult          — typed result dataclass
  NeuralRetriever          — ANN-based retrieval via CodeLensIndex
  BM25Retriever            — keyword-based baseline retriever
  retrieve()               — convenience wrapper: embed query → ANN search
"""

from .models import RetrievalResult
from .neural import NeuralRetriever
from .bm25 import BM25Retriever
from .retrieve import retrieve

__all__ = [
    "RetrievalResult",
    "NeuralRetriever",
    "BM25Retriever",
    "retrieve",
]