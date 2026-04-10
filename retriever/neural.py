"""
retriever.neural
================
Neural bi-encoder retriever using ChromaDB ANN search.

The bi-encoder architecture:
  - Documents (code functions) are pre-encoded OFFLINE by the indexer (M3)
    and stored as 768-dim vectors in ChromaDB.
  - At query time, ONLY the query is encoded (by the embedder, M2).
  - ANN search finds the top-K nearest document vectors in O(log N) time.

This is the primary retrieval method for CodeLens. The BM25Retriever
(bm25.py) provides a keyword baseline for comparison.

Usage
-----
    from indexer import CodeLensIndex
    from retriever import NeuralRetriever

    index = CodeLensIndex(persist_dir="chroma_db/")
    retriever = NeuralRetriever(index, top_k=20)

    from embedder.embed_pipeline import EmbeddedQuery
    import numpy as np
    query = EmbeddedQuery(...)
    results = retriever.search(query)
    # returns list[RetrievalResult], sorted by descending score
"""

import logging
from typing import List

import numpy as np

from indexer.chroma_index import CodeLensIndex
from embedder.embed_pipeline import EmbeddedQuery
from .models import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 20


class NeuralRetriever:
    """
    Retrieves top-K code functions using ANN vector search (ChromaDB/HNSW).

    Parameters
    ----------
    index : CodeLensIndex
        A populated and ready-to-search ChromaDB index.
    top_k : int
        Number of candidates to retrieve. Default 20 (for reranker in M5).
    """

    def __init__(self, index: CodeLensIndex, top_k: int = DEFAULT_TOP_K) -> None:
        if not isinstance(index, CodeLensIndex):
            raise TypeError(f"Expected CodeLensIndex, got {type(index)}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        self._index = index
        self._top_k = top_k
        logger.info(
            "NeuralRetriever ready. Index: '%s' (%d vectors), top_k=%d",
            index.collection_name, index.count(), top_k,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: EmbeddedQuery) -> List[RetrievalResult]:
        """
        Search the index using the query's fused vector.

        Parameters
        ----------
        query : EmbeddedQuery
            Output of embedder.embed_query(). Uses query.fused_vector.

        Returns
        -------
        list[RetrievalResult]
            Top-K results sorted by descending cosine similarity.
            Empty list if the index has no vectors.
        """
        if not isinstance(query, EmbeddedQuery):
            raise TypeError(f"Expected EmbeddedQuery, got {type(query)}")

        if self._index.count() == 0:
            logger.warning("Index is empty — returning no results.")
            return []

        hits = self._index.search(query.fused_vector, top_k=self._top_k)

        results = [
            RetrievalResult.from_chroma_hit(hit, rank=i + 1, method="neural")
            for i, hit in enumerate(hits)
        ]

        logger.debug(
            "NeuralRetriever: query=%r → %d results (top score=%.4f)",
            query.text[:50],
            len(results),
            results[0].retrieval_score if results else 0.0,
        )
        return results

    def search_vector(self, vector: np.ndarray) -> List[RetrievalResult]:
        """
        Search using a raw numpy vector directly (bypasses EmbeddedQuery).
        Useful for evaluation scripts and ablation studies.

        Parameters
        ----------
        vector : np.ndarray, shape (768,)
            L2-normalised query vector.

        Returns
        -------
        list[RetrievalResult]
        """
        if vector.ndim != 1:
            raise ValueError(f"Expected 1-D vector, got shape {vector.shape}")

        if self._index.count() == 0:
            return []

        hits = self._index.search(vector, top_k=self._top_k)
        return [
            RetrievalResult.from_chroma_hit(hit, rank=i + 1, method="neural")
            for i, hit in enumerate(hits)
        ]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def top_k(self) -> int:
        return self._top_k

    @top_k.setter
    def top_k(self, value: int) -> None:
        if value < 1:
            raise ValueError(f"top_k must be >= 1, got {value}")
        self._top_k = value

    @property
    def index_size(self) -> int:
        return self._index.count()

    def __repr__(self) -> str:
        return (
            f"NeuralRetriever(collection={self._index.collection_name!r}, "
            f"index_size={self.index_size}, top_k={self._top_k})"
        )