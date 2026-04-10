
"""
retriever.bm25
==============
BM25 keyword retriever used as a baseline for comparing against neural
retrieval. BM25 is a standard IR weighting scheme (TF-IDF variant) that
operates purely on token overlap — no embeddings needed.

Role in CodeLens:
  - Provides a keyword search fallback
  - Enables ablation: "how much does neural retrieval improve over BM25?"
  - Evaluation module (M7) computes MRR/NDCG for both methods side-by-side

BM25 index is built in-memory from a list of CodeRecord objects.
It is NOT persisted — rebuild it on startup from the Parquet file (fast,
<1s for 5K records).

Library: rank-bm25 (pip install rank-bm25)

Usage
-----
    from data_loader import load_processed
    from retriever import BM25Retriever

    records = load_processed("data/processed/python_train.parquet")
    retriever = BM25Retriever(records)

    results = retriever.search("app crashes at checkout", top_k=20)
"""

import logging
import re
from typing import List

from data_loader.models import CodeRecord
from .models import RetrievalResult

logger = logging.getLogger(__name__)


def _tokenize_query(text: str) -> List[str]:
    """
    Simple whitespace + punctuation tokenizer for BM25 queries.
    Lowercases and splits on non-alphanumeric characters.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """
    BM25 keyword retriever over a corpus of CodeRecord objects.

    The BM25 index is built over func_code_tokens (pre-tokenised code)
    combined with the docstring tokens. This gives BM25 access to both
    identifier names and natural language descriptions.

    Parameters
    ----------
    records : list[CodeRecord]
        The full corpus. tokens field must be populated (always is after
        data_loader.process_split).
    """

    def __init__(self, records: List[CodeRecord]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError("pip install rank-bm25") from e

        if not records:
            raise ValueError("BM25Retriever requires at least one record.")

        self._records = records

        # Build token corpus: code tokens + lowercased docstring tokens
        corpus = []
        for r in records:
            doc_tokens = _tokenize_query(r.docstring)
            combined = [t.lower() for t in r.tokens] + doc_tokens
            corpus.append(combined)

        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(corpus)
        logger.info("BM25Retriever built over %d documents.", len(records))

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query_text: str, top_k: int = 20) -> List[RetrievalResult]:
        """
        Retrieve top-K records by BM25 score for a text query.

        Parameters
        ----------
        query_text : str
            Natural language bug report or keyword query.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[RetrievalResult]
            Top-K results sorted by descending BM25 score.
        """
        if not query_text or not query_text.strip():
            raise ValueError("query_text must be a non-empty string.")

        query_tokens = _tokenize_query(query_text)
        if not query_tokens:
            logger.warning("Query tokenised to empty list: %r", query_text)
            return []

        scores = self._bm25.get_scores(query_tokens)  # shape: (N,)

        # Get top-K indices sorted by descending score
        top_k_clamped = min(top_k, len(self._records))
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k_clamped]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            record = self._records[idx]
            score = float(scores[idx])

            # Build a hit dict compatible with RetrievalResult.from_chroma_hit
            hit = {
                "id": record.id,
                "score": score,
                "func_name": record.func_name,
                "repository": record.repository,
                "url": record.url,
                "language": record.language,
                "docstring_preview": record.docstring[:200],
                "code_preview": record.func_code[:300],
                "document": record.func_code[:500],
            }
            results.append(
                RetrievalResult.from_chroma_hit(hit, rank=rank, method="bm25")
            )

        return results

    @property
    def corpus_size(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"BM25Retriever(corpus_size={self.corpus_size})"