
"""
reranker.rerank
===============
Orchestrates the two-stage retrieval pipeline:

  Stage 1 (M4): Bi-encoder ANN → top-20 candidates  (fast, approximate)
  Stage 2 (M5): Cross-encoder  → re-scored top-5    (slow, precise)

The rerank() function is the primary entry point consumed by:
  - The FastAPI backend (M8): called per user request
  - The evaluator (M7): called per query in the evaluation set

Pipeline:
  candidates: list[RetrievalResult]  ← from retriever.retrieve()
      │
      ├── build (query, code) pairs
      ├── score_pairs() via CrossEncoderReranker
      ├── attach rerank_score to each result
      ├── sort by rerank_score descending
      ├── update rank fields
      └── return top_n results

Usage
-----
    from retriever import retrieve
    from reranker import CrossEncoderReranker, rerank

    # One-time setup
    encoder  = CodeBERTEncoder()
    reranker = CrossEncoderReranker()
    index    = CodeLensIndex("chroma_db/")

    # Per-request pipeline
    candidates = retrieve("null pointer in payment", index, encoder, top_k=20)
    results    = rerank("null pointer in payment", candidates, reranker, top_n=5)

    for r in results:
        print(r.rank, r.func_name, r.rerank_score)
"""

import logging
from typing import List, Optional

from retriever.models import RetrievalResult
from .cross_encoder import CrossEncoderReranker

logger = logging.getLogger(__name__)

DEFAULT_TOP_N = 5


def rerank(
    query_text: str,
    candidates: List[RetrievalResult],
    reranker: CrossEncoderReranker,
    top_n: int = DEFAULT_TOP_N,
    code_field: str = "document",
) -> List[RetrievalResult]:
    """
    Re-rank a list of RetrievalResult candidates using a cross-encoder.

    Each candidate's rerank_score is set in-place, then the list is
    sorted by descending rerank_score and truncated to top_n.
    rank fields are updated to reflect the new ordering.

    Parameters
    ----------
    query_text : str
        The original bug report text (not the fused vector — the raw text,
        because the cross-encoder needs string input).
    candidates : list[RetrievalResult]
        Top-K candidates from the bi-encoder retriever (typically K=20).
    reranker : CrossEncoderReranker
        Loaded cross-encoder model. Load once and reuse.
    top_n : int
        Number of results to return after re-ranking. Default 5.
    code_field : str
        Which field of RetrievalResult to use as the document text.
        "document" (default) = full code snippet stored at index time.
        "code_preview" = first 300 chars.

    Returns
    -------
    list[RetrievalResult]
        Top-N results sorted by descending rerank_score.
        Each result has rerank_score set and rank updated.

    Raises
    ------
    ValueError
        If query_text is empty or candidates is empty.
    TypeError
        If reranker is not a CrossEncoderReranker.
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text must be a non-empty string.")
    if not candidates:
        raise ValueError("candidates list must not be empty.")
    if not isinstance(reranker, CrossEncoderReranker):
        raise TypeError(
            f"Expected CrossEncoderReranker, got {type(reranker).__name__}"
        )

    # Build (query, code) pairs for the cross-encoder
    pairs = []
    for candidate in candidates:
        code_text = getattr(candidate, code_field, "") or candidate.document
        # Truncate very long code to avoid exceeding MAX_LENGTH after tokenisation
        code_text = code_text[:2000]
        pairs.append((query_text.strip(), code_text))

    logger.info(
        "Reranking %d candidates for query: %r", len(candidates), query_text[:60]
    )

    # Score all pairs in one batched call
    scores = reranker.score_pairs(pairs)

    # Attach scores to candidates (mutate in-place)
    for candidate, score in zip(candidates, scores):
        candidate.rerank_score = round(score, 6)

    # Sort by descending rerank_score
    reranked = sorted(candidates, key=lambda r: r.rerank_score, reverse=True)

    # Truncate and update rank fields
    top_results = reranked[:top_n]
    for new_rank, result in enumerate(top_results, start=1):
        result.rank = new_rank

    logger.info(
        "Reranking complete. top_%d scores: %s",
        top_n,
        [f"{r.rerank_score:.4f}" for r in top_results],
    )

    return top_results


def rerank_with_scores(
    query_text: str,
    candidates: List[RetrievalResult],
    reranker: CrossEncoderReranker,
    top_n: int = DEFAULT_TOP_N,
) -> dict:
    """
    Rerank and return a structured dict with both retrieval and rerank scores.
    Useful for evaluation (M7) and debugging.

    Returns
    -------
    dict with keys:
      "results"             : list[RetrievalResult]  — top_n reranked
      "retrieval_scores"    : list[float]             — original ANN scores
      "rerank_scores"       : list[float]             — cross-encoder scores
      "rank_changed"        : bool                    — did top-1 change?
    """
    original_top1_id = candidates[0].id if candidates else None
    original_scores = [c.retrieval_score for c in candidates]

    results = rerank(query_text, candidates, reranker, top_n=top_n)

    rerank_scores = [r.rerank_score for r in results]
    new_top1_id = results[0].id if results else None

    return {
        "results": results,
        "retrieval_scores": original_scores,
        "rerank_scores": rerank_scores,
        "rank_changed": original_top1_id != new_top1_id,
    }