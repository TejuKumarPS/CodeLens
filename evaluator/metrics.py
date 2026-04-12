"""
evaluator.metrics
=================
Core IR evaluation metric implementations.

All three metrics operate on the same input format:
  retrieved_ids  : list[str]  — ordered list of retrieved document IDs
  relevant_ids   : set[str]   — ground-truth relevant document IDs

Metrics implemented:

MRR (Mean Reciprocal Rank)
  For each query, find the rank of the FIRST relevant result.
  Score = 1/rank. Average across all queries.
  Best = 1.0 (relevant result at rank 1 for every query).
  Formula: MRR = (1/|Q|) * Σ_q (1 / rank_q)

NDCG@K (Normalised Discounted Cumulative Gain)
  Measures ranking quality, weighting relevant results by position.
  A relevant result at rank 1 is worth more than one at rank K.
  Formula:
    DCG@K  = Σ_{i=1}^{K} rel_i / log2(i + 1)
    IDCG@K = DCG of the ideal ranking (all relevant results first)
    NDCG@K = DCG@K / IDCG@K
  For binary relevance: rel_i = 1 if retrieved_ids[i] in relevant_ids else 0

Precision@K
  Of the top-K retrieved results, what fraction are relevant?
  Formula: P@K = |{retrieved_ids[:K]} ∩ relevant_ids| / K
"""

import math
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ── Metrics dataclass ─────────────────────────────────────────────────────────

@dataclass
class Metrics:
    """
    Aggregate evaluation metrics across a query set.

    Attributes
    ----------
    mrr : float
        Mean Reciprocal Rank across all queries.
    ndcg_at_k : float
        Mean NDCG@K across all queries.
    precision_at_k : float
        Mean Precision@K across all queries.
    k : int
        The K used for NDCG and Precision.
    num_queries : int
        Number of queries evaluated.
    num_queries_with_hit : int
        Queries where at least one relevant result was in top-K.
    method : str
        Retrieval method label (e.g. "neural", "bm25", "reranked").
    alpha : float or None
        Fusion alpha used, if applicable.
    """

    mrr: float
    ndcg_at_k: float
    precision_at_k: float
    k: int
    num_queries: int
    num_queries_with_hit: int
    method: str = "neural"
    alpha: Optional[float] = None

    @property
    def hit_rate(self) -> float:
        """Fraction of queries with at least one hit in top-K."""
        if self.num_queries == 0:
            return 0.0
        return self.num_queries_with_hit / self.num_queries

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hit_rate"] = round(self.hit_rate, 4)
        return d

    def __repr__(self) -> str:
        return (
            f"Metrics(method={self.method!r}, k={self.k}, "
            f"MRR={self.mrr:.4f}, NDCG@{self.k}={self.ndcg_at_k:.4f}, "
            f"P@{self.k}={self.precision_at_k:.4f}, "
            f"queries={self.num_queries}, hits={self.num_queries_with_hit})"
        )


# ── Per-query metric functions ────────────────────────────────────────────────

def reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> float:
    """
    Compute the reciprocal rank for a single query.

    Parameters
    ----------
    retrieved_ids : list[str]
        Ordered list of retrieved document IDs (rank 1 first).
    relevant_ids : set[str]
        Ground-truth relevant document IDs.

    Returns
    -------
    float
        1/rank of first relevant result, or 0.0 if none found.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int = 10,
) -> float:
    """
    Compute NDCG@K for a single query.

    Parameters
    ----------
    retrieved_ids : list[str]
        Ordered retrieved IDs (rank 1 first).
    relevant_ids : set[str]
        Ground-truth relevant IDs (binary relevance).
    k : int
        Cutoff. Default 10.

    Returns
    -------
    float
        NDCG@K in [0.0, 1.0].
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    # DCG@K
    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)

    # IDCG@K — ideal: all relevant results in the first positions
    num_relevant = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_relevant + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_precision(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int = 10,
) -> float:
    """
    Compute Precision@K for a single query.

    Parameters
    ----------
    retrieved_ids : list[str]
        Ordered retrieved IDs (rank 1 first).
    relevant_ids : set[str]
        Ground-truth relevant IDs.
    k : int
        Cutoff. Default 10.

    Returns
    -------
    float
        P@K in [0.0, 1.0].
    """
    if not retrieved_ids or k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


# ── Aggregate metric functions ────────────────────────────────────────────────

def compute_mrr(
    retrieved_lists: List[List[str]],
    relevant_sets: List[Set[str]],
) -> float:
    """
    Compute Mean Reciprocal Rank over a collection of queries.

    Parameters
    ----------
    retrieved_lists : list[list[str]]
        One ordered retrieval list per query.
    relevant_sets : list[set[str]]
        One set of relevant IDs per query. Must match length of retrieved_lists.

    Returns
    -------
    float
        MRR in [0.0, 1.0].

    Raises
    ------
    ValueError
        If the two lists have different lengths or are empty.
    """
    if len(retrieved_lists) != len(relevant_sets):
        raise ValueError(
            f"retrieved_lists ({len(retrieved_lists)}) and "
            f"relevant_sets ({len(relevant_sets)}) must have the same length."
        )
    if not retrieved_lists:
        raise ValueError("Cannot compute MRR over an empty query set.")

    rr_scores = [
        reciprocal_rank(ret, rel)
        for ret, rel in zip(retrieved_lists, relevant_sets)
    ]
    return float(sum(rr_scores) / len(rr_scores))


def aggregate_metrics(
    retrieved_lists: List[List[str]],
    relevant_sets: List[Set[str]],
    k: int = 10,
    method: str = "neural",
    alpha: Optional[float] = None,
) -> Metrics:
    """
    Compute MRR, NDCG@K, and Precision@K over a full query set.

    Parameters
    ----------
    retrieved_lists : list[list[str]]
        One ordered retrieval list per query.
    relevant_sets : list[set[str]]
        One set of relevant IDs per query.
    k : int
        Cutoff for NDCG and Precision. Default 10.
    method : str
        Label for the retrieval method.
    alpha : float, optional
        Fusion alpha (for alpha-sweep records).

    Returns
    -------
    Metrics
    """
    if len(retrieved_lists) != len(relevant_sets):
        raise ValueError(
            "retrieved_lists and relevant_sets must have the same length."
        )
    if not retrieved_lists:
        raise ValueError("Query set must not be empty.")

    mrr_scores   = []
    ndcg_scores  = []
    prec_scores  = []
    hits         = 0

    for ret, rel in zip(retrieved_lists, relevant_sets):
        rr   = reciprocal_rank(ret, rel)
        ndcg = compute_ndcg(ret, rel, k=k)
        prec = compute_precision(ret, rel, k=k)

        mrr_scores.append(rr)
        ndcg_scores.append(ndcg)
        prec_scores.append(prec)

        if any(doc_id in rel for doc_id in ret[:k]):
            hits += 1

    n = len(retrieved_lists)
    return Metrics(
        mrr=round(sum(mrr_scores) / n, 6),
        ndcg_at_k=round(sum(ndcg_scores) / n, 6),
        precision_at_k=round(sum(prec_scores) / n, 6),
        k=k,
        num_queries=n,
        num_queries_with_hit=hits,
        method=method,
        alpha=alpha,
    )