
"""
evaluator.alpha_eval
====================
Alpha sweep evaluation — finds the optimal fusion weight by evaluating
MRR/NDCG@K across a grid of alpha values.

The fusion formula is:
    fused = alpha * text_vec + (1 - alpha) * projected_img_vec

alpha = 1.0 → text only (no image contribution)
alpha = 0.0 → image only (no text contribution)

This module sweeps alpha from 0.0 to 1.0 in configurable steps and
returns a sorted list of (alpha, Metrics) pairs so the evaluator can
pick the best value for production.

Usage
-----
    from evaluator.alpha_eval import evaluate_alpha_sweep

    results = evaluate_alpha_sweep(
        query_records=test_records,
        get_fused_vector_fn=my_fusion_fn,   # (record, alpha) → np.ndarray
        search_fn=my_search_fn,             # (vector) → list[str]
        relevant_fn=lambda r: {r.id},
        alphas=[0.3, 0.5, 0.7, 0.9],
        k=10,
    )
    best_alpha, best_metrics = results[0]
    print(f"Best alpha: {best_alpha} → MRR={best_metrics.mrr:.4f}")
"""

import logging
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from data_loader.models import CodeRecord
from .metrics import Metrics, aggregate_metrics

logger = logging.getLogger(__name__)

DEFAULT_ALPHAS = [round(a * 0.1, 1) for a in range(11)]   # [0.0, 0.1, ..., 1.0]


def evaluate_alpha_sweep(
    query_records: List[CodeRecord],
    get_fused_vector_fn: Callable[[CodeRecord, float], np.ndarray],
    search_fn: Callable[[np.ndarray], List[str]],
    relevant_fn: Callable[[CodeRecord], Set[str]],
    alphas: Optional[List[float]] = None,
    k: int = 10,
    method_prefix: str = "fusion",
) -> List[Tuple[float, Metrics]]:
    """
    Sweep alpha values and evaluate MRR/NDCG@K for each.

    Parameters
    ----------
    query_records : list[CodeRecord]
        Evaluation query set.
    get_fused_vector_fn : callable(record, alpha) → np.ndarray
        Given a CodeRecord and an alpha, returns a fused query vector.
        Signature: (record: CodeRecord, alpha: float) → np.ndarray shape (768,)
    search_fn : callable(vector) → list[str]
        Given a fused vector, returns an ordered list of retrieved doc IDs.
        Signature: (vector: np.ndarray) → list[str]
    relevant_fn : callable(record) → set[str]
        Given a query record, returns its set of relevant document IDs.
        For CodeSearchNet: lambda r: {r.id}
    alphas : list[float], optional
        Alpha values to evaluate. Default: [0.0, 0.1, ..., 1.0].
    k : int
        Cutoff for NDCG and Precision. Default 10.
    method_prefix : str
        Prefix for the method label in Metrics (e.g. "fusion_alpha_0.7").

    Returns
    -------
    list[tuple[float, Metrics]]
        All (alpha, Metrics) pairs, sorted by descending MRR.
        First element is the best-performing alpha.
    """
    if not query_records:
        raise ValueError("query_records must not be empty.")

    alphas = alphas or DEFAULT_ALPHAS
    invalid = [a for a in alphas if not 0.0 <= a <= 1.0]
    if invalid:
        raise ValueError(f"All alphas must be in [0.0, 1.0]. Got: {invalid}")

    logger.info(
        "Alpha sweep: %d queries × %d alphas (k=%d)",
        len(query_records), len(alphas), k,
    )

    results: List[Tuple[float, Metrics]] = []

    for alpha in alphas:
        retrieved_lists = []
        relevant_sets   = []

        for record in query_records:
            fused_vec  = get_fused_vector_fn(record, alpha)
            ret_ids    = search_fn(fused_vec)
            rel_ids    = relevant_fn(record)
            retrieved_lists.append(ret_ids)
            relevant_sets.append(rel_ids)

        metrics = aggregate_metrics(
            retrieved_lists=retrieved_lists,
            relevant_sets=relevant_sets,
            k=k,
            method=f"{method_prefix}_alpha_{alpha:.1f}",
            alpha=alpha,
        )
        results.append((alpha, metrics))
        logger.info(
            "  alpha=%.1f → MRR=%.4f  NDCG@%d=%.4f  P@%d=%.4f",
            alpha, metrics.mrr, k, metrics.ndcg_at_k, k, metrics.precision_at_k,
        )

    # Sort by descending MRR (primary), then NDCG (tiebreak)
    results.sort(key=lambda x: (x[1].mrr, x[1].ndcg_at_k), reverse=True)

    best_alpha, best_metrics = results[0]
    logger.info(
        "Best alpha: %.1f → MRR=%.4f  NDCG@%d=%.4f",
        best_alpha, best_metrics.mrr, k, best_metrics.ndcg_at_k,
    )

    return results


def build_alpha_sweep_report(
    sweep_results: List[Tuple[float, Metrics]],
) -> dict:
    """
    Convert alpha sweep results into a JSON-serialisable report dict.

    Parameters
    ----------
    sweep_results : list[tuple[float, Metrics]]
        Output of evaluate_alpha_sweep().

    Returns
    -------
    dict
        {
          "best_alpha": float,
          "best_mrr":   float,
          "results":    list[dict]  — one per alpha, sorted by MRR desc
        }
    """
    if not sweep_results:
        return {"best_alpha": None, "best_mrr": 0.0, "results": []}

    best_alpha, best_metrics = sweep_results[0]
    return {
        "best_alpha": best_alpha,
        "best_mrr":   best_metrics.mrr,
        "best_ndcg":  best_metrics.ndcg_at_k,
        "results": [
            {"alpha": alpha, **metrics.to_dict()}
            for alpha, metrics in sweep_results
        ],
    }