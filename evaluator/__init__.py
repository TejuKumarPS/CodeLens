"""
evaluator — CodeLens Module 7
==============================
Responsible for:
  - Computing MRR (Mean Reciprocal Rank)
  - Computing NDCG@K (Normalized Discounted Cumulative Gain)
  - Computing Precision@K
  - Running full evaluation over a query set with ground-truth labels
  - Alpha sweep evaluation to tune the fusion hyperparameter
  - Saving evaluation reports as JSON

Public API:
  Metrics              — dataclass holding MRR, NDCG@K, P@K
  compute_mrr()        — compute MRR over a list of ranked result lists
  compute_ndcg()       — compute NDCG@K for a single query
  compute_precision()  — compute Precision@K for a single query
  Evaluator            — runs full evaluation pipeline over a query set
  evaluate_alpha_sweep()  — find best alpha by sweeping values
"""

from .metrics import Metrics, compute_mrr, compute_ndcg, compute_precision
from .evaluator import Evaluator
from .alpha_eval import evaluate_alpha_sweep

__all__ = [
    "Metrics",
    "compute_mrr",
    "compute_ndcg",
    "compute_precision",
    "Evaluator",
    "evaluate_alpha_sweep",
]