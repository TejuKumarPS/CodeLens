
"""
tests/test_alpha_eval.py
========================
Tests for evaluator.alpha_eval:
  - evaluate_alpha_sweep()
  - build_alpha_sweep_report()

All tests use deterministic mock functions — no model or index needed.

Run:
    pytest tests/test_alpha_eval.py -v
"""

import numpy as np
import pytest

from data_loader.models import CodeRecord
from evaluator.alpha_eval import (
    evaluate_alpha_sweep,
    build_alpha_sweep_report,
    DEFAULT_ALPHAS,
)
from evaluator.metrics import Metrics


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_record(i: int) -> CodeRecord:
    return CodeRecord(
        id=f"repo::func_{i}::test_{i}",
        func_name=f"func_{i}",
        func_code=f"def func_{i}(x): return x",
        docstring=f"Returns x unchanged. Utility function number {i}.",
        language="python",
        repository="org/repo",
        url=f"https://github.com/org/repo/f{i}.py",
        tokens=["def", f"func_{i}", "return", "x"],
        partition="test",
    )


def make_unit_vec(seed: int, dim: int = 768) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# Deterministic mock functions for the sweep

def fused_fn_perfect(record: CodeRecord, alpha: float) -> np.ndarray:
    """Always returns the exact vector that matches this record."""
    # Use the record's position in a stable mapping via hash of its id
    seed = abs(hash(record.id)) % 1000
    return make_unit_vec(seed=seed)


def search_fn_perfect(vector: np.ndarray) -> list:
    """
    Returns the record ID whose seed vector is closest to `vector`.
    For perfect fn: the correct record always ranked #1.
    """
    best_id = None
    best_score = -1.0
    for i in range(20):
        candidate = make_unit_vec(seed=i)
        score = float(np.dot(vector, candidate))
        if score > best_score:
            best_score = score
            best_id = f"repo::func_{i}::test_{i}"
    return [best_id]


def search_fn_always_wrong(vector: np.ndarray) -> list:
    return ["wrong::id::test_999"]


def relevant_fn(record: CodeRecord) -> set:
    return {record.id}


# ── evaluate_alpha_sweep ──────────────────────────────────────────────────────

class TestEvaluateAlphaSweep:
    def test_returns_list_of_tuples(self):
        records = [make_record(i) for i in range(3)]
        results = evaluate_alpha_sweep(
            query_records=records,
            get_fused_vector_fn=fused_fn_perfect,
            search_fn=search_fn_perfect,
            relevant_fn=relevant_fn,
            alphas=[0.5, 1.0],
            k=1,
        )
        assert isinstance(results, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in results)

    def test_each_element_is_alpha_metrics_pair(self):
        records = [make_record(0)]
        results = evaluate_alpha_sweep(
            query_records=records,
            get_fused_vector_fn=fused_fn_perfect,
            search_fn=search_fn_perfect,
            relevant_fn=relevant_fn,
            alphas=[0.7],
            k=1,
        )
        alpha, metrics = results[0]
        assert isinstance(alpha, float)
        assert isinstance(metrics, Metrics)

    def test_correct_number_of_alphas(self):
        records = [make_record(0)]
        alphas = [0.2, 0.5, 0.8]
        results = evaluate_alpha_sweep(
            query_records=records,
            get_fused_vector_fn=lambda r, a: make_unit_vec(0),
            search_fn=search_fn_always_wrong,
            relevant_fn=relevant_fn,
            alphas=alphas,
            k=1,
        )
        assert len(results) == 3

    def test_sorted_by_mrr_descending(self):
        records = [make_record(i) for i in range(5)]
        results = evaluate_alpha_sweep(
            query_records=records,
            get_fused_vector_fn=fused_fn_perfect,
            search_fn=search_fn_perfect,
            relevant_fn=relevant_fn,
            alphas=[0.0, 0.5, 1.0],
            k=1,
        )
        mrrs = [m.mrr for _, m in results]
        assert mrrs == sorted(mrrs, reverse=True)

    def test_default_alphas_11_values(self):
        records = [make_record(0)]
        results = evaluate_alpha_sweep(
            query_records=records,
            get_fused_vector_fn=lambda r, a: make_unit_vec(0),
            search_fn=search_fn_always_wrong,
            relevant_fn=relevant_fn,
            k=1,
        )
        assert len(results) == 11

    def test_alpha_stored_in_metrics(self):
        records = [make_record(0)]
        results = evaluate_alpha_sweep(
            query_records=records,
            get_fused_vector_fn=lambda r, a: make_unit_vec(0),
            search_fn=search_fn_always_wrong,
            relevant_fn=relevant_fn,
            alphas=[0.3],
            k=1,
        )
        alpha, metrics = results[0]
        assert alpha == pytest.approx(0.3)
        assert metrics.alpha == pytest.approx(0.3)

    def test_all_wrong_gives_mrr_zero(self):
        records = [make_record(i) for i in range(5)]
        results = evaluate_alpha_sweep(
            query_records=records,
            get_fused_vector_fn=lambda r, a: make_unit_vec(0),
            search_fn=search_fn_always_wrong,
            relevant_fn=relevant_fn,
            alphas=[0.5],
            k=5,
        )
        _, metrics = results[0]
        assert metrics.mrr == pytest.approx(0.0)

    def test_empty_query_records_raises(self):
        with pytest.raises(ValueError, match="empty"):
            evaluate_alpha_sweep(
                query_records=[],
                get_fused_vector_fn=lambda r, a: make_unit_vec(0),
                search_fn=search_fn_always_wrong,
                relevant_fn=relevant_fn,
            )

    def test_invalid_alpha_raises(self):
        records = [make_record(0)]
        with pytest.raises(ValueError, match="1.0"):
            evaluate_alpha_sweep(
                query_records=records,
                get_fused_vector_fn=lambda r, a: make_unit_vec(0),
                search_fn=search_fn_always_wrong,
                relevant_fn=relevant_fn,
                alphas=[0.5, 1.5],
            )

    def test_method_prefix_in_label(self):
        records = [make_record(0)]
        results = evaluate_alpha_sweep(
            query_records=records,
            get_fused_vector_fn=lambda r, a: make_unit_vec(0),
            search_fn=search_fn_always_wrong,
            relevant_fn=relevant_fn,
            alphas=[0.7],
            k=1,
            method_prefix="fusion",
        )
        _, metrics = results[0]
        assert "fusion" in metrics.method
        assert "0.7" in metrics.method

    def test_num_queries_correct(self):
        records = [make_record(i) for i in range(8)]
        results = evaluate_alpha_sweep(
            query_records=records,
            get_fused_vector_fn=lambda r, a: make_unit_vec(0),
            search_fn=search_fn_always_wrong,
            relevant_fn=relevant_fn,
            alphas=[0.5],
            k=5,
        )
        _, metrics = results[0]
        assert metrics.num_queries == 8


# ── build_alpha_sweep_report ──────────────────────────────────────────────────

class TestBuildAlphaSweepReport:
    def _make_sweep_results(self) -> list:
        records = [make_record(i) for i in range(3)]
        return evaluate_alpha_sweep(
            query_records=records,
            get_fused_vector_fn=lambda r, a: make_unit_vec(0),
            search_fn=search_fn_always_wrong,
            relevant_fn=relevant_fn,
            alphas=[0.3, 0.7],
            k=5,
        )

    def test_returns_dict(self):
        results = self._make_sweep_results()
        report = build_alpha_sweep_report(results)
        assert isinstance(report, dict)

    def test_expected_keys(self):
        results = self._make_sweep_results()
        report = build_alpha_sweep_report(results)
        for key in ["best_alpha", "best_mrr", "best_ndcg", "results"]:
            assert key in report

    def test_results_list_length(self):
        results = self._make_sweep_results()
        report = build_alpha_sweep_report(results)
        assert len(report["results"]) == 2

    def test_empty_input_returns_defaults(self):
        report = build_alpha_sweep_report([])
        assert report["best_alpha"] is None
        assert report["best_mrr"] == 0.0
        assert report["results"] == []

    def test_report_is_json_serialisable(self):
        import json
        results = self._make_sweep_results()
        report = build_alpha_sweep_report(results)
        json.dumps(report)   # must not raise

    def test_best_alpha_is_float(self):
        results = self._make_sweep_results()
        report = build_alpha_sweep_report(results)
        assert isinstance(report["best_alpha"], float)