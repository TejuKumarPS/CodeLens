"""
tests/test_metrics.py
=====================
Tests for evaluator.metrics:
  - reciprocal_rank()
  - compute_ndcg()
  - compute_precision()
  - compute_mrr()
  - aggregate_metrics()
  - Metrics dataclass

Run:
    pytest tests/test_metrics.py -v
"""

import json
import pytest
from evaluator.metrics import (
    Metrics,
    reciprocal_rank,
    compute_ndcg,
    compute_precision,
    compute_mrr,
    aggregate_metrics,
)


# ── reciprocal_rank ───────────────────────────────────────────────────────────

class TestReciprocalRank:
    def test_relevant_at_rank_1(self):
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_relevant_at_rank_2(self):
        assert reciprocal_rank(["a", "b", "c"], {"b"}) == pytest.approx(0.5)

    def test_relevant_at_rank_3(self):
        assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1/3)

    def test_no_relevant_returns_zero(self):
        assert reciprocal_rank(["a", "b", "c"], {"z"}) == pytest.approx(0.0)

    def test_empty_retrieved_returns_zero(self):
        assert reciprocal_rank([], {"a"}) == pytest.approx(0.0)

    def test_empty_relevant_returns_zero(self):
        assert reciprocal_rank(["a", "b"], set()) == pytest.approx(0.0)

    def test_multiple_relevant_uses_first_hit(self):
        # Both "b" and "c" are relevant; rank of first hit "b" = 2
        assert reciprocal_rank(["a", "b", "c"], {"b", "c"}) == pytest.approx(0.5)

    def test_relevant_at_rank_5(self):
        ids = ["a", "b", "c", "d", "e"]
        assert reciprocal_rank(ids, {"e"}) == pytest.approx(0.2)


# ── compute_ndcg ──────────────────────────────────────────────────────────────

class TestComputeNDCG:
    def test_perfect_ranking(self):
        # Single relevant doc at rank 1 → NDCG = 1.0
        assert compute_ndcg(["a", "b", "c"], {"a"}, k=3) == pytest.approx(1.0)

    def test_relevant_at_rank_2(self):
        # Ideal: at rank 1 → IDCG = 1/log2(2) = 1.0
        # Actual: at rank 2 → DCG  = 1/log2(3) ≈ 0.631
        import math
        expected = (1 / math.log2(3)) / (1 / math.log2(2))
        assert compute_ndcg(["a", "b", "c"], {"b"}, k=3) == pytest.approx(expected, abs=1e-5)

    def test_not_in_top_k_returns_zero(self):
        assert compute_ndcg(["a", "b", "c"], {"d"}, k=3) == pytest.approx(0.0)

    def test_empty_retrieved_returns_zero(self):
        assert compute_ndcg([], {"a"}, k=5) == pytest.approx(0.0)

    def test_empty_relevant_returns_zero(self):
        assert compute_ndcg(["a", "b"], set(), k=5) == pytest.approx(0.0)

    def test_k_cutoff_respected(self):
        # Relevant at rank 4, but k=3 → score should be 0
        assert compute_ndcg(["a", "b", "c", "d"], {"d"}, k=3) == pytest.approx(0.0)

    def test_multiple_relevant_docs(self):
        # "a" at rank 1, "b" at rank 2 — both relevant
        score = compute_ndcg(["a", "b", "c"], {"a", "b"}, k=3)
        assert 0.0 < score <= 1.0

    def test_all_relevant_returns_1(self):
        # All retrieved are relevant → perfect NDCG
        assert compute_ndcg(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)

    def test_output_in_unit_range(self):
        for seed in range(10):
            retrieved = [chr(ord("a") + i) for i in range(5)]
            relevant  = {chr(ord("a") + seed % 5)}
            score = compute_ndcg(retrieved, relevant, k=5)
            assert 0.0 <= score <= 1.0


# ── compute_precision ─────────────────────────────────────────────────────────

class TestComputePrecision:
    def test_all_relevant_returns_1(self):
        assert compute_precision(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)

    def test_none_relevant_returns_0(self):
        assert compute_precision(["a", "b", "c"], {"x", "y"}, k=3) == pytest.approx(0.0)

    def test_half_relevant(self):
        assert compute_precision(["a", "b", "c", "d"], {"a", "c"}, k=4) == pytest.approx(0.5)

    def test_k_1_relevant_at_rank_1(self):
        assert compute_precision(["a", "b"], {"a"}, k=1) == pytest.approx(1.0)

    def test_k_1_not_relevant_at_rank_1(self):
        assert compute_precision(["a", "b"], {"b"}, k=1) == pytest.approx(0.0)

    def test_k_larger_than_retrieved(self):
        # k=10 but only 3 retrieved; denominator is still k=10
        score = compute_precision(["a", "b", "c"], {"a"}, k=10)
        assert score == pytest.approx(1 / 10)

    def test_empty_retrieved_returns_0(self):
        assert compute_precision([], {"a"}, k=5) == pytest.approx(0.0)

    def test_k_0_returns_0(self):
        assert compute_precision(["a", "b"], {"a"}, k=0) == pytest.approx(0.0)


# ── compute_mrr ───────────────────────────────────────────────────────────────

class TestComputeMRR:
    def test_all_hits_at_rank_1(self):
        retrieved = [["a"], ["b"], ["c"]]
        relevant  = [{"a"}, {"b"}, {"c"}]
        assert compute_mrr(retrieved, relevant) == pytest.approx(1.0)

    def test_all_misses_returns_0(self):
        retrieved = [["a"], ["b"]]
        relevant  = [{"x"}, {"y"}]
        assert compute_mrr(retrieved, relevant) == pytest.approx(0.0)

    def test_mixed_queries(self):
        # Query 1: hit at rank 1 (RR=1.0), Query 2: hit at rank 2 (RR=0.5)
        retrieved = [["a", "b"], ["x", "b"]]
        relevant  = [{"a"}, {"b"}]
        assert compute_mrr(retrieved, relevant) == pytest.approx(0.75)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_mrr([["a"]], [{"a"}, {"b"}])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_mrr([], [])

    def test_single_query(self):
        assert compute_mrr([["a", "b"]], [{"b"}]) == pytest.approx(0.5)


# ── aggregate_metrics ─────────────────────────────────────────────────────────

class TestAggregateMetrics:
    def _run(self, retrieved, relevant, k=5, method="neural"):
        return aggregate_metrics(retrieved, relevant, k=k, method=method)

    def test_returns_metrics_instance(self):
        m = self._run([["a"]], [{"a"}])
        assert isinstance(m, Metrics)

    def test_perfect_retrieval(self):
        # Both queries have relevant at rank 1
        m = self._run([["a", "b"], ["c", "d"]], [{"a"}, {"c"}])
        assert m.mrr == pytest.approx(1.0)
        assert m.ndcg_at_k == pytest.approx(1.0)

    def test_num_queries_correct(self):
        m = self._run([["a"]] * 7, [{"a"}] * 7)
        assert m.num_queries == 7

    def test_hit_count_correct(self):
        retrieved = [["a", "b"], ["x", "y"]]
        relevant  = [{"a"}, {"z"}]
        m = self._run(retrieved, relevant, k=2)
        assert m.num_queries_with_hit == 1

    def test_method_label_stored(self):
        m = self._run([["a"]], [{"a"}], method="bm25")
        assert m.method == "bm25"

    def test_alpha_stored(self):
        m = aggregate_metrics([["a"]], [{"a"}], k=5, alpha=0.7)
        assert m.alpha == pytest.approx(0.7)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            aggregate_metrics([], [], k=5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            aggregate_metrics([["a"]], [{"a"}, {"b"}], k=5)

    def test_k_field_stored(self):
        m = self._run([["a"]], [{"a"}], k=7)
        assert m.k == 7


# ── Metrics dataclass ─────────────────────────────────────────────────────────

class TestMetricsDataclass:
    def _make(self, **kwargs) -> Metrics:
        defaults = dict(
            mrr=0.75, ndcg_at_k=0.65, precision_at_k=0.4,
            k=10, num_queries=100, num_queries_with_hit=80,
            method="neural", alpha=None,
        )
        defaults.update(kwargs)
        return Metrics(**defaults)

    def test_hit_rate_correct(self):
        m = self._make(num_queries=100, num_queries_with_hit=80)
        assert m.hit_rate == pytest.approx(0.8)

    def test_hit_rate_zero_queries(self):
        m = self._make(num_queries=0, num_queries_with_hit=0)
        assert m.hit_rate == pytest.approx(0.0)

    def test_to_dict_keys(self):
        m = self._make()
        d = m.to_dict()
        for key in ["mrr", "ndcg_at_k", "precision_at_k", "k",
                    "num_queries", "num_queries_with_hit", "hit_rate"]:
            assert key in d

    def test_to_dict_json_serialisable(self):
        m = self._make()
        json.dumps(m.to_dict())   # must not raise

    def test_repr_contains_mrr(self):
        m = self._make(mrr=0.8765)
        assert "0.8765" in repr(m)

    def test_repr_contains_method(self):
        m = self._make(method="bm25")
        assert "bm25" in repr(m)