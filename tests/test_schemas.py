"""
tests/test_schemas.py
=====================
Tests for api.schemas — Pydantic request/response models.
No HTTP client needed; validates model behaviour directly.

Run:
    pytest tests/test_schemas.py -v
"""

import pytest
from pydantic import ValidationError

from api.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    HealthResponse,
    IndexStatsResponse,
    EvaluateRequest,
    EvaluateResponse,
)


# ── SearchRequest ─────────────────────────────────────────────────────────────

class TestSearchRequest:
    def test_minimal_valid(self):
        req = SearchRequest(query="app crashes at checkout")
        assert req.query == "app crashes at checkout"
        assert req.image_b64 is None
        assert req.top_k_retrieval == 20
        assert req.top_n_results == 5
        assert req.alpha == 0.7

    def test_query_stripped(self):
        req = SearchRequest(query="  null pointer  ")
        assert req.query == "null pointer"

    def test_empty_query_raises(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="")

    def test_whitespace_only_query_raises(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="   ")

    def test_query_too_long_raises(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="x" * 2001)

    def test_alpha_bounds_valid(self):
        SearchRequest(query="q", alpha=0.0)
        SearchRequest(query="q", alpha=1.0)
        SearchRequest(query="q", alpha=0.5)

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="q", alpha=1.1)
        with pytest.raises(ValidationError):
            SearchRequest(query="q", alpha=-0.1)

    def test_top_k_retrieval_bounds(self):
        SearchRequest(query="q", top_k_retrieval=1)
        SearchRequest(query="q", top_k_retrieval=100)

    def test_top_k_retrieval_out_of_bounds_raises(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="q", top_k_retrieval=0)
        with pytest.raises(ValidationError):
            SearchRequest(query="q", top_k_retrieval=101)

    def test_top_n_results_bounds(self):
        SearchRequest(query="q", top_n_results=1)
        SearchRequest(query="q", top_n_results=20)

    def test_top_n_results_out_of_bounds_raises(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="q", top_n_results=0)
        with pytest.raises(ValidationError):
            SearchRequest(query="q", top_n_results=21)

    def test_language_lowercased(self):
        req = SearchRequest(query="q", language="Python")
        assert req.language == "python"

    def test_language_none_allowed(self):
        req = SearchRequest(query="q", language=None)
        assert req.language is None

    def test_with_image_b64(self):
        req = SearchRequest(query="crash", image_b64="abc123==")
        assert req.image_b64 == "abc123=="


# ── SearchResultItem ──────────────────────────────────────────────────────────

class TestSearchResultItem:
    def _make(self, **kwargs) -> SearchResultItem:
        defaults = dict(
            rank=1, func_name="checkout_handler",
            repository="shop/app", url="https://github.com/shop/app",
            language="python", docstring_preview="Handles checkout.",
            code_preview="def checkout(): pass",
            retrieval_score=0.92, rerank_score=0.85,
            retrieval_method="neural",
        )
        defaults.update(kwargs)
        return SearchResultItem(**defaults)

    def test_valid_construction(self):
        item = self._make()
        assert item.rank == 1
        assert item.func_name == "checkout_handler"

    def test_rerank_score_optional(self):
        item = self._make(rerank_score=None)
        assert item.rerank_score is None

    def test_serialises_to_dict(self):
        item = self._make()
        d = item.model_dump()
        assert "rank" in d
        assert "rerank_score" in d


# ── SearchResponse ────────────────────────────────────────────────────────────

class TestSearchResponse:
    def test_valid_empty_results(self):
        resp = SearchResponse(
            query="crash", has_image=False,
            alpha=0.7, results=[], num_results=0,
        )
        assert resp.num_results == 0

    def test_serialises(self):
        resp = SearchResponse(
            query="q", has_image=True,
            alpha=0.5, results=[], num_results=0,
        )
        d = resp.model_dump()
        assert d["has_image"] is True


# ── HealthResponse ────────────────────────────────────────────────────────────

class TestHealthResponse:
    def test_defaults(self):
        h = HealthResponse(index_loaded=True, index_size=4904, models_loaded=True)
        assert h.status == "ok"

    def test_not_ready(self):
        h = HealthResponse(status="initialising", index_loaded=False,
                           index_size=0, models_loaded=False)
        assert h.status == "initialising"
        assert not h.models_loaded


# ── IndexStatsResponse ────────────────────────────────────────────────────────

class TestIndexStatsResponse:
    def test_valid(self):
        s = IndexStatsResponse(
            collection_name="codelens_python",
            num_vectors=4904,
            persist_dir="chroma_db/",
        )
        assert s.num_vectors == 4904


# ── EvaluateRequest / EvaluateResponse ───────────────────────────────────────

class TestEvaluateSchemas:
    def test_evaluate_request_defaults(self):
        req = EvaluateRequest(parquet_path="data/processed/python_test.parquet")
        assert req.limit == 100
        assert req.k == 10

    def test_evaluate_request_limit_ge1(self):
        with pytest.raises(ValidationError):
            EvaluateRequest(parquet_path="p", limit=0)

    def test_evaluate_response_valid(self):
        resp = EvaluateResponse(
            mrr=0.75, ndcg_at_k=0.6, precision_at_k=0.4,
            k=10, num_queries=100, num_queries_with_hit=80,
            hit_rate=0.8, method="neural",
        )
        assert resp.mrr == pytest.approx(0.75)
        assert resp.hit_rate == pytest.approx(0.8)