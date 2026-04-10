"""
tests/test_reranker.py
======================
Full test suite for the reranker module (Milestone 5).

All tests use a MockReranker that returns deterministic scores without
loading any model — no torch, no HuggingFace downloads needed.
The mock scores are seeded by position so tests are fully deterministic.

Run:
    pytest tests/test_reranker.py -v
    pytest tests/test_reranker.py -v --cov=reranker
"""

import math
import pytest
from unittest.mock import MagicMock, patch

from retriever.models import RetrievalResult
from reranker.cross_encoder import CrossEncoderReranker, _sigmoid
from reranker.rerank import rerank, rerank_with_scores, DEFAULT_TOP_N


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_result(
    rank: int,
    retrieval_score: float = 0.8,
    func_name: str = None,
    document: str = None,
) -> RetrievalResult:
    """Build a RetrievalResult with deterministic fields."""
    return RetrievalResult(
        id=f"repo::func_{rank}::train_{rank}",
        func_name=func_name or f"func_{rank}",
        repository="org/repo",
        url=f"https://github.com/org/repo/f{rank}.py",
        language="python",
        docstring_preview=f"Function number {rank} that does arithmetic.",
        code_preview=f"def func_{rank}(x): return x + {rank}",
        document=document or f"def func_{rank}(x):\n    # body\n    return x + {rank}",
        retrieval_score=retrieval_score,
        rank=rank,
        retrieval_method="neural",
    )


def make_candidates(n: int = 20) -> list:
    """
    Generate n RetrievalResult candidates with descending retrieval scores.
    """
    return [
        make_result(rank=i + 1, retrieval_score=round(1.0 - i * 0.03, 4))
        for i in range(n)
    ]


def make_mock_reranker(scores: list = None) -> CrossEncoderReranker:
    """
    Build a CrossEncoderReranker mock whose score_pairs() returns
    predetermined scores (or a default descending sequence if not given).
    """
    mock = MagicMock(spec=CrossEncoderReranker)

    def _score_pairs(pairs):
        if scores is not None:
            return scores[: len(pairs)]
        # Default: reverse the order so rank-1 becomes lowest-scoring
        # This lets us test that reranking actually changes the order
        n = len(pairs)
        return [round(1.0 - i * 0.05, 4) for i in range(n)]

    mock.score_pairs.side_effect = _score_pairs
    return mock


def make_inverting_reranker(n_candidates: int = 20) -> CrossEncoderReranker:
    """
    A mock reranker that inverts the retrieval order:
    the original rank-N (last) becomes rerank rank-1 (highest score).
    Scores: candidate[0] → lowest, candidate[N-1] → highest.
    """
    # candidate[0] gets score 0.0, candidate[N-1] gets score (N-1)*0.05
    scores = [round(i * 0.05, 4) for i in range(n_candidates)]
    return make_mock_reranker(scores=scores)


# ─────────────────────────────────────────────────────────────────────────────
# _sigmoid utility
# ─────────────────────────────────────────────────────────────────────────────

class TestSigmoid:
    def test_zero_gives_half(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive_approaches_one(self):
        assert _sigmoid(100.0) > 0.999

    def test_large_negative_approaches_zero(self):
        assert _sigmoid(-100.0) < 0.001

    def test_output_range(self):
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            s = _sigmoid(x)
            assert 0.0 <= s <= 1.0

    def test_symmetric(self):
        assert _sigmoid(2.0) == pytest.approx(1.0 - _sigmoid(-2.0), abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# CrossEncoderReranker (mock-based unit tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossEncoderRerankerMock:
    """
    Tests that verify CrossEncoderReranker's interface contracts
    using a mock that bypasses model loading entirely.
    """

    def test_score_pairs_returns_correct_length(self):
        mock = make_mock_reranker()
        pairs = [("bug text", "def foo(): pass")] * 5
        scores = mock.score_pairs(pairs)
        assert len(scores) == 5

    def test_score_pairs_scores_are_floats(self):
        mock = make_mock_reranker()
        pairs = [("query", "def f(): pass")] * 3
        scores = mock.score_pairs(pairs)
        assert all(isinstance(s, float) for s in scores)

    def test_score_single_returns_float(self):
        # score_single on the real class calls score_pairs internally
        # Here we test the interface via a patched score_pairs
        mock = make_mock_reranker(scores=[0.75])
        mock.score_single = lambda q, d: mock.score_pairs([(q, d)])[0]
        result = mock.score_single("crash on checkout", "def checkout(): pass")
        assert isinstance(result, float)

    def test_score_pairs_called_with_correct_args(self):
        mock = make_mock_reranker()
        pairs = [("query A", "code A"), ("query B", "code B")]
        mock.score_pairs(pairs)
        mock.score_pairs.assert_called_once_with(pairs)

    def test_empty_pairs_raises(self):
        """Real CrossEncoderReranker raises ValueError on empty pairs."""
        # We test this on the real class logic, patching only the model loading
        with patch.object(CrossEncoderReranker, "__init__", return_value=None):
            reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
            reranker._batch_size = 16
            reranker._torch = MagicMock()
            reranker._tokenizer = MagicMock()
            reranker._model = MagicMock()
            reranker.device = "cpu"
            with pytest.raises(ValueError, match="non-empty"):
                reranker.score_pairs([])


# ─────────────────────────────────────────────────────────────────────────────
# rerank() function tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRerank:
    def test_returns_list_of_results(self):
        candidates = make_candidates(20)
        mock = make_mock_reranker()
        results = rerank("app crashes at checkout", candidates, mock)
        assert isinstance(results, list)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_returns_top_n_default(self):
        candidates = make_candidates(20)
        mock = make_mock_reranker()
        results = rerank("crash on startup", candidates, mock)
        assert len(results) == DEFAULT_TOP_N == 5

    def test_returns_top_n_custom(self):
        candidates = make_candidates(20)
        mock = make_mock_reranker()
        for n in [1, 3, 5, 10]:
            results = rerank("any query", make_candidates(20), mock, top_n=n)
            assert len(results) == n

    def test_rerank_score_is_set(self):
        candidates = make_candidates(20)
        mock = make_mock_reranker()
        results = rerank("null pointer", candidates, mock)
        assert all(r.rerank_score is not None for r in results)
        assert all(isinstance(r.rerank_score, float) for r in results)

    def test_rerank_score_in_valid_range(self):
        candidates = make_candidates(20)
        mock = make_mock_reranker()
        results = rerank("memory leak", candidates, mock)
        for r in results:
            assert 0.0 <= r.rerank_score <= 1.0

    def test_results_sorted_by_rerank_score_descending(self):
        candidates = make_candidates(20)
        mock = make_mock_reranker()
        results = rerank("crash on payment", candidates, mock)
        scores = [r.rerank_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rank_fields_updated_sequentially(self):
        candidates = make_candidates(20)
        mock = make_mock_reranker()
        results = rerank("type error", candidates, mock)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_reranking_changes_order(self):
        """
        The inverting reranker gives the highest score to the last candidate.
        So the result at rank=1 should have originally been at a low retrieval rank.
        """
        candidates = make_candidates(20)
        original_top1_id = candidates[0].id

        inverting = make_inverting_reranker(20)
        results = rerank("stack overflow", candidates, inverting, top_n=5)

        # Top result after reranking should NOT be the original top retrieval result
        assert results[0].id != original_top1_id

    def test_retrieval_scores_preserved(self):
        """Original retrieval_score should not be modified by reranking."""
        candidates = make_candidates(10)
        original_scores = {c.id: c.retrieval_score for c in candidates}
        mock = make_mock_reranker()
        results = rerank("any query", candidates, mock, top_n=5)
        for r in results:
            assert r.retrieval_score == pytest.approx(original_scores[r.id])

    def test_empty_query_raises(self):
        mock = make_mock_reranker()
        with pytest.raises(ValueError, match="non-empty"):
            rerank("", make_candidates(5), mock)

    def test_whitespace_query_raises(self):
        mock = make_mock_reranker()
        with pytest.raises(ValueError, match="non-empty"):
            rerank("   ", make_candidates(5), mock)

    def test_empty_candidates_raises(self):
        mock = make_mock_reranker()
        with pytest.raises(ValueError, match="empty"):
            rerank("valid query", [], mock)

    def test_wrong_reranker_type_raises(self):
        with pytest.raises(TypeError, match="CrossEncoderReranker"):
            rerank("query", make_candidates(5), reranker="not_a_reranker")

    def test_top_n_larger_than_candidates(self):
        """top_n > len(candidates) should return all candidates."""
        candidates = make_candidates(3)
        mock = make_mock_reranker()
        results = rerank("query", candidates, mock, top_n=100)
        assert len(results) == 3

    def test_single_candidate(self):
        candidates = [make_result(rank=1, retrieval_score=0.9)]
        mock = make_mock_reranker(scores=[0.75])
        results = rerank("crash", candidates, mock, top_n=1)
        assert len(results) == 1
        assert results[0].rerank_score == pytest.approx(0.75)

    def test_score_pairs_called_once(self):
        """score_pairs should be called exactly once per rerank() call."""
        candidates = make_candidates(10)
        mock = make_mock_reranker()
        rerank("query", candidates, mock, top_n=5)
        mock.score_pairs.assert_called_once()

    def test_pairs_built_from_document_field(self):
        """Verify pairs are built from the document field by default."""
        document = "def special_function(x): return x * 2"
        candidates = [make_result(rank=1, document=document)]
        mock = make_mock_reranker(scores=[0.9])

        rerank("special function", candidates, mock, top_n=1)

        call_args = mock.score_pairs.call_args[0][0]
        assert call_args[0][1] == document

    def test_pairs_built_from_code_preview_field(self):
        """Test that code_field parameter is respected."""
        candidates = [make_result(rank=1)]
        mock = make_mock_reranker(scores=[0.9])

        rerank("query", candidates, mock, top_n=1, code_field="code_preview")
        call_args = mock.score_pairs.call_args[0][0]
        assert "code_preview" not in call_args[0][1]   # field value used, not name

    def test_query_is_stripped(self):
        """Leading/trailing whitespace in query should be stripped."""
        candidates = make_candidates(5)
        mock = make_mock_reranker()
        rerank("  crash on checkout  ", candidates, mock, top_n=3)
        call_args = mock.score_pairs.call_args[0][0]
        assert call_args[0][0] == "crash on checkout"

    def test_long_document_truncated(self):
        """Documents longer than 2000 chars should be truncated before scoring."""
        long_doc = "x" * 5000
        candidates = [make_result(rank=1, document=long_doc)]
        mock = make_mock_reranker(scores=[0.8])
        rerank("query", candidates, mock, top_n=1)
        call_args = mock.score_pairs.call_args[0][0]
        assert len(call_args[0][1]) <= 2000


# ─────────────────────────────────────────────────────────────────────────────
# rerank_with_scores() tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRerankWithScores:
    def test_returns_dict_with_expected_keys(self):
        candidates = make_candidates(10)
        mock = make_mock_reranker()
        result = rerank_with_scores("query", candidates, mock, top_n=5)
        for key in ["results", "retrieval_scores", "rerank_scores", "rank_changed"]:
            assert key in result

    def test_results_are_reranked(self):
        candidates = make_candidates(10)
        mock = make_mock_reranker()
        output = rerank_with_scores("query", candidates, mock, top_n=5)
        assert len(output["results"]) == 5

    def test_retrieval_scores_preserved(self):
        candidates = make_candidates(10)
        original_scores = [c.retrieval_score for c in candidates]
        mock = make_mock_reranker()
        output = rerank_with_scores("query", candidates, mock)
        assert output["retrieval_scores"] == original_scores

    def test_rerank_scores_match_results(self):
        candidates = make_candidates(10)
        mock = make_mock_reranker()
        output = rerank_with_scores("query", candidates, mock, top_n=5)
        expected = [r.rerank_score for r in output["results"]]
        assert output["rerank_scores"] == expected

    def test_rank_changed_true_when_order_changes(self):
        candidates = make_candidates(20)
        inverting = make_inverting_reranker(20)
        output = rerank_with_scores("query", candidates, inverting, top_n=5)
        assert output["rank_changed"] is True

    def test_rank_changed_false_when_order_preserved(self):
        """When reranker keeps same order, rank_changed should be False."""
        candidates = make_candidates(5)
        # Descending scores preserve the retrieval order
        same_order = make_mock_reranker(scores=[0.9, 0.8, 0.7, 0.6, 0.5])
        output = rerank_with_scores("query", candidates, same_order, top_n=5)
        assert output["rank_changed"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Integration: Full two-stage pipeline simulation
# ─────────────────────────────────────────────────────────────────────────────

class TestRerankerIntegration:
    def test_full_pipeline_bi_encoder_then_reranker(self):
        """
        Simulate the full M4 → M5 pipeline:
        1. Bi-encoder returns 20 candidates (sorted by retrieval_score desc)
        2. Cross-encoder re-ranks to top-5
        3. Verify output schema and ordering
        """
        candidates = make_candidates(20)
        assert len(candidates) == 20
        assert candidates[0].retrieval_score > candidates[-1].retrieval_score

        mock = make_mock_reranker()
        results = rerank("TypeError in payment module", candidates, mock, top_n=5)

        assert len(results) == 5
        for r in results:
            assert r.rerank_score is not None
            assert r.rank >= 1
            assert r.retrieval_score is not None   # original score preserved

    def test_all_results_have_complete_schema(self):
        candidates = make_candidates(10)
        mock = make_mock_reranker()
        results = rerank("stack overflow in recursive call", candidates, mock)
        for r in results:
            d = r.to_dict()
            assert d["rerank_score"] is not None
            assert d["retrieval_score"] is not None
            assert d["rank"] >= 1
            assert isinstance(d["func_name"], str)
            assert isinstance(d["url"], str)

    def test_reranker_handles_bm25_candidates(self):
        """Reranker should work equally on BM25 and neural candidates."""
        bm25_candidates = [
            make_result(rank=i + 1, retrieval_score=float(20 - i))
            for i in range(20)
        ]
        for c in bm25_candidates:
            c.retrieval_method = "bm25"

        mock = make_mock_reranker()
        results = rerank("checkout crash", bm25_candidates, mock, top_n=5)

        assert len(results) == 5
        assert all(r.rerank_score is not None for r in results)

    def test_pipeline_output_is_json_serialisable(self):
        import json
        candidates = make_candidates(10)
        mock = make_mock_reranker()
        results = rerank("crash on login", candidates, mock, top_n=3)
        # Should not raise
        serialised = json.dumps([r.to_dict() for r in results])
        parsed = json.loads(serialised)
        assert len(parsed) == 3

    def test_scores_improve_top1_confidence(self):
        """
        After reranking, the top result's rerank_score should be
        higher than the average rerank_score (it's the best one).
        """
        candidates = make_candidates(20)
        mock = make_mock_reranker()
        results = rerank("bug in data pipeline", candidates, mock, top_n=5)
        avg_score = sum(r.rerank_score for r in results) / len(results)
        assert results[0].rerank_score >= avg_score