
"""
tests/test_evaluator.py
=======================
Tests for evaluator.evaluator.Evaluator

All tests mock the retriever and index so no model or ChromaDB needed.

Run:
    pytest tests/test_evaluator.py -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_loader.models import CodeRecord
from retriever.models import RetrievalResult
from evaluator.evaluator import Evaluator
from evaluator.metrics import Metrics


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_record(i: int) -> CodeRecord:
    return CodeRecord(
        id=f"repo::func_{i}::train_{i}",
        func_name=f"func_{i}",
        func_code=f"def func_{i}(x): return x + {i}",
        docstring=f"Adds x and {i}. Returns the arithmetic result.",
        language="python",
        repository="org/repo",
        url=f"https://github.com/org/repo/f{i}.py",
        tokens=["def", f"func_{i}", "x", "return"],
        partition="test",
    )


def make_result(record_id: str, rank: int, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        id=record_id,
        func_name="func",
        repository="org/repo",
        url="https://github.com",
        language="python",
        docstring_preview="preview",
        code_preview="def f(): pass",
        document="def f(): pass",
        retrieval_score=score,
        rank=rank,
    )


def make_mock_evaluator(hit_rank: int = 1) -> Evaluator:
    """
    Build an Evaluator whose retrieve() always returns the correct answer
    at the specified rank position.
    """
    mock_index   = MagicMock()
    mock_encoder = MagicMock()

    evaluator = Evaluator(
        index=mock_index,
        text_encoder=mock_encoder,
        k=10,
        top_k_retrieval=20,
    )
    return evaluator, hit_rank


class TestEvaluator:
    def _make_evaluator_with_perfect_retrieval(self):
        """Patch retrieve() so the correct record is always rank 1."""
        mock_index   = MagicMock()
        mock_encoder = MagicMock()
        return Evaluator(index=mock_index, text_encoder=mock_encoder, k=10)

    def test_run_returns_metrics(self):
        ev = self._make_evaluator_with_perfect_retrieval()
        records = [make_record(i) for i in range(5)]

        def mock_retrieve(text, index, text_encoder, top_k=20, **kwargs):
            # Always return the first record as top result
            return [make_result(records[0].id, rank=1)]

        with patch("retriever.retrieve.retrieve", side_effect=mock_retrieve):
            metrics = ev.run(records, show_progress=False)

        assert isinstance(metrics, Metrics)

    def test_run_perfect_retrieval_mrr_is_1(self):
        ev = self._make_evaluator_with_perfect_retrieval()
        records = [make_record(i) for i in range(10)]

        def mock_retrieve(text, index, text_encoder, top_k=20, **kwargs):
            # Find which record is being queried by its docstring
            for r in records:
                if r.docstring == text:
                    return [make_result(r.id, rank=1, score=0.99)]
            return []

        with patch("retriever.retrieve.retrieve", side_effect=mock_retrieve):
            metrics = ev.run(records, show_progress=False)

        assert metrics.mrr == pytest.approx(1.0)
        assert metrics.num_queries == 10

    def test_run_zero_retrieval_mrr_is_0(self):
        ev = self._make_evaluator_with_perfect_retrieval()
        records = [make_record(i) for i in range(5)]

        def mock_retrieve(**kwargs):
            return [make_result("wrong_id", rank=1)]

        with patch("retriever.retrieve.retrieve", side_effect=mock_retrieve):
            metrics = ev.run(records, show_progress=False)

        assert metrics.mrr == pytest.approx(0.0)
        assert metrics.num_queries_with_hit == 0

    def test_run_limit_reduces_query_count(self):
        ev = self._make_evaluator_with_perfect_retrieval()
        records = [make_record(i) for i in range(20)]

        def mock_retrieve(**kwargs):
            return []

        with patch("retriever.retrieve.retrieve", side_effect=mock_retrieve):
            metrics = ev.run(records, show_progress=False, limit=5)

        assert metrics.num_queries == 5

    def test_run_method_label_stored(self):
        ev = self._make_evaluator_with_perfect_retrieval()
        records = [make_record(0)]

        with patch("retriever.retrieve.retrieve", return_value=[]):
            metrics = ev.run(records, method="bm25", show_progress=False)

        assert metrics.method == "bm25"

    def test_run_k_field_matches_evaluator_k(self):
        ev = Evaluator(index=MagicMock(), text_encoder=MagicMock(), k=5)
        records = [make_record(0)]

        with patch("retriever.retrieve.retrieve", return_value=[]):
            metrics = ev.run(records, show_progress=False)

        assert metrics.k == 5

    def test_hit_at_rank_2_mrr_is_half(self):
        ev = self._make_evaluator_with_perfect_retrieval()
        records = [make_record(0)]

        def mock_retrieve(text, index, text_encoder, **kwargs):
            return [
                make_result("wrong_id", rank=1, score=0.9),
                make_result(records[0].id, rank=2, score=0.8),
            ]

        with patch("retriever.retrieve.retrieve", side_effect=mock_retrieve):
            metrics = ev.run(records, show_progress=False)

        assert metrics.mrr == pytest.approx(0.5)

    def test_save_report_creates_json(self):
        ev = self._make_evaluator_with_perfect_retrieval()
        m = Metrics(
            mrr=0.75, ndcg_at_k=0.6, precision_at_k=0.4,
            k=10, num_queries=50, num_queries_with_hit=40,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.json")
            ev.save_report(m, path)
            assert Path(path).exists()
            with open(path) as f:
                data = json.load(f)
            assert data["mrr"] == pytest.approx(0.75)
            assert data["num_queries"] == 50

    def test_run_bm25_returns_metrics(self):
        ev = self._make_evaluator_with_perfect_retrieval()
        records = [make_record(i) for i in range(5)]

        with patch("retriever.bm25.BM25Retriever") as MockBM25:
            mock_bm25_instance = MockBM25.return_value
            mock_bm25_instance.search.return_value = []
            # Import path must match where BM25Retriever is used
            with patch("retriever.retrieve.retrieve", return_value=[]):
                metrics = ev.run_bm25(
                    query_records=records,
                    corpus_records=records,
                )

        assert isinstance(metrics, Metrics)
        assert metrics.method == "bm25"