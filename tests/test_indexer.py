"""
tests/test_indexer.py
=====================
Full test suite for the indexer module (Milestone 3).

All tests use an in-memory ChromaDB EphemeralClient — no disk I/O.
Each test gets a FRESH index via the `fresh_index` fixture, which creates
a uniquely named collection so tests never share state.

Run:
    pytest tests/test_indexer.py -v
    pytest tests/test_indexer.py -v --cov=indexer
"""

import uuid
import numpy as np
import pytest

import chromadb

from data_loader.models import CodeRecord
from indexer.chroma_index import CodeLensIndex


# ── Shared in-memory ChromaDB client (one per test session) ──────────────────
# EphemeralClient is in-memory only. We use unique collection names per test
# so that tests are fully isolated without restarting the client.

_SHARED_CLIENT = chromadb.EphemeralClient()

EMBED_DIM = 768


def make_fresh_index() -> CodeLensIndex:
    """Create a CodeLensIndex with a unique collection name for test isolation."""
    unique_name = f"test_{uuid.uuid4().hex[:8]}"
    index = CodeLensIndex.__new__(CodeLensIndex)
    index._persist_dir = ":memory:"
    index._collection_name = unique_name
    index._client = _SHARED_CLIENT
    index._collection = _SHARED_CLIENT.get_or_create_collection(
        name=unique_name,
        metadata={"hnsw:space": "cosine"},
    )
    return index


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def idx():
    """Fresh empty index for each test."""
    return make_fresh_index()


def make_unit_vector(seed: int = 0, dim: int = EMBED_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def make_record(i: int, with_embedding: bool = True) -> CodeRecord:
    r = CodeRecord(
        id=f"org_repo_{i % 3}::func_{i}::train_{i}",
        func_name=f"func_{i}",
        func_code=f"def func_{i}(x):\n    return x + {i}",
        docstring=f"Returns x plus {i}. Useful for arithmetic operations.",
        language="python",
        repository=f"org/repo_{i % 3}",
        url=f"https://github.com/org/repo_{i % 3}/blob/main/f{i}.py",
        tokens=["def", f"func_{i}", "x", "return"],
        partition="train",
    )
    if with_embedding:
        r.embedding = make_unit_vector(seed=i).tolist()
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Upsert tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUpsert:
    def test_upsert_increases_count(self, idx):
        n = idx.upsert([make_record(i) for i in range(10)], show_progress=False)
        assert n == 10
        assert idx.count() == 10

    def test_upsert_returns_count(self, idx):
        assert idx.upsert([make_record(i) for i in range(5)], show_progress=False) == 5

    def test_upsert_is_idempotent(self, idx):
        records = [make_record(i) for i in range(5)]
        idx.upsert(records, show_progress=False)
        idx.upsert(records, show_progress=False)
        assert idx.count() == 5

    def test_upsert_skips_records_without_embedding(self, idx):
        records = [make_record(i, with_embedding=False) for i in range(5)]
        n = idx.upsert(records, show_progress=False)
        assert n == 0
        assert idx.count() == 0

    def test_upsert_mixed_embedded_and_not(self, idx):
        embedded = [make_record(i, with_embedding=True) for i in range(3)]
        not_embedded = [make_record(i + 10, with_embedding=False) for i in range(2)]
        n = idx.upsert(embedded + not_embedded, show_progress=False)
        assert n == 3
        assert idx.count() == 3

    def test_upsert_empty_list(self, idx):
        n = idx.upsert([], show_progress=False)
        assert n == 0
        assert idx.count() == 0

    def test_upsert_batching_large_set(self, idx):
        records = [make_record(i) for i in range(150)]
        n = idx.upsert(records, batch_size=50, show_progress=False)
        assert n == 150
        assert idx.count() == 150

    def test_upsert_metadata_stored_correctly(self, idx):
        record = make_record(42)
        idx.upsert([record], show_progress=False)
        result = idx.get_by_id(record.id)
        assert result is not None
        assert result["func_name"] == "func_42"
        assert result["language"] == "python"
        assert "org/repo" in result["repository"]

    def test_upsert_updates_existing_record(self, idx):
        record = make_record(0)
        idx.upsert([record], show_progress=False)
        record.func_name = "updated_func"
        record.embedding = make_unit_vector(seed=99).tolist()
        idx.upsert([record], show_progress=False)
        assert idx.count() == 1
        result = idx.get_by_id(record.id)
        assert result["func_name"] == "updated_func"


# ─────────────────────────────────────────────────────────────────────────────
# Search tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSearch:
    def test_search_returns_list(self, idx):
        idx.upsert([make_record(i) for i in range(20)], show_progress=False)
        assert isinstance(idx.search(make_unit_vector(0), top_k=5), list)

    def test_search_top_k_respected(self, idx):
        idx.upsert([make_record(i) for i in range(20)], show_progress=False)
        assert len(idx.search(make_unit_vector(999), top_k=5)) == 5

    def test_search_top_k_capped_at_collection_size(self, idx):
        idx.upsert([make_record(i) for i in range(3)], show_progress=False)
        assert len(idx.search(make_unit_vector(0), top_k=100)) == 3

    def test_search_result_keys(self, idx):
        idx.upsert([make_record(0)], show_progress=False)
        results = idx.search(make_unit_vector(0), top_k=1)
        for key in ["id", "score", "func_name", "repository", "url",
                    "language", "docstring_preview", "code_preview", "document"]:
            assert key in results[0]

    def test_search_scores_between_0_and_1(self, idx):
        idx.upsert([make_record(i) for i in range(10)], show_progress=False)
        for r in idx.search(make_unit_vector(7), top_k=10):
            assert -0.01 <= r["score"] <= 1.01

    def test_search_scores_descending(self, idx):
        idx.upsert([make_record(i) for i in range(20)], show_progress=False)
        scores = [r["score"] for r in idx.search(make_unit_vector(3), top_k=10)]
        assert scores == sorted(scores, reverse=True)

    def test_exact_vector_gets_highest_score(self, idx):
        idx.upsert([make_record(i) for i in range(10)], show_progress=False)
        results = idx.search(make_unit_vector(seed=5), top_k=10)
        assert results[0]["score"] > 0.99

    def test_search_empty_collection_returns_empty(self, idx):
        assert idx.search(make_unit_vector(0), top_k=5) == []

    def test_search_result_ids_are_strings(self, idx):
        idx.upsert([make_record(0)], show_progress=False)
        assert isinstance(idx.search(make_unit_vector(0), top_k=1)[0]["id"], str)


# ─────────────────────────────────────────────────────────────────────────────
# Utility tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUtilities:
    def test_count_empty(self, idx):
        assert idx.count() == 0

    def test_count_after_upsert(self, idx):
        idx.upsert([make_record(i) for i in range(7)], show_progress=False)
        assert idx.count() == 7

    def test_peek_returns_n_records(self, idx):
        idx.upsert([make_record(i) for i in range(10)], show_progress=False)
        assert len(idx.peek(3)) == 3

    def test_peek_result_has_expected_keys(self, idx):
        idx.upsert([make_record(0)], show_progress=False)
        result = idx.peek(1)[0]
        for key in ["id", "func_name", "repository", "language"]:
            assert key in result

    def test_get_by_id_found(self, idx):
        record = make_record(7)
        idx.upsert([record], show_progress=False)
        result = idx.get_by_id(record.id)
        assert result is not None
        assert result["func_name"] == "func_7"

    def test_get_by_id_not_found(self, idx):
        idx.upsert([make_record(0)], show_progress=False)
        assert idx.get_by_id("nonexistent::id::train_999") is None

    def test_delete_collection_resets_count(self, idx):
        idx.upsert([make_record(i) for i in range(5)], show_progress=False)
        assert idx.count() == 5
        idx.delete_collection()
        assert idx.count() == 0

    def test_repr_contains_collection_name(self, idx):
        assert idx.collection_name in repr(idx)

    def test_collection_name_property(self, idx):
        assert idx.collection_name.startswith("test_")


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIndexerIntegration:
    def test_embed_index_search_roundtrip(self, idx):
        records = [make_record(i) for i in range(15)]
        idx.upsert(records, show_progress=False)
        assert idx.count() == 15
        results = idx.search(make_unit_vector(seed=3), top_k=5)
        assert len(results) == 5
        assert results[0]["func_name"] == "func_3"
        assert results[0]["score"] > 0.99

    def test_all_result_metadata_non_empty(self, idx):
        idx.upsert([make_record(i) for i in range(5)], show_progress=False)
        for r in idx.search(make_unit_vector(0), top_k=5):
            assert r["func_name"] != ""
            assert r["repository"] != ""
            assert r["id"] != ""

    def test_rebuild_after_delete(self, idx):
        idx.upsert([make_record(i) for i in range(5)], show_progress=False)
        assert idx.count() == 5
        idx.delete_collection()
        assert idx.count() == 0
        idx.upsert([make_record(i + 100) for i in range(3)], show_progress=False)
        assert idx.count() == 3
        assert len(idx.search(make_unit_vector(100), top_k=3)) == 3