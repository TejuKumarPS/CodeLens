"""
tests/test_retriever.py
=======================
Full test suite for the retriever module (Milestone 4).

All tests are mock-based — no ChromaDB disk I/O, no model downloads.
The NeuralRetriever tests use a real in-memory ChromaDB index (EphemeralClient)
seeded with deterministic unit vectors, matching the pattern from test_indexer.py.

Run:
    pytest tests/test_retriever.py -v
    pytest tests/test_retriever.py -v --cov=retriever
"""

import uuid
import numpy as np
import pytest
from unittest.mock import MagicMock

import chromadb

from data_loader.models import CodeRecord
from embedder.embed_pipeline import EmbeddedQuery
from indexer.chroma_index import CodeLensIndex
from retriever.models import RetrievalResult
from retriever.neural import NeuralRetriever
from retriever.bm25 import BM25Retriever, _tokenize_query
from retriever.retrieve import retrieve

# ── Shared in-memory ChromaDB client ─────────────────────────────────────────
_CLIENT = chromadb.EphemeralClient()
EMBED_DIM = 768


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_unit_vector(seed: int = 0, dim: int = EMBED_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def make_fresh_index() -> CodeLensIndex:
    name = f"test_{uuid.uuid4().hex[:8]}"
    idx = CodeLensIndex.__new__(CodeLensIndex)
    idx._persist_dir = ":memory:"
    idx._collection_name = name
    idx._client = _CLIENT
    idx._collection = _CLIENT.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )
    return idx


def make_record(i: int, with_embedding: bool = True) -> CodeRecord:
    r = CodeRecord(
        id=f"repo_{i % 3}::func_{i}::train_{i}",
        func_name=f"func_{i}",
        func_code=f"def func_{i}(x, y):\n    return x + y + {i}",
        docstring=f"Adds two numbers and an offset of {i}. Useful utility.",
        language="python",
        repository=f"org/repo_{i % 3}",
        url=f"https://github.com/org/repo_{i % 3}/f{i}.py",
        tokens=["def", f"func_{i}", "x", "y", "return", "x", "y"],
        partition="train",
    )
    if with_embedding:
        r.embedding = make_unit_vector(seed=i).tolist()
    return r


def make_populated_index(n: int = 20) -> CodeLensIndex:
    idx = make_fresh_index()
    records = [make_record(i) for i in range(n)]
    idx.upsert(records, show_progress=False)
    return idx


def make_embedded_query(
    text: str = "app crashes at checkout",
    seed: int = 42,
    has_image: bool = False,
) -> EmbeddedQuery:
    vec = make_unit_vector(seed=seed)
    return EmbeddedQuery(
        text=text,
        text_vector=vec,
        fused_vector=vec,
        alpha=1.0 if not has_image else 0.7,
        has_image=has_image,
        image_vector=make_unit_vector(seed=seed + 1, dim=512) if has_image else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RetrievalResult tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrievalResult:
    def _make_hit(self, score: float = 0.85) -> dict:
        return {
            "id": "repo::func_1::train_1",
            "score": score,
            "func_name": "func_1",
            "repository": "org/repo",
            "url": "https://github.com/org/repo",
            "language": "python",
            "docstring_preview": "Adds two numbers.",
            "code_preview": "def func_1(x): return x",
            "document": "def func_1(x): return x",
        }

    def test_from_chroma_hit_basic(self):
        result = RetrievalResult.from_chroma_hit(self._make_hit(), rank=1)
        assert result.rank == 1
        assert result.func_name == "func_1"
        assert result.retrieval_score == pytest.approx(0.85)
        assert result.retrieval_method == "neural"
        assert result.rerank_score is None

    def test_from_chroma_hit_bm25(self):
        result = RetrievalResult.from_chroma_hit(self._make_hit(), rank=3, method="bm25")
        assert result.retrieval_method == "bm25"
        assert result.rank == 3

    def test_to_dict_keys(self):
        result = RetrievalResult.from_chroma_hit(self._make_hit(), rank=1)
        d = result.to_dict()
        for key in ["id", "func_name", "repository", "url", "language",
                    "retrieval_score", "rank", "rerank_score", "retrieval_method"]:
            assert key in d

    def test_to_dict_is_json_serialisable(self):
        import json
        result = RetrievalResult.from_chroma_hit(self._make_hit(), rank=1)
        json.dumps(result.to_dict())   # should not raise

    def test_rerank_score_settable(self):
        result = RetrievalResult.from_chroma_hit(self._make_hit(), rank=1)
        result.rerank_score = 0.92
        assert result.rerank_score == pytest.approx(0.92)

    def test_repr_contains_rank_and_func(self):
        result = RetrievalResult.from_chroma_hit(self._make_hit(), rank=2)
        r = repr(result)
        assert "rank=2" in r
        assert "func_1" in r

    def test_missing_optional_fields_default_to_empty(self):
        sparse_hit = {"id": "x", "score": 0.5}
        result = RetrievalResult.from_chroma_hit(sparse_hit, rank=1)
        assert result.func_name == ""
        assert result.document == ""


# ─────────────────────────────────────────────────────────────────────────────
# NeuralRetriever tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNeuralRetriever:
    def test_init_valid(self):
        idx = make_populated_index(10)
        r = NeuralRetriever(idx, top_k=5)
        assert r.top_k == 5
        assert r.index_size == 10

    def test_init_wrong_type_raises(self):
        with pytest.raises(TypeError):
            NeuralRetriever("not_an_index")

    def test_init_invalid_top_k_raises(self):
        idx = make_populated_index(5)
        with pytest.raises(ValueError):
            NeuralRetriever(idx, top_k=0)

    def test_search_returns_list_of_results(self):
        idx = make_populated_index(20)
        r = NeuralRetriever(idx, top_k=5)
        query = make_embedded_query()
        results = r.search(query)
        assert isinstance(results, list)
        assert all(isinstance(res, RetrievalResult) for res in results)

    def test_search_returns_top_k(self):
        idx = make_populated_index(20)
        r = NeuralRetriever(idx, top_k=7)
        results = r.search(make_embedded_query())
        assert len(results) == 7

    def test_search_ranks_are_sequential(self):
        idx = make_populated_index(20)
        r = NeuralRetriever(idx, top_k=5)
        results = r.search(make_embedded_query())
        assert [res.rank for res in results] == [1, 2, 3, 4, 5]

    def test_search_scores_descending(self):
        idx = make_populated_index(20)
        r = NeuralRetriever(idx, top_k=10)
        results = r.search(make_embedded_query())
        scores = [res.retrieval_score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_scores_in_valid_range(self):
        idx = make_populated_index(15)
        r = NeuralRetriever(idx, top_k=10)
        for res in r.search(make_embedded_query()):
            assert -0.01 <= res.retrieval_score <= 1.01

    def test_search_method_is_neural(self):
        idx = make_populated_index(10)
        r = NeuralRetriever(idx, top_k=3)
        for res in r.search(make_embedded_query()):
            assert res.retrieval_method == "neural"

    def test_search_empty_index_returns_empty(self):
        idx = make_fresh_index()
        r = NeuralRetriever(idx, top_k=5)
        assert r.search(make_embedded_query()) == []

    def test_search_wrong_query_type_raises(self):
        idx = make_populated_index(5)
        r = NeuralRetriever(idx, top_k=3)
        with pytest.raises(TypeError):
            r.search("not an EmbeddedQuery")

    def test_search_vector_direct(self):
        idx = make_populated_index(15)
        r = NeuralRetriever(idx, top_k=5)
        vec = make_unit_vector(seed=0)
        results = r.search_vector(vec)
        assert len(results) == 5
        assert all(isinstance(res, RetrievalResult) for res in results)

    def test_search_vector_wrong_shape_raises(self):
        idx = make_populated_index(5)
        r = NeuralRetriever(idx, top_k=3)
        with pytest.raises(ValueError):
            r.search_vector(np.ones((5, 768)))

    def test_exact_match_is_top_result(self):
        """Query with the same vector as record 7 → record 7 should rank #1."""
        idx = make_populated_index(15)
        r = NeuralRetriever(idx, top_k=5)
        exact_vec = make_unit_vector(seed=7)
        results = r.search_vector(exact_vec)
        assert results[0].func_name == "func_7"
        assert results[0].retrieval_score > 0.99

    def test_top_k_setter(self):
        idx = make_populated_index(10)
        r = NeuralRetriever(idx, top_k=5)
        r.top_k = 10
        assert r.top_k == 10

    def test_top_k_setter_invalid_raises(self):
        idx = make_populated_index(5)
        r = NeuralRetriever(idx, top_k=5)
        with pytest.raises(ValueError):
            r.top_k = -1

    def test_top_k_capped_at_index_size(self):
        idx = make_populated_index(3)
        r = NeuralRetriever(idx, top_k=100)
        results = r.search(make_embedded_query())
        assert len(results) == 3

    def test_repr_contains_collection_and_top_k(self):
        idx = make_populated_index(5)
        r = NeuralRetriever(idx, top_k=10)
        assert "top_k=10" in repr(r)
        assert "index_size=5" in repr(r)

    def test_multimodal_query_same_shape_results(self):
        """Multimodal fused vector should work identically to text-only."""
        idx = make_populated_index(15)
        r = NeuralRetriever(idx, top_k=5)
        q_text = make_embedded_query(has_image=False, seed=10)
        q_multi = make_embedded_query(has_image=True, seed=10)
        results_text = r.search(q_text)
        results_multi = r.search(q_multi)
        # Both return 5 results of the correct type
        assert len(results_text) == 5
        assert len(results_multi) == 5


# ─────────────────────────────────────────────────────────────────────────────
# BM25Retriever tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenizeQuery:
    def test_basic_split(self):
        assert _tokenize_query("app crashes at checkout") == \
               ["app", "crashes", "at", "checkout"]

    def test_lowercases(self):
        assert _tokenize_query("NullPointerException") == ["nullpointerexception"]

    def test_strips_punctuation(self):
        tokens = _tokenize_query("stack-overflow in func()")
        assert "stack" in tokens
        assert "overflow" in tokens

    def test_empty_string(self):
        assert _tokenize_query("") == []

    def test_numbers_kept(self):
        assert "404" in _tokenize_query("HTTP 404 error")


class TestBM25Retriever:
    @pytest.fixture
    def records(self):
        return [make_record(i) for i in range(20)]

    @pytest.fixture
    def bm25(self, records):
        return BM25Retriever(records)

    def test_init_sets_corpus_size(self, bm25, records):
        assert bm25.corpus_size == len(records)

    def test_init_empty_raises(self):
        with pytest.raises(ValueError):
            BM25Retriever([])

    def test_search_returns_list(self, bm25):
        assert isinstance(bm25.search("add numbers"), list)

    def test_search_returns_retrieval_results(self, bm25):
        results = bm25.search("add numbers", top_k=5)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_search_top_k_respected(self, bm25):
        results = bm25.search("add numbers offset", top_k=5)
        assert len(results) == 5

    def test_search_top_k_capped_at_corpus(self, bm25):
        results = bm25.search("function", top_k=1000)
        assert len(results) == 20

    def test_search_ranks_sequential(self, bm25):
        results = bm25.search("add numbers", top_k=5)
        assert [r.rank for r in results] == [1, 2, 3, 4, 5]

    def test_search_method_is_bm25(self, bm25):
        for r in bm25.search("return value", top_k=3):
            assert r.retrieval_method == "bm25"

    def test_search_empty_query_raises(self, bm25):
        with pytest.raises(ValueError):
            bm25.search("")

    def test_search_whitespace_query_raises(self, bm25):
        with pytest.raises(ValueError):
            bm25.search("   ")

    def test_keyword_match_gets_high_score(self):
        """A record with matching tokens should score higher than unrelated ones."""
        records = [
            make_record(0),   # generic
            make_record(1),   # generic
        ]
        # Add a specialised record whose docstring exactly matches the query
        special = CodeRecord(
            id="special::checkout::train_99",
            func_name="process_checkout",
            func_code="def process_checkout(cart): return cart.total",
            docstring="Process the checkout flow for the shopping cart.",
            language="python",
            repository="shop/app",
            url="https://github.com/shop/app",
            tokens=["def", "process_checkout", "cart", "return", "total"],
            partition="train",
        )
        retriever = BM25Retriever(records + [special])
        results = retriever.search("checkout cart process", top_k=3)
        # The special record should rank #1
        assert results[0].func_name == "process_checkout"

    def test_rerank_score_is_none_by_default(self, bm25):
        results = bm25.search("function", top_k=1)
        assert results[0].rerank_score is None

    def test_repr(self, bm25):
        assert "20" in repr(bm25)


# ─────────────────────────────────────────────────────────────────────────────
# retrieve() convenience function tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrieveFunction:
    def _make_mock_encoder(self, seed: int = 42) -> MagicMock:
        enc = MagicMock()
        vec = make_unit_vector(seed=seed)
        enc.encode.return_value = vec
        enc.encode_batch.return_value = np.vstack([vec])
        return enc

    def test_retrieve_returns_results(self):
        idx = make_populated_index(20)
        encoder = self._make_mock_encoder()
        results = retrieve("app crashes at checkout", index=idx,
                           text_encoder=encoder, top_k=5)
        assert len(results) == 5
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieve_empty_text_raises(self):
        idx = make_populated_index(5)
        encoder = self._make_mock_encoder()
        with pytest.raises(ValueError):
            retrieve("", index=idx, text_encoder=encoder)

    def test_retrieve_ranks_start_at_one(self):
        idx = make_populated_index(15)
        encoder = self._make_mock_encoder()
        results = retrieve("null pointer error", index=idx,
                           text_encoder=encoder, top_k=5)
        assert results[0].rank == 1
        assert results[-1].rank == 5

    def test_retrieve_scores_descending(self):
        idx = make_populated_index(20)
        encoder = self._make_mock_encoder()
        results = retrieve("memory leak in parser", index=idx,
                           text_encoder=encoder, top_k=10)
        scores = [r.retrieval_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_top_k_respected(self):
        idx = make_populated_index(20)
        encoder = self._make_mock_encoder()
        for k in [1, 5, 10, 20]:
            results = retrieve("test query", index=idx,
                               text_encoder=encoder, top_k=k)
            assert len(results) == k

    def test_retrieve_empty_index_returns_empty(self):
        idx = make_fresh_index()
        encoder = self._make_mock_encoder()
        results = retrieve("any query", index=idx,
                           text_encoder=encoder, top_k=5)
        assert results == []

    def test_retrieve_method_is_neural(self):
        idx = make_populated_index(10)
        encoder = self._make_mock_encoder()
        results = retrieve("crash on login", index=idx,
                           text_encoder=encoder, top_k=3)
        assert all(r.retrieval_method == "neural" for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: Neural vs BM25 comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrieverIntegration:
    def test_neural_and_bm25_return_same_schema(self):
        """Both retrievers must return RetrievalResult with same fields."""
        records = [make_record(i) for i in range(20)]

        # Neural
        idx = make_populated_index(20)
        neural = NeuralRetriever(idx, top_k=5)
        neural_results = neural.search(make_embedded_query())

        # BM25
        bm25 = BM25Retriever(records)
        bm25_results = bm25.search("add numbers offset", top_k=5)

        for results in [neural_results, bm25_results]:
            assert len(results) == 5
            for r in results:
                assert hasattr(r, "id")
                assert hasattr(r, "retrieval_score")
                assert hasattr(r, "rank")
                assert hasattr(r, "rerank_score")

    def test_rerank_score_can_be_set_post_retrieval(self):
        """Simulate reranker setting scores on retrieved results."""
        idx = make_populated_index(10)
        neural = NeuralRetriever(idx, top_k=5)
        results = neural.search(make_embedded_query())

        # Simulate reranker assigning new scores
        for i, result in enumerate(results):
            result.rerank_score = 0.9 - i * 0.1

        assert results[0].rerank_score == pytest.approx(0.9)
        assert results[4].rerank_score == pytest.approx(0.5)

    def test_result_to_dict_roundtrip(self):
        idx = make_populated_index(5)
        neural = NeuralRetriever(idx, top_k=3)
        results = neural.search(make_embedded_query())
        for r in results:
            d = r.to_dict()
            assert d["rank"] == r.rank
            assert d["retrieval_score"] == r.retrieval_score
            assert d["rerank_score"] is None