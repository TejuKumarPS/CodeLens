
"""
tests/test_embedder.py
Full test suite for the embedder module (Milestone 2).

All tests use mock encoders — NO model downloads required.
Mock encoders produce deterministic random vectors of the correct shape
and dtype, allowing us to test all pipeline logic without torch/HuggingFace.

Run:
    pytest tests/test_embedder.py -v
    pytest tests/test_embedder.py -v --cov=embedder
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from data_loader.models import CodeRecord
from embedder.embed_pipeline import (
    embed_records,
    embed_query,
    EmbeddedQuery,
    _pad_to_dim,
    DEFAULT_ALPHA,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_code_record(idx: int = 0) -> CodeRecord:
    return CodeRecord(
        id=f"repo::func_{idx}::train_{idx}",
        func_name=f"func_{idx}",
        func_code=f"def func_{idx}(x): return x + {idx}",
        docstring=f"Returns x plus {idx}. A useful arithmetic helper function.",
        language="python",
        repository="org/repo",
        url="https://github.com/org/repo",
        tokens=["def", f"func_{idx}", "x", "return", "x"],
        partition="train",
    )


def make_mock_text_encoder(dim: int = 768) -> MagicMock:
    """Returns a mock CodeBERTEncoder whose encode/encode_batch return unit vectors."""
    enc = MagicMock()
    enc.embedding_dim = dim

    def _encode(text):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.random(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def _encode_batch(texts, batch_size=16, show_progress=False):
        if not texts:
            return np.zeros((0, dim), dtype=np.float32)
        return np.vstack([_encode(t) for t in texts]).astype(np.float32)

    enc.encode.side_effect = _encode
    enc.encode_batch.side_effect = _encode_batch
    return enc


def make_mock_image_encoder(dim: int = 512) -> MagicMock:
    """Returns a mock CLIPImageEncoder whose encode returns a unit vector."""
    enc = MagicMock()
    enc.embedding_dim = dim

    def _encode(image):
        rng = np.random.default_rng(42)
        v = rng.random(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    enc.encode.side_effect = _encode
    return enc


def make_dummy_image() -> Image.Image:
    """Create a tiny solid-colour PIL image for testing."""
    return Image.new("RGB", (64, 64), color=(128, 64, 200))

# ─────────────────────────────────────────────────────────────────────────────
# CLIPImageEncoder unit tests (mock-based)
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIPImageEncoder:
    def test_encode_returns_512_dim(self):
        enc = make_mock_image_encoder(512)
        img = make_dummy_image()
        vec = enc.encode(img)
        assert vec.shape == (512,)

    def test_encode_is_unit_norm(self):
        enc = make_mock_image_encoder(512)
        img = make_dummy_image()
        vec = enc.encode(img)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_encode_dtype_float32(self):
        enc = make_mock_image_encoder(512)
        img = make_dummy_image()
        vec = enc.encode(img)
        assert vec.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# embed_records tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbedRecords:
    def test_embeddings_set_on_all_records(self):
        records = [make_code_record(i) for i in range(5)]
        enc = make_mock_text_encoder()
        result = embed_records(records, enc, batch_size=2, show_progress=False)
        assert all(r.embedding is not None for r in result)

    def test_embedding_is_list_of_floats(self):
        records = [make_code_record(0)]
        enc = make_mock_text_encoder()
        embed_records(records, enc, show_progress=False)
        emb = records[0].embedding
        assert isinstance(emb, list)
        assert len(emb) == 768
        assert all(isinstance(v, float) for v in emb)

    def test_returns_same_list_object(self):
        records = [make_code_record(i) for i in range(3)]
        enc = make_mock_text_encoder()
        result = embed_records(records, enc, show_progress=False)
        assert result is records   # mutates in-place AND returns same list

    def test_empty_records_handled(self):
        enc = make_mock_text_encoder()
        result = embed_records([], enc, show_progress=False)
        assert result == []

    def test_batch_size_respected(self):
        """Verify encode_batch is called with correct batching."""
        records = [make_code_record(i) for i in range(10)]
        enc = make_mock_text_encoder()
        embed_records(records, enc, batch_size=3, show_progress=False)
        # encode_batch should have been called once with all 10 texts
        enc.encode_batch.assert_called_once()
        call_args = enc.encode_batch.call_args
        assert len(call_args[0][0]) == 10   # first positional arg is the texts list

    def test_embedding_unit_norm(self):
        """Embeddings stored as list should still represent unit vectors."""
        records = [make_code_record(0)]
        enc = make_mock_text_encoder()
        embed_records(records, enc, show_progress=False)
        vec = np.array(records[0].embedding)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_different_functions_get_different_embeddings(self):
        records = [make_code_record(0), make_code_record(1)]
        enc = make_mock_text_encoder()
        embed_records(records, enc, show_progress=False)
        v0 = np.array(records[0].embedding)
        v1 = np.array(records[1].embedding)
        # Different code → different hash seeds → different vectors
        assert not np.allclose(v0, v1)


# ─────────────────────────────────────────────────────────────────────────────
# embed_query tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbedQuery:
    def test_text_only_query(self):
        text_enc = make_mock_text_encoder()
        query = embed_query("app crashes at checkout", text_encoder=text_enc)
        assert isinstance(query, EmbeddedQuery)
        assert query.has_image is False
        assert query.image_vector is None
        assert query.alpha == 1.0
        assert query.fused_vector.shape == (768,)

    def test_text_only_fused_equals_text_vector(self):
        text_enc = make_mock_text_encoder()
        query = embed_query("null pointer exception in cart", text_encoder=text_enc)
        np.testing.assert_array_almost_equal(query.fused_vector, query.text_vector)

    def test_multimodal_query_has_image(self):
        text_enc = make_mock_text_encoder()
        img_enc = make_mock_image_encoder()
        img = make_dummy_image()
        query = embed_query(
            "UI freezes on login screen",
            text_encoder=text_enc,
            image=img,
            image_encoder=img_enc,
        )
        assert query.has_image is True
        assert query.image_vector is not None
        assert query.image_vector.shape == (512,)

    def test_multimodal_fused_vector_shape(self):
        text_enc = make_mock_text_encoder()
        img_enc = make_mock_image_encoder()
        query = embed_query(
            "stack overflow in recursive function",
            text_encoder=text_enc,
            image=make_dummy_image(),
            image_encoder=img_enc,
        )
        assert query.fused_vector.shape == (768,)

    def test_multimodal_fused_is_unit_norm(self):
        text_enc = make_mock_text_encoder()
        img_enc = make_mock_image_encoder()
        query = embed_query(
            "IndexError in list processing",
            text_encoder=text_enc,
            image=make_dummy_image(),
            image_encoder=img_enc,
        )
        assert abs(np.linalg.norm(query.fused_vector) - 1.0) < 1e-5

    def test_multimodal_fused_differs_from_text_only(self):
        """Adding an image should change the fused vector."""
        text_enc = make_mock_text_encoder()
        img_enc = make_mock_image_encoder()

        q_text = embed_query("app crashes", text_encoder=text_enc)
        q_multi = embed_query(
            "app crashes",
            text_encoder=text_enc,
            image=make_dummy_image(),
            image_encoder=img_enc,
        )
        assert not np.allclose(q_text.fused_vector, q_multi.fused_vector)

    def test_alpha_stored_correctly(self):
        text_enc = make_mock_text_encoder()
        img_enc = make_mock_image_encoder()
        query = embed_query(
            "crash on startup",
            text_encoder=text_enc,
            image=make_dummy_image(),
            image_encoder=img_enc,
            alpha=0.8,
        )
        assert query.alpha == 0.8

    def test_default_alpha_is_07(self):
        text_enc = make_mock_text_encoder()
        img_enc = make_mock_image_encoder()
        query = embed_query(
            "crash on startup",
            text_encoder=text_enc,
            image=make_dummy_image(),
            image_encoder=img_enc,
        )
        assert query.alpha == DEFAULT_ALPHA == 0.7

    def test_image_without_encoder_raises(self):
        text_enc = make_mock_text_encoder()
        with pytest.raises(ValueError, match="image_encoder must be provided"):
            embed_query(
                "crash on startup",
                text_encoder=text_enc,
                image=make_dummy_image(),
                image_encoder=None,
            )

    def test_empty_text_raises(self):
        text_enc = make_mock_text_encoder()
        with pytest.raises(ValueError, match="non-empty"):
            embed_query("", text_encoder=text_enc)

    def test_whitespace_text_raises(self):
        text_enc = make_mock_text_encoder()
        with pytest.raises(ValueError, match="non-empty"):
            embed_query("   ", text_encoder=text_enc)

    def test_text_stored_in_query(self):
        text_enc = make_mock_text_encoder()
        query = embed_query("memory leak in parser", text_encoder=text_enc)
        assert query.text == "memory leak in parser"

    def test_text_is_stripped(self):
        text_enc = make_mock_text_encoder()
        query = embed_query("  memory leak  ", text_encoder=text_enc)
        assert query.text == "memory leak"


# ─────────────────────────────────────────────────────────────────────────────
# EmbeddedQuery dataclass tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddedQuery:
    def test_repr_shows_text_preview(self):
        v = np.ones(768, dtype=np.float32) / np.sqrt(768)
        q = EmbeddedQuery(
            text="app crashes at checkout screen",
            text_vector=v,
            fused_vector=v,
            alpha=1.0,
            has_image=False,
        )
        r = repr(q)
        assert "app crashes" in r
        assert "has_image=False" in r

    def test_image_vector_defaults_none(self):
        v = np.ones(768, dtype=np.float32) / np.sqrt(768)
        q = EmbeddedQuery(
            text="test",
            text_vector=v,
            fused_vector=v,
            alpha=1.0,
            has_image=False,
        )
        assert q.image_vector is None


# ─────────────────────────────────────────────────────────────────────────────
# Integration smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbedderIntegration:
    def test_full_pipeline_records_then_query(self):
        """Simulate: embed a corpus, then embed a query, verify shapes align."""
        records = [make_code_record(i) for i in range(8)]
        text_enc = make_mock_text_encoder(768)
        img_enc = make_mock_image_encoder(512)

        # Step 1: embed corpus
        embedded = embed_records(records, text_enc, batch_size=4, show_progress=False)
        assert all(len(r.embedding) == 768 for r in embedded)

        # Step 2: embed text-only query
        q_text = embed_query("TypeError in data processing loop", text_encoder=text_enc)
        assert q_text.fused_vector.shape == (768,)

        # Step 3: embed multimodal query
        q_multi = embed_query(
            "TypeError in data processing loop",
            text_encoder=text_enc,
            image=make_dummy_image(),
            image_encoder=img_enc,
            alpha=0.6,
        )
        assert q_multi.fused_vector.shape == (768,)

        # Step 4: verify query vectors can be compared with corpus vectors
        corpus_matrix = np.array([r.embedding for r in embedded])  # (8, 768)
        scores_text = corpus_matrix @ q_text.fused_vector           # (8,)
        scores_multi = corpus_matrix @ q_multi.fused_vector          # (8,)

        assert scores_text.shape == (8,)
        assert scores_multi.shape == (8,)
        assert np.all(np.abs(scores_text) <= 1.01)   # cosine similarity ≤ 1

# ─────────────────────────────────────────────────────────────────────────────
# _pad_to_dim utility
# ─────────────────────────────────────────────────────────────────────────────

class TestPadToDim:
    def test_pad_smaller_to_larger(self):
        v = np.ones(512, dtype=np.float32)
        v /= np.linalg.norm(v)
        result = _pad_to_dim(v, 768)
        assert result.shape == (768,)

    def test_truncate_larger_to_smaller(self):
        v = np.ones(768, dtype=np.float32)
        result = _pad_to_dim(v, 512)
        assert result.shape == (512,)

    def test_same_dim_returns_copy(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = _pad_to_dim(v, 3)
        assert result.shape == (3,)
        assert not np.shares_memory(result, v)

    def test_output_is_unit_norm(self):
        v = np.random.rand(512).astype(np.float32)
        result = _pad_to_dim(v, 768)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_zero_vector_no_division_error(self):
        v = np.zeros(512, dtype=np.float32)
        result = _pad_to_dim(v, 768)
        assert result.shape == (768,)
        assert np.all(result == 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# CodeBERTEncoder unit tests (mock-based)
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeBERTEncoder:
    def test_encode_returns_768_dim(self):
        enc = make_mock_text_encoder(768)
        vec = enc.encode("def foo(): return 42")
        assert vec.shape == (768,)

    def test_encode_is_unit_norm(self):
        enc = make_mock_text_encoder(768)
        vec = enc.encode("def foo(): return 42")
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_encode_batch_shape(self):
        enc = make_mock_text_encoder(768)
        texts = ["def foo(): pass", "def bar(): return 1", "def baz(x): return x"]
        vecs = enc.encode_batch(texts)
        assert vecs.shape == (3, 768)

    def test_encode_batch_empty(self):
        enc = make_mock_text_encoder(768)
        vecs = enc.encode_batch([])
        assert vecs.shape[0] == 0

    def test_encode_batch_dtype_float32(self):
        enc = make_mock_text_encoder(768)
        vecs = enc.encode_batch(["def foo(): pass"])
        assert vecs.dtype == np.float32

    def test_different_texts_produce_different_vectors(self):
        enc = make_mock_text_encoder(768)
        v1 = enc.encode("def process_checkout(cart): return order")
        v2 = enc.encode("def render_homepage(request): return template")
        # Should not be identical (mock uses hash-based seeding)
        assert not np.allclose(v1, v2)
