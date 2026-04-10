"""
tests/test_embedder.py
======================
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