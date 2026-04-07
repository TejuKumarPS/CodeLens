"""
tests/test_data_loader.py
=========================
Full test suite for the data_loader module (Milestone 1).

Tests are designed to run WITHOUT downloading the full CodeSearchNet dataset.
Mock data mimics the exact HuggingFace record structure.

Run:
    pytest tests/test_data_loader.py -v
    pytest tests/test_data_loader.py -v --cov=data_loader
"""

import json
import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch, MagicMock

import pytest

# ── Imports under test ────────────────────────────────────────────────────────
from data_loader.models import CodeRecord
from data_loader.cleaner import clean_record, compute_cleaning_stats
from data_loader.pipeline import process_split, save_processed, load_processed


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_raw_record(
    func_name: str = "checkout_handler",
    func_code: str = None,
    docstring: str = "Handles the checkout process for the shopping cart.",
    repository: str = "django/django",
    url: str = "https://github.com/django/django/blob/main/cart.py#L42",
    tokens: List[str] = None,
    language: str = "python",
) -> dict:
    """Build a raw HuggingFace-style record for testing."""
    if func_code is None:
        func_code = (
            "def checkout_handler(request, cart_id):\n"
            "    cart = Cart.objects.get(id=cart_id)\n"
            "    if not cart.items.exists():\n"
            "        raise EmptyCartError('Cart is empty')\n"
            "    order = Order.create_from_cart(cart)\n"
            "    cart.clear()\n"
            "    return order\n"
        )
    if tokens is None:
        tokens = func_code.split()

    return {
        "func_name": func_name,
        "func_code_string": func_code,
        "func_documentation_string": docstring,
        "repository_name": repository,
        "func_code_url": url,
        "func_code_tokens": tokens,
        "language": language,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CodeRecord model tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeRecord:
    def test_basic_construction(self):
        record = CodeRecord(
            id="repo::func::train_0",
            func_name="my_func",
            func_code="def my_func(): pass",
            docstring="Does something useful.",
            language="python",
            repository="org/repo",
            url="https://github.com/org/repo",
            tokens=["def", "my_func", "pass"],
            partition="train",
        )
        assert record.id == "repo::func::train_0"
        assert record.embedding is None

    def test_to_dict_excludes_embedding(self):
        record = CodeRecord(
            id="x::y::train_1",
            func_name="f",
            func_code="def f(): pass",
            docstring="Short docstring here for testing.",
            language="python",
            repository="r",
            url="u",
            tokens=["def", "f", "pass"],
            partition="train",
            embedding=[0.1, 0.2, 0.3],
        )
        d = record.to_dict()
        assert "embedding" not in d
        assert d["func_name"] == "f"

    def test_to_dict_and_from_dict_roundtrip(self):
        record = CodeRecord(
            id="a::b::test_99",
            func_name="handle_error",
            func_code="def handle_error(e): raise e",
            docstring="Propagates exceptions up the call stack.",
            language="python",
            repository="utils/lib",
            url="https://example.com",
            tokens=["def", "handle_error", "raise", "e"],
            partition="test",
        )
        reconstructed = CodeRecord.from_dict(record.to_dict())
        assert reconstructed.id == record.id
        assert reconstructed.func_name == record.func_name
        assert reconstructed.tokens == record.tokens
        assert reconstructed.embedding is None

    def test_chroma_metadata_no_lists(self):
        record = CodeRecord(
            id="x::y::train_0",
            func_name="fetch_data",
            func_code="def fetch_data(): return []",
            docstring="Fetches all data from the database connection pool.",
            language="python",
            repository="myrepo",
            url="http://example.com",
            tokens=["def", "fetch_data", "return"],
            partition="train",
        )
        meta = record.to_chroma_metadata()
        for v in meta.values():
            assert isinstance(v, (str, int, float, bool)), \
                f"ChromaDB metadata value must be scalar, got {type(v)} for {v!r}"

    def test_docstring_preview_truncated(self):
        long_doc = "A" * 500
        record = CodeRecord(
            id="x::y::train_0",
            func_name="f",
            func_code="def f(): pass",
            docstring=long_doc,
            language="python",
            repository="r",
            url="u",
            tokens=["def", "f"],
            partition="train",
        )
        meta = record.to_chroma_metadata()
        assert len(meta["docstring_preview"]) <= 200


# ─────────────────────────────────────────────────────────────────────────────
# Cleaner tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCleaner:
    def test_valid_record_passes(self):
        raw = make_raw_record()
        result = clean_record(raw, idx=0, partition="train")
        assert result is not None
        assert isinstance(result, CodeRecord)

    def test_empty_code_rejected(self):
        raw = make_raw_record(func_code="", tokens=[])
        result = clean_record(raw, idx=1, partition="train")
        assert result is None

    def test_too_few_tokens_rejected(self):
        raw = make_raw_record(func_code="def f(): pass", tokens=["def", "f"])
        result = clean_record(raw, idx=2, partition="train")
        assert result is None

    def test_empty_docstring_rejected(self):
        raw = make_raw_record(docstring="")
        result = clean_record(raw, idx=3, partition="train")
        assert result is None

    def test_short_docstring_rejected(self):
        raw = make_raw_record(docstring="Fix bug.")
        result = clean_record(raw, idx=4, partition="train")
        assert result is None

    def test_long_code_is_truncated_not_rejected(self):
        long_code = "def f():\n" + "    x = 1\n" * 1000  # > 8000 chars
        raw = make_raw_record(
            func_code=long_code,
            tokens=long_code.split()[:50],
        )
        result = clean_record(raw, idx=5, partition="train")
        assert result is not None
        assert len(result.func_code) <= 8_000

    def test_long_docstring_is_truncated(self):
        raw = make_raw_record(docstring="This is a docstring. " * 200)
        result = clean_record(raw, idx=6, partition="train")
        assert result is not None
        assert len(result.docstring) <= 2_000

    def test_id_format(self):
        raw = make_raw_record(func_name="my_func", repository="org/repo")
        result = clean_record(raw, idx=7, partition="validation")
        assert result is not None
        assert "my_func" in result.id
        assert "validation_7" in result.id

    def test_missing_func_name_fallback(self):
        raw = make_raw_record(func_name="")
        raw["func_code_string"] = "def inferred_func(x): return x * 2"
        result = clean_record(raw, idx=8, partition="train")
        assert result is not None
        assert result.func_name == "inferred_func"

    def test_partition_set_correctly(self):
        for partition in ["train", "validation", "test"]:
            raw = make_raw_record()
            result = clean_record(raw, idx=0, partition=partition)
            assert result is not None
            assert result.partition == partition

    def test_tokens_preserved(self):
        tokens = ["def", "checkout", "request", "cart_id", "cart", "get",
                  "order", "create", "return", "order", "clear"]  # 11 tokens >= MIN=10
        raw = make_raw_record(tokens=tokens)
        result = clean_record(raw, idx=0, partition="train")
        assert result is not None
        assert result.tokens == tokens


class TestCleaningStats:
    def test_full_retention(self):
        stats = compute_cleaning_stats(100, 100)
        assert stats["retention_pct"] == 100.0
        assert stats["dropped"] == 0

    def test_partial_retention(self):
        stats = compute_cleaning_stats(1000, 850)
        assert stats["dropped"] == 150
        assert stats["retention_pct"] == 85.0

    def test_zero_raw_no_division_error(self):
        stats = compute_cleaning_stats(0, 0)
        assert stats["retention_pct"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline tests (with mocked HuggingFace data)
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_dataset(n: int = 5) -> List[dict]:
    """Generate n mock raw records with enough tokens to pass quality filters."""
    records = []
    for i in range(n):
        code = (
            f"def function_{i}(x, y, z=0):\n"
            f"    \"\"\"Add numbers together with optional offset.\"\"\"\n"
            f"    result = x + y + z + {i}\n"
            f"    if result < 0:\n"
            f"        return 0\n"
            f"    return result\n"
        )
        records.append(make_raw_record(
            func_name=f"function_{i}",
            func_code=code,
            docstring=f"Computes the sum of x and y with an offset of {i}. Returns integer result.",
            repository=f"org/repo_{i % 3}",
            tokens=code.split(),
        ))
    return records


class TestPipeline:
    def test_process_split_with_mock_data(self):
        mock_records = _make_mock_dataset(10)

        with patch("data_loader.pipeline.load_raw", return_value=iter(mock_records)):
            records = process_split(split="train", language="python", limit=10)

        assert len(records) > 0
        assert all(isinstance(r, CodeRecord) for r in records)
        assert all(r.partition == "train" for r in records)

    def test_deduplication_removes_same_repo_func(self):
        # Same repository + same func_name => only one should survive
        raw1 = make_raw_record(func_name="duplicate_func", repository="same/repo")
        raw2 = make_raw_record(func_name="duplicate_func", repository="same/repo")
        raw3 = make_raw_record(func_name="unique_func", repository="same/repo")

        with patch("data_loader.pipeline.load_raw", return_value=iter([raw1, raw2, raw3])):
            records = process_split(split="train", language="python")

        func_names = [r.func_name for r in records]
        assert func_names.count("duplicate_func") == 1
        assert "unique_func" in func_names

    def test_save_and_load_processed_roundtrip(self):
        mock_records = _make_mock_dataset(5)

        with patch("data_loader.pipeline.load_raw", return_value=iter(mock_records)):
            records = process_split(split="test", language="python")

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = save_processed(records, tmpdir, split="test", language="python")
            assert parquet_path.exists()

            loaded = load_processed(str(parquet_path))
            assert len(loaded) == len(records)

            for orig, restored in zip(records, loaded):
                assert orig.id == restored.id
                assert orig.func_name == restored.func_name
                assert orig.func_code == restored.func_code
                assert orig.docstring == restored.docstring
                assert orig.tokens == restored.tokens

    def test_save_produces_parquet_file(self):
        records = [
            CodeRecord(
                id="r::f::train_0",
                func_name="f",
                func_code="def f(): return 42",
                docstring="Returns the answer to life, the universe, and everything.",
                language="python",
                repository="r",
                url="u",
                tokens=["def", "f", "return", "42"],
                partition="train",
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_processed(records, tmpdir, split="train", language="python")
            assert path.suffix == ".parquet"
            assert path.stat().st_size > 0

    def test_load_processed_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_processed("/nonexistent/path/file.parquet")

    def test_empty_records_handled_gracefully(self):
        # All records fail cleaning (empty code)
        bad_records = [make_raw_record(func_code="", tokens=[]) for _ in range(5)]

        with patch("data_loader.pipeline.load_raw", return_value=iter(bad_records)):
            records = process_split(split="train", language="python")

        assert records == []


# ─────────────────────────────────────────────────────────────────────────────
# Integration smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationSmoke:
    def test_full_pipeline_end_to_end(self):
        """Simulate full pipeline: mock data → clean → save → load → verify."""
        mock_raws = _make_mock_dataset(20)

        with patch("data_loader.pipeline.load_raw", return_value=iter(mock_raws)):
            records = process_split(split="validation", language="python", limit=20)

        assert len(records) >= 1

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_processed(records, tmpdir, split="validation", language="python")
            reloaded = load_processed(str(path))

        assert len(reloaded) == len(records)
        ids_orig = {r.id for r in records}
        ids_reloaded = {r.id for r in reloaded}
        assert ids_orig == ids_reloaded

        # Verify chroma metadata is always scalar-valued
        for r in reloaded:
            for k, v in r.to_chroma_metadata().items():
                assert isinstance(v, (str, int, float, bool)), \
                    f"Non-scalar metadata at key={k}: {type(v)}"