
"""
tests/test_api_routes.py
========================
HTTP endpoint tests for all CodeLens API routes.

Uses FastAPI's TestClient (backed by httpx) with a fully mocked AppState
so no models, ChromaDB, or files are needed.

Run:
    pytest tests/test_api_routes.py -v
"""

import base64
import io
import json
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.dependencies import AppState
from api.schemas import SearchRequest
from retriever.models import RetrievalResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_result(rank: int, score: float = 0.9, rerank: float = 0.85) -> RetrievalResult:
    return RetrievalResult(
        id=f"repo::func_{rank}::train_{rank}",
        func_name=f"func_{rank}",
        repository="org/repo",
        url=f"https://github.com/org/repo/f{rank}.py",
        language="python",
        docstring_preview=f"Function {rank} description.",
        code_preview=f"def func_{rank}(x): return x",
        document=f"def func_{rank}(x): return x + {rank}",
        retrieval_score=score,
        rank=rank,
        rerank_score=rerank,
    )


def make_mock_state(
    index_size: int = 100,
    models_loaded: bool = True,
    with_reranker: bool = True,
) -> AppState:
    state = AppState()

    # Mock text encoder
    import numpy as np
    mock_encoder = MagicMock()
    vec = np.random.randn(768).astype(np.float32)
    vec /= np.linalg.norm(vec)
    mock_encoder.encode.return_value = vec
    mock_encoder.encode_batch.return_value = vec.reshape(1, -1)
    state.text_encoder = mock_encoder

    # Mock index
    mock_index = MagicMock()
    mock_index.count.return_value = index_size
    mock_index.collection_name = "codelens_python"
    mock_index.persist_dir = "chroma_db/"
    state.index = mock_index

    # Mock reranker
    if with_reranker:
        mock_reranker = MagicMock()
        mock_reranker.score_pairs.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
        state.reranker = mock_reranker

    return state


def make_test_client(state: AppState = None) -> TestClient:
    """Build a TestClient with a fresh app and injected AppState."""
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"],
        allow_methods=["*"], allow_headers=["*"],
    )
    app.include_router(router, prefix="/api/v1")
    app.state.codelens = state or make_mock_state()
    return TestClient(app)


def make_dummy_image_b64() -> str:
    """Create a tiny valid PNG image encoded as base64."""
    from PIL import Image
    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── GET /health ───────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_ok(self):
        client = make_test_client()
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is True
        assert data["index_size"] == 100

    def test_health_no_state(self):
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        # No app.state.codelens set
        client = TestClient(app)
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "initialising"
        assert data["models_loaded"] is False

    def test_health_response_keys(self):
        client = make_test_client()
        data = client.get("/api/v1/health").json()
        for key in ["status", "index_loaded", "index_size", "models_loaded"]:
            assert key in data


# ── GET /index/stats ──────────────────────────────────────────────────────────

class TestIndexStatsEndpoint:
    def test_returns_200(self):
        client = make_test_client()
        resp = client.get("/api/v1/index/stats")
        assert resp.status_code == 200

    def test_response_fields(self):
        client = make_test_client()
        data = client.get("/api/v1/index/stats").json()
        assert data["collection_name"] == "codelens_python"
        assert data["num_vectors"] == 100
        assert "persist_dir" in data

    def test_503_when_not_loaded(self):
        state = AppState()   # empty state, models_loaded = False
        client = make_test_client(state)
        resp = client.get("/api/v1/index/stats")
        assert resp.status_code == 503


# ── POST /search ──────────────────────────────────────────────────────────────

class TestSearchEndpoint:
    def _search(self, client: TestClient, **kwargs) -> dict:
        body = {"query": "app crashes at checkout", **kwargs}
        resp = client.post("/api/v1/search", json=body)
        return resp

    def test_text_only_returns_200(self):
        import numpy as np
        state = make_mock_state()
        mock_results = [make_result(i + 1) for i in range(5)]

        vec = np.random.randn(768).astype(np.float32)
        vec /= np.linalg.norm(vec)
        mock_embedded = MagicMock()
        mock_embedded.text_vector = vec
        mock_embedded.fused_vector = vec
        mock_embedded.image_vector = None
        mock_embedded.has_image = False
        mock_embedded.text = "app crashes at checkout"

        with patch("embedder.embed_pipeline.embed_query", return_value=mock_embedded), \
             patch("api.routes.NeuralRetriever") as MockNeural, \
             patch("api.routes.rerank", return_value=mock_results[:5]):
            MockNeural.return_value.search.return_value = mock_results
            client = make_test_client(state)
            resp = self._search(client)

        assert resp.status_code == 200

    def test_response_schema(self):
        import numpy as np
        state = make_mock_state()
        mock_results = [make_result(i + 1) for i in range(3)]

        vec = np.random.randn(768).astype(np.float32)
        vec /= np.linalg.norm(vec)
        mock_embedded = MagicMock()
        mock_embedded.text_vector = vec
        mock_embedded.fused_vector = vec
        mock_embedded.image_vector = None
        mock_embedded.has_image = False
        mock_embedded.text = "crash"

        with patch("embedder.embed_pipeline.embed_query", return_value=mock_embedded), \
             patch("api.routes.NeuralRetriever") as MockNeural, \
             patch("api.routes.rerank", return_value=mock_results):
            MockNeural.return_value.search.return_value = mock_results
            client = make_test_client(state)
            data = self._search(client).json()

        assert "query" in data
        assert "results" in data
        assert "num_results" in data
        assert "has_image" in data
        assert "alpha" in data

    def test_empty_results_handled(self):
        import numpy as np
        state = make_mock_state()

        vec = np.random.randn(768).astype(np.float32)
        vec /= np.linalg.norm(vec)
        mock_embedded = MagicMock()
        mock_embedded.text_vector = vec
        mock_embedded.fused_vector = vec
        mock_embedded.image_vector = None
        mock_embedded.has_image = False
        mock_embedded.text = "crash"

        with patch("embedder.embed_pipeline.embed_query", return_value=mock_embedded), \
             patch("api.routes.NeuralRetriever") as MockNeural:
            MockNeural.return_value.search.return_value = []
            client = make_test_client(state)
            data = self._search(client).json()

        assert data["num_results"] == 0
        assert data["results"] == []

    def test_empty_query_returns_422(self):
        client = make_test_client()
        resp = client.post("/api/v1/search", json={"query": ""})
        assert resp.status_code == 422

    def test_missing_query_returns_422(self):
        client = make_test_client()
        resp = client.post("/api/v1/search", json={})
        assert resp.status_code == 422

    def test_503_when_not_loaded(self):
        client = make_test_client(AppState())
        resp = client.post("/api/v1/search", json={"query": "crash"})
        assert resp.status_code == 503

    def test_invalid_image_b64_returns_400(self):
        import numpy as np
        state = make_mock_state()
        state.image_encoder = MagicMock()

        vec = np.random.randn(768).astype(np.float32)
        vec /= np.linalg.norm(vec)
        mock_embedded = MagicMock()
        mock_embedded.text_vector = vec
        mock_embedded.fused_vector = vec
        mock_embedded.image_vector = None
        mock_embedded.has_image = False
        mock_embedded.text = "crash"

        with patch("embedder.embed_pipeline.embed_query", return_value=mock_embedded):
            client = make_test_client(state)
            resp = client.post("/api/v1/search", json={
                "query": "crash",
                "image_b64": "!!!not_valid_base64!!!",
            })

        assert resp.status_code == 400

    def test_alpha_default_in_response(self):
        import numpy as np
        state = make_mock_state()

        vec = np.random.randn(768).astype(np.float32)
        vec /= np.linalg.norm(vec)
        mock_embedded = MagicMock()
        mock_embedded.text_vector = vec
        mock_embedded.fused_vector = vec
        mock_embedded.image_vector = None
        mock_embedded.has_image = False
        mock_embedded.text = "crash"

        with patch("embedder.embed_pipeline.embed_query", return_value=mock_embedded), \
             patch("api.routes.NeuralRetriever") as MockNeural:
            MockNeural.return_value.search.return_value = []
            client = make_test_client(state)
            data = self._search(client).json()

        assert data["alpha"] == pytest.approx(0.7)


# ── POST /evaluate ────────────────────────────────────────────────────────────

class TestEvaluateEndpoint:
    def test_404_when_parquet_missing(self):
        client = make_test_client()
        resp = client.post("/api/v1/evaluate", json={
            "parquet_path": "/nonexistent/path.parquet",
            "limit": 5,
        })
        assert resp.status_code == 404

    def test_503_when_not_loaded(self):
        client = make_test_client(AppState())
        resp = client.post("/api/v1/evaluate", json={
            "parquet_path": "data/processed/python_test.parquet"
        })
        assert resp.status_code == 503

    def test_evaluate_runs_with_mocked_pipeline(self):
        import tempfile, pandas as pd, json as _json, numpy as np
        from pathlib import Path

        state = make_mock_state()

        # Create a minimal parquet file
        rows = [{
            "id": f"r::f_{i}::test_{i}", "func_name": f"f_{i}",
            "func_code": f"def f_{i}(): pass",
            "docstring": f"Function {i} does arithmetic.",
            "language": "python", "repository": "org/repo",
            "url": "https://github.com", "tokens": "[]", "partition": "test",
        } for i in range(3)]
        df = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.parquet")
            df.to_parquet(path, index=False)

            with patch("retriever.retrieve.retrieve", return_value=[]):
                client = make_test_client(state)
                resp = client.post("/api/v1/evaluate", json={
                    "parquet_path": path, "limit": 3, "k": 5,
                })

        assert resp.status_code == 200
        data = resp.json()
        for key in ["mrr", "ndcg_at_k", "precision_at_k", "k",
                    "num_queries", "hit_rate"]:
            assert key in data