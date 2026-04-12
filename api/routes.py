"""
api.routes
==========
All FastAPI route handlers for CodeLens.

Endpoints
---------
POST /search
    Main retrieval endpoint. Accepts a bug report (text + optional image),
    runs the full pipeline (embed → fuse → retrieve → rerank), returns
    top-N ranked code functions.

GET /health
    Liveness/readiness probe. Returns model and index load status.

GET /index/stats
    Returns ChromaDB collection metadata (size, name, persist dir).

POST /evaluate
    Admin endpoint: runs evaluation over a test Parquet file and returns
    MRR/NDCG/Precision metrics. Requires models to be loaded.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status

from retriever.neural import NeuralRetriever
from reranker.rerank import rerank

from .schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    HealthResponse,
    IndexStatsResponse,
    EvaluateRequest,
    EvaluateResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Helper ────────────────────────────────────────────────────────────────────

def _decode_image(image_b64: str):
    """
    Decode a base64-encoded image string into a PIL Image.

    Raises HTTPException(400) if the data is invalid or not an image.
    """
    try:
        from PIL import Image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image_b64: could not decode image. {e}",
        )


def _get_state(request: Request):
    """Extract AppState from app.state, raising 503 if not ready."""
    state = getattr(request.app.state, "codelens", None)
    if state is None or not state.models_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not loaded. Server is still initialising.",
        )
    return state


# ── POST /search ──────────────────────────────────────────────────────────────

@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request) -> SearchResponse:
    """
    Retrieve the most relevant code functions for a bug report.

    Pipeline:
      1. Decode image (if provided)
      2. Encode query text → text_vector (CodeBERT)
      3. Encode image → image_vector (CLIP) — if provided
      4. Fuse text + image → fused_vector (LateFusion)
      5. ANN search in ChromaDB → top_k_retrieval candidates
      6. Re-rank with cross-encoder → top_n_results
      7. Return structured JSON
    """
    state = _get_state(request)

    # 1. Decode image
    image = None
    if body.image_b64:
        if state.image_encoder is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Image encoder not loaded. Restart server with load_image_encoder=True.",
            )
        image = _decode_image(body.image_b64)

    # 2-3. Encode text (and optionally image)
    from embedder.embed_pipeline import embed_query
    embedded = embed_query(
        text=body.query,
        text_encoder=state.text_encoder,
        image=image,
        image_encoder=state.image_encoder if image else None,
        alpha=body.alpha,
    )

    # 4. Override with proper LateFusion if image provided
    if image and state.fusion and state.image_encoder:
        import numpy as np
        fused_vec = state.fusion.fuse(
            text_vector=embedded.text_vector,
            image_vector=embedded.image_vector,
            alpha=body.alpha,
        )
        # Patch the embedded query's fused vector
        embedded = embedded.__class__(
            text=embedded.text,
            text_vector=embedded.text_vector,
            image_vector=embedded.image_vector,
            fused_vector=fused_vec,
            alpha=body.alpha,
            has_image=True,
        )

    # 5. ANN retrieval
    neural = NeuralRetriever(index=state.index, top_k=body.top_k_retrieval)
    candidates = neural.search(embedded)

    # Apply language filter if requested
    if body.language:
        candidates = [c for c in candidates if c.language == body.language]

    if not candidates:
        return SearchResponse(
            query=body.query,
            has_image=image is not None,
            alpha=body.alpha,
            results=[],
            num_results=0,
        )

    # 6. Re-rank
    if state.reranker:
        candidates = rerank(
            query_text=body.query,
            candidates=candidates,
            reranker=state.reranker,
            top_n=body.top_n_results,
        )
        method = "neural+rerank"
    else:
        candidates = candidates[: body.top_n_results]
        method = "neural"

    # 7. Build response
    results = [
        SearchResultItem(
            rank=c.rank,
            func_name=c.func_name,
            repository=c.repository,
            url=c.url,
            language=c.language,
            docstring_preview=c.docstring_preview,
            code_preview=c.code_preview,
            retrieval_score=c.retrieval_score,
            rerank_score=c.rerank_score,
            retrieval_method=c.retrieval_method,
        )
        for c in candidates
    ]

    return SearchResponse(
        query=body.query,
        has_image=image is not None,
        alpha=body.alpha,
        results=results,
        num_results=len(results),
        retrieval_method=method,
    )


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness and readiness probe."""
    state = getattr(request.app.state, "codelens", None)
    if state is None:
        return HealthResponse(
            status="initialising",
            index_loaded=False,
            index_size=0,
            models_loaded=False,
        )
    return HealthResponse(
        status="ok",
        index_loaded=state.index is not None,
        index_size=state.index_size,
        models_loaded=state.models_loaded,
    )


# ── GET /index/stats ──────────────────────────────────────────────────────────

@router.get("/index/stats", response_model=IndexStatsResponse)
async def index_stats(request: Request) -> IndexStatsResponse:
    """Return metadata about the loaded ChromaDB collection."""
    state = _get_state(request)
    return IndexStatsResponse(
        collection_name=state.index.collection_name,
        num_vectors=state.index_size,
        persist_dir=state.index.persist_dir,
    )


# ── POST /evaluate ────────────────────────────────────────────────────────────

@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(body: EvaluateRequest, request: Request) -> EvaluateResponse:
    """
    Run IR evaluation over a test Parquet file.

    This is an admin/development endpoint — not intended for end-user traffic.
    Can take several minutes for large limit values.
    """
    state = _get_state(request)

    from pathlib import Path
    if not Path(body.parquet_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Parquet file not found: {body.parquet_path}",
        )

    from data_loader.pipeline import load_processed
    from evaluator.evaluator import Evaluator

    test_records = load_processed(body.parquet_path)
    ev = Evaluator(
        index=state.index,
        text_encoder=state.text_encoder,
        k=body.k,
        reranker=state.reranker,
    )
    metrics = ev.run(
        query_records=test_records,
        limit=body.limit,
        show_progress=False,
    )

    return EvaluateResponse(
        mrr=metrics.mrr,
        ndcg_at_k=metrics.ndcg_at_k,
        precision_at_k=metrics.precision_at_k,
        k=metrics.k,
        num_queries=metrics.num_queries,
        num_queries_with_hit=metrics.num_queries_with_hit,
        hit_rate=metrics.hit_rate,
        method=metrics.method,
    )