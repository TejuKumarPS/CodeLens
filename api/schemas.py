"""
api.schemas
===========
Pydantic v2 request and response models for all API endpoints.

Keeping schemas in a separate file from the route handlers means:
  - The React frontend (M9) can be built against these contracts
  - Tests can import schemas independently of the full app
  - Schema changes are isolated from routing logic
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Search request ────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """
    Body of POST /search.

    Fields
    ------
    query : str
        Natural language bug report description. Required, non-empty.
    image_b64 : str, optional
        Base64-encoded screenshot image (PNG or JPEG).
        If omitted, the system runs in text-only mode.
    top_k_retrieval : int
        Number of candidates to retrieve before reranking. Default 20.
    top_n_results : int
        Number of final results to return after reranking. Default 5.
    alpha : float
        Late-fusion weight for text vs image. Default 0.7.
        1.0 = text only, 0.0 = image only.
    language : str, optional
        Filter results by programming language (e.g. "python").
        If omitted, no language filter is applied.
    """

    query: str = Field(..., min_length=1, max_length=2000,
                       description="Bug report text")
    image_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded screenshot (PNG/JPEG)"
    )
    top_k_retrieval: int = Field(default=20, ge=1, le=100)
    top_n_results: int   = Field(default=5,  ge=1, le=20)
    alpha: float         = Field(default=0.7, ge=0.0, le=1.0)
    language: Optional[str] = Field(default=None)

    @field_validator("query")
    @classmethod
    def query_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank.")
        return v.strip()

    @field_validator("language")
    @classmethod
    def language_lowercase(cls, v: Optional[str]) -> Optional[str]:
        return v.lower() if v else None

    model_config = {"json_schema_extra": {
        "example": {
            "query": "App crashes at checkout when cart is empty",
            "top_k_retrieval": 20,
            "top_n_results": 5,
            "alpha": 0.7,
        }
    }}


# ── Search result item ────────────────────────────────────────────────────────

class SearchResultItem(BaseModel):
    """A single ranked result returned by /search."""

    rank: int
    func_name: str
    repository: str
    url: str
    language: str
    docstring_preview: str
    code_preview: str
    retrieval_score: float
    rerank_score: Optional[float] = None
    retrieval_method: str


# ── Search response ───────────────────────────────────────────────────────────

class SearchResponse(BaseModel):
    """Response body of POST /search."""

    query: str
    has_image: bool
    alpha: float
    results: List[SearchResultItem]
    num_results: int
    retrieval_method: str = "neural+rerank"

    model_config = {"json_schema_extra": {
        "example": {
            "query": "App crashes at checkout",
            "has_image": False,
            "alpha": 0.7,
            "results": [],
            "num_results": 0,
        }
    }}


# ── Health response ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body of GET /health."""

    status: str = "ok"
    index_loaded: bool
    index_size: int
    models_loaded: bool


# ── Index stats response ──────────────────────────────────────────────────────

class IndexStatsResponse(BaseModel):
    """Response body of GET /index/stats."""

    collection_name: str
    num_vectors: int
    persist_dir: str


# ── Evaluate request/response ─────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    """Body of POST /evaluate."""

    parquet_path: str = Field(
        ...,
        description="Path to test split Parquet file"
    )
    limit: Optional[int] = Field(
        default=100,
        ge=1,
        description="Number of queries to evaluate (default 100)"
    )
    k: int = Field(default=10, ge=1, le=100)


class EvaluateResponse(BaseModel):
    """Response body of POST /evaluate."""

    mrr: float
    ndcg_at_k: float
    precision_at_k: float
    k: int
    num_queries: int
    num_queries_with_hit: int
    hit_rate: float
    method: str