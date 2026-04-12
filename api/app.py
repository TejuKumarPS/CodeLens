
"""
api.app
=======
FastAPI application factory.

create_app() returns a configured FastAPI instance with:
  - CORS middleware (for React frontend on localhost:3000)
  - Lifespan context manager (load models at startup, log shutdown)
  - All routes from api.routes mounted under /api/v1
  - OpenAPI docs at /docs and /redoc

Usage
-----
    # Development server
    uvicorn api.app:app --reload --port 8000

    # Production
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 1

    # Programmatic (for tests)
    from api.app import create_app
    app = create_app(chroma_dir="chroma_db/", load_reranker=False)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import AppState, load_components
from .routes import router

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def create_app(
    chroma_dir: str = "chroma_db",
    collection_name: str = "codelens_python",
    device: Optional[str] = None,
    load_image_encoder: bool = True,
    load_reranker: bool = True,
    cors_origins: Optional[list] = None,
) -> FastAPI:
    """
    Create and configure the CodeLens FastAPI application.

    Parameters
    ----------
    chroma_dir : str
        Path to ChromaDB persist directory.
    collection_name : str
        ChromaDB collection name.
    device : str, optional
        Torch device override ("cpu", "cuda", "mps").
    load_image_encoder : bool
        Load CLIP encoder for multimodal queries.
    load_reranker : bool
        Load cross-encoder reranker.
    cors_origins : list, optional
        Allowed CORS origins. Defaults to localhost dev origins.

    Returns
    -------
    FastAPI
        Fully configured application instance.
    """

    # ── Lifespan: load models at startup ──────────────────────────────────────
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("CodeLens API starting up...")
        try:
            state = load_components(
                chroma_dir=chroma_dir,
                collection_name=collection_name,
                device=device,
                load_image_encoder=load_image_encoder,
                load_reranker=load_reranker,
            )
            app.state.codelens = state
            logger.info(
                "Startup complete. Index: %d vectors, models_loaded=%s",
                state.index_size, state.models_loaded,
            )
        except Exception as e:
            logger.error("Startup failed: %s", e)
            # Store an empty state so /health can still respond
            app.state.codelens = AppState()

        yield

        logger.info("CodeLens API shutting down.")

    # ── Application ────────────────────────────────────────────────────────────
    app = FastAPI(
        title="CodeLens API",
        description=(
            "Cross-Modal Bug Report Retrieval using Fused Text and "
            "Visual Crash Evidence. PES University | AIR Final Project."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    origins = cors_origins or [
        "http://localhost:3000",    # React dev server
        "http://localhost:5173",    # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")

    return app


# Module-level app instance for uvicorn
# Uses environment variables for configuration
app = create_app(
    chroma_dir=os.getenv("CODELENS_CHROMA_DIR", "chroma_db"),
    collection_name=os.getenv("CODELENS_COLLECTION", "codelens_python"),
    device=os.getenv("CODELENS_DEVICE", None),
    load_image_encoder=os.getenv("CODELENS_LOAD_IMAGE", "true").lower() == "true",
    load_reranker=os.getenv("CODELENS_LOAD_RERANKER", "true").lower() == "true",
)