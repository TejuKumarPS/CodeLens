
"""
api.dependencies
================
Manages singleton instances of the heavy components:
  - CodeBERTEncoder      (loads ~500 MB model once)
  - CLIPImageEncoder     (loads ~350 MB model once)
  - CrossEncoderReranker (loads ~80 MB model once)
  - CodeLensIndex        (connects to ChromaDB on disk)
  - LateFusion           (lightweight, holds projection matrix)

FastAPI dependency injection pattern:
  Each route handler receives these via Depends(get_app_state).
  The AppState is populated at startup via the lifespan context manager.

AppState is stored on app.state so it survives across requests without
re-initialising models.

Design:
  - Models are loaded lazily on first request if not pre-loaded at startup
  - The index must be pre-built (run indexer.build first)
  - All fields are Optional so health checks can report partial readiness
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """
    Holds all singleton model and index instances for the lifetime of the app.

    Populated by load_components() at startup.
    """
    text_encoder:  Optional[object] = field(default=None)
    image_encoder: Optional[object] = field(default=None)
    reranker:      Optional[object] = field(default=None)
    index:         Optional[object] = field(default=None)
    fusion:        Optional[object] = field(default=None)

    @property
    def models_loaded(self) -> bool:
        return self.text_encoder is not None and self.index is not None

    @property
    def index_size(self) -> int:
        if self.index is None:
            return 0
        return self.index.count()


def load_components(
    chroma_dir: str = "chroma_db",
    collection_name: str = "codelens_python",
    device: Optional[str] = None,
    load_image_encoder: bool = True,
    load_reranker: bool = True,
) -> AppState:
    """
    Load all heavy components and return a populated AppState.

    Called once at application startup via the lifespan handler.

    Parameters
    ----------
    chroma_dir : str
        Path to the ChromaDB persist directory.
    collection_name : str
        ChromaDB collection to connect to.
    device : str, optional
        Force CPU/CUDA/MPS. Auto-detected if None.
    load_image_encoder : bool
        Whether to load CLIP (set False for text-only deployments).
    load_reranker : bool
        Whether to load the cross-encoder (set False to skip reranking).

    Returns
    -------
    AppState
    """
    from indexer.chroma_index import CodeLensIndex
    from embedder.code_encoder import CodeBERTEncoder
    from fusion.late_fusion import LateFusion
    from fusion.projection import LinearProjection

    state = AppState()

    logger.info("Loading CodeBERT encoder...")
    state.text_encoder = CodeBERTEncoder(device=device)

    logger.info("Connecting to ChromaDB at %s...", chroma_dir)
    state.index = CodeLensIndex(
        persist_dir=chroma_dir,
        collection_name=collection_name,
    )
    logger.info("Index loaded: %d vectors", state.index_size)

    if load_image_encoder:
        try:
            from embedder.image_encoder import CLIPImageEncoder
            logger.info("Loading CLIP image encoder...")
            state.image_encoder = CLIPImageEncoder(device=device)
        except Exception as e:
            logger.warning("CLIP encoder failed to load: %s", e)

    if load_reranker:
        try:
            from reranker.cross_encoder import CrossEncoderReranker
            logger.info("Loading cross-encoder reranker...")
            state.reranker = CrossEncoderReranker(device=device)
        except Exception as e:
            logger.warning("Reranker failed to load: %s", e)

    proj = LinearProjection(input_dim=512, output_dim=768, init="orthogonal", seed=42)
    state.fusion = LateFusion(projection=proj, alpha=0.7)

    logger.info("All components loaded. models_loaded=%s", state.models_loaded)
    return state