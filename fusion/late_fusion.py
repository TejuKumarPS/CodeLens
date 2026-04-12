
"""
fusion.late_fusion
==================
LateFusion implements the weighted addition formula from the requirements doc:

    fused_vector = alpha * text_vector + (1 - alpha) * projected_image_vector

where projected_image_vector = LinearProjection.project(image_vector).

Key design decisions:
  - Alpha is validated to [0.0, 1.0]
  - Alpha = 1.0 → text only (image ignored), used when no screenshot provided
  - Alpha = 0.0 → image only (text ignored), not recommended but supported
  - Default alpha = 0.7 (text-dominant, from requirements doc)
  - Output is always L2-normalised so cosine similarity == dot product

Integration with M2 (embedder):
  This module REPLACES the _pad_to_dim() stub in embedder/embed_pipeline.py.
  The API (M8) will use LateFusion directly instead of calling embed_query().

Usage
-----
    from fusion.late_fusion import LateFusion
    from fusion.projection import LinearProjection
    import numpy as np

    proj   = LinearProjection()
    fusion = LateFusion(projection=proj, alpha=0.7)

    text_vec = np.random.randn(768).astype(np.float32)
    text_vec /= np.linalg.norm(text_vec)
    img_vec  = np.random.randn(512).astype(np.float32)
    img_vec  /= np.linalg.norm(img_vec)

    fused = fusion.fuse(text_vec, img_vec)
    print(fused.shape, np.linalg.norm(fused))  # (768,) ≈ 1.0
"""

import logging
from typing import Optional

import numpy as np

from .projection import LinearProjection

logger = logging.getLogger(__name__)

DEFAULT_ALPHA    = 0.7
TEXT_DIM         = 768
IMAGE_DIM        = 512


class LateFusion:
    """
    Fuses a text embedding and a projected image embedding via weighted addition.

    Parameters
    ----------
    projection : LinearProjection
        Projects image vectors from IMAGE_DIM → TEXT_DIM.
    alpha : float
        Weight for the text vector. Must be in [0.0, 1.0].
        Default 0.7 (text-dominant).
    """

    def __init__(
        self,
        projection: Optional[LinearProjection] = None,
        alpha: float = DEFAULT_ALPHA,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"alpha must be in [0.0, 1.0], got {alpha}."
            )
        self._projection = projection or LinearProjection()
        self._alpha      = alpha
        logger.info(
            "LateFusion ready. alpha=%.2f, projection=%r",
            alpha, self._projection,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def fuse(
        self,
        text_vector: np.ndarray,
        image_vector: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """
        Fuse text and (optional) image vectors into a single query vector.

        Parameters
        ----------
        text_vector : np.ndarray, shape (768,)
            L2-normalised CodeBERT embedding.
        image_vector : np.ndarray, shape (512,), optional
            L2-normalised CLIP image embedding.
            If None, returns text_vector unchanged (alpha forced to 1.0).
        alpha : float, optional
            Override the instance alpha for this call only.

        Returns
        -------
        np.ndarray, shape (768,)
            L2-normalised fused vector.

        Raises
        ------
        ValueError
            If text_vector has wrong shape, or alpha out of range.
        """
        self._validate_text_vector(text_vector)

        effective_alpha = self._resolve_alpha(alpha)

        # Text-only mode: no image provided
        if image_vector is None:
            return text_vector.copy().astype(np.float32)

        # Validate + project image vector
        self._validate_image_vector(image_vector)
        projected_img = self._projection.project(image_vector)   # (768,)

        # Weighted addition
        fused = (
            effective_alpha * text_vector.astype(np.float32)
            + (1.0 - effective_alpha) * projected_img
        )

        # L2-normalise
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm

        return fused.astype(np.float32)

    def fuse_batch(
        self,
        text_vectors: np.ndarray,
        image_vectors: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """
        Fuse a batch of text and image vectors.

        Parameters
        ----------
        text_vectors : np.ndarray, shape (N, 768)
        image_vectors : np.ndarray, shape (N, 512), optional

        Returns
        -------
        np.ndarray, shape (N, 768)
            Row-wise L2-normalised fused vectors.
        """
        if text_vectors.ndim != 2 or text_vectors.shape[1] != TEXT_DIM:
            raise ValueError(
                f"text_vectors must be shape (N, {TEXT_DIM}), got {text_vectors.shape}"
            )

        effective_alpha = self._resolve_alpha(alpha)

        if image_vectors is None:
            return text_vectors.astype(np.float32).copy()

        if image_vectors.ndim != 2 or image_vectors.shape[1] != IMAGE_DIM:
            raise ValueError(
                f"image_vectors must be shape (N, {IMAGE_DIM}), got {image_vectors.shape}"
            )
        if text_vectors.shape[0] != image_vectors.shape[0]:
            raise ValueError(
                f"Batch size mismatch: text_vectors has {text_vectors.shape[0]} rows, "
                f"image_vectors has {image_vectors.shape[0]} rows."
            )

        # Project all image vectors at once
        projected_imgs = self._projection.project_batch(image_vectors)  # (N, 768)

        fused = (
            effective_alpha * text_vectors.astype(np.float32)
            + (1.0 - effective_alpha) * projected_imgs
        )

        # Row-wise L2 normalise
        norms = np.linalg.norm(fused, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (fused / norms).astype(np.float32)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _resolve_alpha(self, override: Optional[float]) -> float:
        if override is None:
            return self._alpha
        if not 0.0 <= override <= 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {override}.")
        return override

    @staticmethod
    def _validate_text_vector(v: np.ndarray) -> None:
        if v.ndim != 1 or v.shape[0] != TEXT_DIM:
            raise ValueError(
                f"text_vector must be shape ({TEXT_DIM},), got {v.shape}."
            )

    @staticmethod
    def _validate_image_vector(v: np.ndarray) -> None:
        if v.ndim != 1 or v.shape[0] != IMAGE_DIM:
            raise ValueError(
                f"image_vector must be shape ({IMAGE_DIM},), got {v.shape}."
            )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {value}.")
        self._alpha = value

    @property
    def projection(self) -> LinearProjection:
        return self._projection

    def __repr__(self) -> str:
        return f"LateFusion(alpha={self._alpha}, projection={self._projection!r})"