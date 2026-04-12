
"""
fusion.fuse
===========
Convenience functions consumed by the API (M8) and evaluator (M7).

fuse()
  One-line call that instantiates LateFusion (or reuses a provided one)
  and returns a fused vector. Designed to replace embed_query()'s
  internal _pad_to_dim() with a proper projection.

alpha_sweep()
  Sweeps a list of alpha values and returns one fused vector per alpha.
  Used by the evaluator (M7) to find the alpha that maximises MRR/NDCG.

  Typical sweep:  [0.0, 0.1, 0.2, ..., 1.0]  (11 values)
  For each alpha, the evaluator runs retrieval and records the metric.
  The best alpha is then set as the default for production.

Usage
-----
    from fusion import fuse, alpha_sweep
    import numpy as np

    text_vec = np.random.randn(768).astype(np.float32)
    text_vec /= np.linalg.norm(text_vec)
    img_vec  = np.random.randn(512).astype(np.float32)
    img_vec  /= np.linalg.norm(img_vec)

    # Single fusion
    fused = fuse(text_vec, img_vec, alpha=0.7)

    # Alpha sweep
    results = alpha_sweep(text_vec, img_vec, alphas=[0.3, 0.5, 0.7, 0.9])
    for alpha, vec in results.items():
        print(f"alpha={alpha:.1f} → norm={np.linalg.norm(vec):.4f}")
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from .projection import LinearProjection
from .late_fusion import LateFusion, DEFAULT_ALPHA, TEXT_DIM, IMAGE_DIM

logger = logging.getLogger(__name__)

# Module-level singleton so projection matrix is initialised only once
_DEFAULT_FUSION: Optional[LateFusion] = None


def _get_default_fusion() -> LateFusion:
    """Return (or create) the module-level LateFusion singleton."""
    global _DEFAULT_FUSION
    if _DEFAULT_FUSION is None:
        _DEFAULT_FUSION = LateFusion(
            projection=LinearProjection(
                input_dim=IMAGE_DIM,
                output_dim=TEXT_DIM,
                init="orthogonal",
                seed=42,
            ),
            alpha=DEFAULT_ALPHA,
        )
    return _DEFAULT_FUSION


def fuse(
    text_vector: np.ndarray,
    image_vector: Optional[np.ndarray] = None,
    alpha: float = DEFAULT_ALPHA,
    fusion: Optional[LateFusion] = None,
) -> np.ndarray:
    """
    Fuse a text vector and optional image vector into a single query vector.

    Replaces the _pad_to_dim() stub used in M2's embed_pipeline.py.

    Parameters
    ----------
    text_vector : np.ndarray, shape (768,)
        L2-normalised CodeBERT embedding.
    image_vector : np.ndarray, shape (512,), optional
        L2-normalised CLIP image embedding. If None → text-only mode.
    alpha : float
        Text weight in [0.0, 1.0]. Default 0.7.
    fusion : LateFusion, optional
        Provide a pre-built LateFusion instance to reuse its projection.
        If None, the module-level singleton is used.

    Returns
    -------
    np.ndarray, shape (768,)
        L2-normalised fused vector.
    """
    f = fusion or _get_default_fusion()
    return f.fuse(text_vector, image_vector, alpha=alpha)


def alpha_sweep(
    text_vector: np.ndarray,
    image_vector: np.ndarray,
    alphas: Optional[List[float]] = None,
    fusion: Optional[LateFusion] = None,
) -> Dict[float, np.ndarray]:
    """
    Compute fused vectors for multiple alpha values.

    Used by the evaluator (M7) to tune the fusion hyperparameter.

    Parameters
    ----------
    text_vector : np.ndarray, shape (768,)
        L2-normalised text embedding.
    image_vector : np.ndarray, shape (512,)
        L2-normalised image embedding.
    alphas : list[float], optional
        Alpha values to sweep. Default: [0.0, 0.1, ..., 1.0].
    fusion : LateFusion, optional
        Pre-built LateFusion instance. If None, uses module singleton.

    Returns
    -------
    dict[float, np.ndarray]
        Maps each alpha → fused vector of shape (768,).

    Raises
    ------
    ValueError
        If any alpha is outside [0.0, 1.0].
    """
    if alphas is None:
        alphas = [round(a * 0.1, 1) for a in range(11)]  # [0.0, 0.1, ..., 1.0]

    invalid = [a for a in alphas if not 0.0 <= a <= 1.0]
    if invalid:
        raise ValueError(
            f"All alphas must be in [0.0, 1.0]. Invalid values: {invalid}"
        )

    f = fusion or _get_default_fusion()

    results: Dict[float, np.ndarray] = {}
    for alpha in alphas:
        results[alpha] = f.fuse(text_vector, image_vector, alpha=alpha)

    logger.info(
        "alpha_sweep: computed %d fused vectors for alphas %s",
        len(alphas), alphas,
    )
    return results


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.

    Utility used by the evaluator to compare fused vectors against
    ground-truth embeddings across alpha values.

    Parameters
    ----------
    a, b : np.ndarray, shape (D,)

    Returns
    -------
    float
        Cosine similarity in [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))