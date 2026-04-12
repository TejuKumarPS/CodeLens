"""
fusion.projection
=================
LinearProjection maps CLIP image vectors (512-dim) into CodeBERT's
embedding space (768-dim) via a learned linear transformation.

Why we need this:
  In M2 (embed_pipeline.py) we used zero-padding to align dimensions:
    [v_0, v_1, ..., v_511, 0, 0, ..., 0]   ← 768-dim, 256 zeros appended

  Zero-padding wastes 256 dimensions and distorts the geometry of the
  image vector after L2-normalisation.

  A linear projection W ∈ R^{768×512} maps the image vector to the full
  768-dimensional space. When trained on paired (image, code) examples,
  W learns to align image features with code semantics.

  For CodeLens (academic project without paired training data) we initialise
  W with a random orthogonal matrix — this is far better than zero-padding
  because orthogonal projections preserve norms and distances exactly.

Initialisation options:
  "orthogonal"  — random orthogonal matrix (default, norm-preserving)
  "random"      — standard normal, then L2-normalised rows
  "identity_pad"— zero-padded identity (same as M2 stub, for ablation)

Save/load:
  Weights are saved as .npy files so the same projection is reused across
  sessions. Without saving, each restart uses a different random matrix,
  making results non-reproducible.

Usage
-----
    from fusion.projection import LinearProjection
    import numpy as np

    proj = LinearProjection(input_dim=512, output_dim=768)
    img_vec = np.random.randn(512).astype(np.float32)
    img_vec /= np.linalg.norm(img_vec)
    proj_vec = proj.project(img_vec)   # shape: (768,), unit norm
    print(proj_vec.shape)              # (768,)
"""

import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

InitStrategy = Literal["orthogonal", "random", "identity_pad"]

DEFAULT_INPUT_DIM  = 512   # CLIP ViT-B/32 output
DEFAULT_OUTPUT_DIM = 768   # CodeBERT hidden size


class LinearProjection:
    """
    Projects vectors from input_dim → output_dim via a fixed linear map.

    The projection matrix W is initialised once and optionally saved to
    disk for reproducibility.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vectors (CLIP: 512).
    output_dim : int
        Dimensionality of the output vectors (CodeBERT: 768).
    init : str
        Initialisation strategy: "orthogonal" | "random" | "identity_pad".
    seed : int
        Random seed for reproducibility.
    weights_path : str, optional
        Path to a .npy file. If the file exists, weights are loaded from it.
        If it does not exist, weights are initialised and saved there.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        init: InitStrategy = "orthogonal",
        seed: int = 42,
        weights_path: Optional[str] = None,
    ) -> None:
        self._input_dim  = input_dim
        self._output_dim = output_dim
        self._init       = init

        if weights_path and Path(weights_path).exists():
            self._W = np.load(weights_path).astype(np.float32)
            logger.info(
                "LinearProjection: loaded weights from %s (shape %s)",
                weights_path, self._W.shape,
            )
            if self._W.shape != (output_dim, input_dim):
                raise ValueError(
                    f"Loaded weight shape {self._W.shape} does not match "
                    f"expected ({output_dim}, {input_dim})."
                )
        else:
            self._W = self._init_weights(seed)
            logger.info(
                "LinearProjection: initialised %s matrix (%d→%d)",
                init, input_dim, output_dim,
            )
            if weights_path:
                self.save(weights_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def project(self, vector: np.ndarray) -> np.ndarray:
        """
        Project a single input vector to output_dim.

        Parameters
        ----------
        vector : np.ndarray, shape (input_dim,)
            L2-normalised input vector (e.g. CLIP image embedding).

        Returns
        -------
        np.ndarray, shape (output_dim,)
            L2-normalised projected vector.

        Raises
        ------
        ValueError
            If vector has wrong shape.
        """
        if vector.ndim != 1 or vector.shape[0] != self._input_dim:
            raise ValueError(
                f"Expected 1-D vector of length {self._input_dim}, "
                f"got shape {vector.shape}."
            )

        projected = self._W @ vector.astype(np.float32)   # (output_dim,)
        return self._normalise(projected)

    def project_batch(self, vectors: np.ndarray) -> np.ndarray:
        """
        Project a batch of input vectors.

        Parameters
        ----------
        vectors : np.ndarray, shape (N, input_dim)

        Returns
        -------
        np.ndarray, shape (N, output_dim)
            Each row is L2-normalised.
        """
        if vectors.ndim != 2 or vectors.shape[1] != self._input_dim:
            raise ValueError(
                f"Expected 2-D array of shape (N, {self._input_dim}), "
                f"got {vectors.shape}."
            )
        projected = (self._W @ vectors.astype(np.float32).T).T  # (N, output_dim)
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return projected / norms

    def save(self, path: str) -> None:
        """Save projection weights to a .npy file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), self._W)
        logger.info("LinearProjection: weights saved → %s", path)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init_weights(self, seed: int) -> np.ndarray:
        """Initialise W ∈ R^{output_dim × input_dim}."""
        rng = np.random.default_rng(seed)

        if self._init == "orthogonal":
            # Build a random orthogonal matrix via QR decomposition
            # We need output_dim × input_dim; generate a square matrix
            # of size max_dim × max_dim, then slice.
            max_dim = max(self._input_dim, self._output_dim)
            A = rng.standard_normal((max_dim, max_dim)).astype(np.float32)
            Q, _ = np.linalg.qr(A)           # Q is orthogonal (max_dim, max_dim)
            W = Q[: self._output_dim, : self._input_dim]
            return W

        elif self._init == "random":
            W = rng.standard_normal(
                (self._output_dim, self._input_dim)
            ).astype(np.float32)
            # L2-normalise each row
            norms = np.linalg.norm(W, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return W / norms

        elif self._init == "identity_pad":
            # Replicates the M2 zero-padding behaviour for ablation comparison
            W = np.zeros(
                (self._output_dim, self._input_dim), dtype=np.float32
            )
            min_dim = min(self._input_dim, self._output_dim)
            W[:min_dim, :min_dim] = np.eye(min_dim, dtype=np.float32)
            return W

        else:
            raise ValueError(
                f"Unknown init strategy: {self._init!r}. "
                "Choose 'orthogonal', 'random', or 'identity_pad'."
            )

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def weights(self) -> np.ndarray:
        """Return a read-only view of the projection matrix."""
        w = self._W.view()
        w.flags.writeable = False
        return w

    def __repr__(self) -> str:
        return (
            f"LinearProjection(input={self._input_dim}, "
            f"output={self._output_dim}, init={self._init!r})"
        )