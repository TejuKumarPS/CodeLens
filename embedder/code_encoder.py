"""
embedder.code_encoder
=====================
Encodes source code and natural language queries into 768-dimensional
vectors using sentence-transformers.

Model: flax-sentence-embeddings/st-codesearch-distilroberta-base
  Trained specifically for code search on CodeSearchNet with a
  contrastive objective — produces proper semantic similarity scores.
  This is the model used in the original CodeSearchNet evaluation paper.

Why not raw microsoft/codebert-base?
  CodeBERT is a masked language model, NOT trained for similarity.
  Both CLS and mean pooling collapse (cosine ≈ 0.97+ for all pairs),
  making retrieval scores uniformly high and results semantically random.
  sentence-transformers wraps models with the pooling+normalisation that
  was used during contrastive training, which is critical for retrieval.

Usage
-----
    from embedder.code_encoder import CodeBERTEncoder

    enc = CodeBERTEncoder()
    vec = enc.encode("def checkout(cart): return cart.total")
    vecs = enc.encode_batch(["def foo()...", "def bar()..."], batch_size=32)
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME    = "flax-sentence-embeddings/st-codesearch-distilroberta-base"
EMBEDDING_DIM = 768


class CodeBERTEncoder:
    """
    Encodes source code or natural language text using a sentence-transformer
    model trained for code search.

    Despite the class name (kept for API compatibility), this uses
    st-codesearch-distilroberta-base — trained on CodeSearchNet with
    a contrastive objective for proper semantic code search.

    Parameters
    ----------
    device : str, optional
        "cuda" | "cpu" | "mps". Auto-detected if None.
    cache_dir : str, optional
        HuggingFace model cache directory.
    model_name : str, optional
        Override the default model.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        model_name: str = MODEL_NAME,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("pip install sentence-transformers") from e

        self._model_name = model_name
        self.device = device or self._auto_device()

        logger.info("CodeBERTEncoder: loading %s on %s", model_name, self.device)

        kwargs = {}
        if cache_dir:
            kwargs["cache_folder"] = cache_dir

        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, device=self.device, **kwargs)

        # Detect actual embedding dimension
        test_vec = self._model.encode("test", convert_to_numpy=True)
        self._embedding_dim = test_vec.shape[0]

        logger.info(
            "CodeBERTEncoder ready. model=%s, dim=%d, device=%s",
            model_name, self._embedding_dim, self.device,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single string to a normalised embedding vector.

        Parameters
        ----------
        text : str
            Source code snippet or natural language string.

        Returns
        -------
        np.ndarray, shape (embedding_dim,)
            L2-normalised embedding vector.
        """
        vec = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.astype(np.float32)

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of strings in batches.

        Parameters
        ----------
        texts : list[str]
        batch_size : int
        show_progress : bool

        Returns
        -------
        np.ndarray, shape (N, embedding_dim)
            Row-wise L2-normalised embedding matrix.
        """
        if not texts:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)

        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return vecs.astype(np.float32)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def __repr__(self) -> str:
        return (
            f"CodeBERTEncoder(model={self._model_name!r}, "
            f"device={self.device!r}, dim={self._embedding_dim})"
        )