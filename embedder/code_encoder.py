
"""
embedder.code_encoder
=====================
Wraps microsoft/codebert-base to produce 768-dimensional embeddings
for both source code and natural language bug report text.

CodeBERT is a bimodal model pre-trained on (NL, code) pairs — it
naturally produces vectors in a shared NL-code semantic space, which
is exactly what we need for cross-modal retrieval.

Pooling strategy: CLS token (index 0 of last hidden state).
This is the standard approach for sentence/document-level embeddings
with BERT-family models.

Usage
-----
    from embedder.code_encoder import CodeBERTEncoder

    enc = CodeBERTEncoder()                    # loads model once
    vec = enc.encode("def foo(): return 42")   # single string → np.ndarray (768,)
    vecs = enc.encode_batch(["def foo()...", "def bar()..."], batch_size=32)
"""

import logging
from typing import List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_NAME = "microsoft/codebert-base"
EMBEDDING_DIM = 768
MAX_SEQ_LEN = 512       # CodeBERT hard limit


class CodeBERTEncoder:
    """
    Encodes source code or natural language text using microsoft/codebert-base.

    Parameters
    ----------
    device : str, optional
        "cuda" | "cpu" | "mps". Auto-detected if None.
    cache_dir : str, optional
        HuggingFace model cache directory.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError as e:
            raise ImportError("pip install transformers") from e

        self.device = device or self._auto_device()
        logger.info("CodeBERTEncoder: loading %s on %s", MODEL_NAME, self.device)

        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **kwargs)
        self.model = AutoModel.from_pretrained(MODEL_NAME, **kwargs)
        self.model.eval()
        self.model.to(self.device)

        logger.info("CodeBERTEncoder ready. Embedding dim: %d", EMBEDDING_DIM)

    # ── Public API ────────────────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single string (code or NL text) to a 768-dim vector.

        Parameters
        ----------
        text : str
            Source code snippet or natural language string.

        Returns
        -------
        np.ndarray, shape (768,)
            L2-normalised embedding vector.
        """
        return self.encode_batch([text], batch_size=1)[0]

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of strings in batches.

        Parameters
        ----------
        texts : list[str]
        batch_size : int
            Number of strings per forward pass. Reduce if OOM on GPU.
        show_progress : bool
            Show tqdm progress bar.

        Returns
        -------
        np.ndarray, shape (N, 768)
            Row-wise L2-normalised embedding matrix.
        """
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        all_embeddings: List[np.ndarray] = []
        iterator = range(0, len(texts), batch_size)

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="CodeBERT encoding", unit="batch")

        with torch.no_grad():
            for start in iterator:
                batch = texts[start : start + batch_size]
                embeddings = self._forward(batch)
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings).astype(np.float32)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _forward(self, texts: List[str]) -> np.ndarray:
        """Run one tokenise → forward → pool → normalise pass."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        # Move tensors to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        outputs = self.model(**encoded)

        # CLS token pooling: first token of last hidden state
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (B, 768)

        # Move to CPU and convert to numpy
        embeddings = cls_embeddings.cpu().numpy()

        # L2 normalise so cosine similarity == dot product (ChromaDB default)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)   # avoid div-by-zero
        return embeddings / norms

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    def __repr__(self) -> str:
        return f"CodeBERTEncoder(model={MODEL_NAME!r}, device={self.device!r})"

