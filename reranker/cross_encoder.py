"""
reranker.cross_encoder
======================
Wraps cross-encoder/ms-marco-MiniLM-L-6-v2 to re-score (query, code) pairs.

Architecture contrast vs bi-encoder (M2/M4):
  Bi-encoder:   query → vec_q,  code → vec_d,  score = cosine(vec_q, vec_d)
                Vectors computed INDEPENDENTLY → fast (pre-computed docs)

  Cross-encoder: [CLS] query [SEP] code [SEP] → single score
                Query and document fed TOGETHER → full attention interaction
                Cannot pre-compute → slow, but MUCH more accurate

Usage in CodeLens pipeline:
  1. Bi-encoder retrieves top-20 candidates  (M4, fast)
  2. Cross-encoder re-scores top-20          (M5, slow but accurate)
  3. Return top-5 by rerank score            (M5 output)

Scores are raw logits from the classification head — NOT probabilities.
Higher = more relevant. We normalise to [0, 1] via sigmoid for display.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~22M parameters (tiny, fast on CPU)
  - Trained on MS MARCO passage retrieval
  - Input max: 512 tokens (query + code concatenated)

Usage
-----
    from reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    scores = reranker.score_pairs([("bug report text", "def foo(): ..."), ...])
"""

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_LENGTH = 512


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid: maps raw logit → [0, 1]."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class CrossEncoderReranker:
    """
    Re-scores (query, document) pairs using a cross-encoder model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: ms-marco-MiniLM-L-6-v2.
    device : str, optional
        "cuda" | "cpu" | "mps". Auto-detected if None.
    cache_dir : str, optional
        HuggingFace model cache directory.
    batch_size : int
        Pairs per forward pass. Default 16.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        batch_size: int = 16,
    ) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as e:
            raise ImportError("pip install transformers torch") from e

        import torch
        self._torch = torch
        self._model_name = model_name
        self._batch_size = batch_size
        self.device = device or self._auto_device()

        logger.info(
            "CrossEncoderReranker: loading %s on %s", model_name, self.device
        )

        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **kwargs
        )
        self._model.eval()
        self._model.to(self.device)

        logger.info("CrossEncoderReranker ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def score_pairs(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[float]:
        """
        Score a list of (query, document) pairs.

        Parameters
        ----------
        pairs : list[tuple[str, str]]
            Each tuple is (query_text, code_snippet).

        Returns
        -------
        list[float]
            Sigmoid-normalised relevance scores in [0, 1].
            Higher = more relevant.
            Same length and order as input pairs.

        Raises
        ------
        ValueError
            If pairs is empty.
        """
        if not pairs:
            raise ValueError("pairs must be a non-empty list.")

        all_scores: List[float] = []

        with self._torch.no_grad():
            for start in range(0, len(pairs), self._batch_size):
                batch = pairs[start : start + self._batch_size]
                batch_scores = self._score_batch(batch)
                all_scores.extend(batch_scores)

        return all_scores

    def score_single(self, query: str, document: str) -> float:
        """
        Score a single (query, document) pair.

        Parameters
        ----------
        query : str
            Bug report text.
        document : str
            Code snippet to score against the query.

        Returns
        -------
        float
            Sigmoid-normalised relevance score in [0, 1].
        """
        return self.score_pairs([(query, document)])[0]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _score_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Tokenise and run one forward pass for a batch of pairs."""
        queries = [p[0] for p in pairs]
        docs    = [p[1] for p in pairs]

        encoded = self._tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        outputs = self._model(**encoded)

        # outputs.logits shape: (B, num_labels)
        # For ms-marco model: num_labels=1 (single relevance score)
        logits = outputs.logits.squeeze(-1).cpu().numpy()

        # Handle both scalar and array outputs
        if logits.ndim == 0:
            logits = np.array([float(logits)])

        return [_sigmoid(float(l)) for l in logits]

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
    def model_name(self) -> str:
        return self._model_name

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __repr__(self) -> str:
        return (
            f"CrossEncoderReranker(model={self._model_name!r}, "
            f"device={self.device!r}, batch_size={self._batch_size})"
        )