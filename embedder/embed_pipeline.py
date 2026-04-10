"""
embedder.embed_pipeline
========================
High-level pipeline functions consumed by:
  - Milestone 3 (indexer): embed_records() to bulk-encode CodeSearchNet
  - Milestone 4 (retriever): embed_query() to encode each incoming bug report

Two core functions:

  embed_records(records, encoder, batch_size, show_progress)
      Takes a list of CodeRecord (from data_loader), encodes each
      func_code using CodeBERT, stores the 768-dim vector back into
      record.embedding, and returns the updated list.

  embed_query(text, image, text_encoder, image_encoder, alpha)
      Encodes a bug report text + optional screenshot, fuses them via
      weighted addition (late fusion), and returns an EmbeddedQuery.

Late fusion formula (from requirements doc):
    fused = alpha * text_vector + (1 - alpha) * image_vector

    When image is None:  fused = text_vector  (alpha forced to 1.0)
    Default alpha = 0.7  (text-dominant, tunable hyperparameter)

Usage
-----
    from data_loader import load_processed
    from embedder import CodeBERTEncoder, CLIPImageEncoder
    from embedder import embed_records, embed_query

    encoder = CodeBERTEncoder()
    records = load_processed("data/processed/python_train.parquet")
    records = embed_records(records, encoder, batch_size=32, show_progress=True)

    query = embed_query("app crashes at checkout", image=None, text_encoder=encoder)
    print(query.fused_vector.shape)   # (768,)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from data_loader.models import CodeRecord
from .code_encoder import CodeBERTEncoder, EMBEDDING_DIM as CODE_DIM
from .image_encoder import CLIPImageEncoder, EMBEDDING_DIM as IMG_DIM

logger = logging.getLogger(__name__)

# Default late-fusion weight: 0.7 text, 0.3 image
DEFAULT_ALPHA = 0.7


@dataclass
class EmbeddedQuery:
    """
    Holds all vector representations of a single bug report query.

    Attributes
    ----------
    text : str
        Original bug report text.
    text_vector : np.ndarray, shape (768,)
        CodeBERT embedding of the text description.
    image_vector : np.ndarray or None, shape (512,)
        CLIP embedding of the screenshot. None if no image provided.
    fused_vector : np.ndarray, shape (768,)
        Late-fused query vector used for ANN search.
        If image_vector is None, this equals text_vector.
        Otherwise: alpha * text_vector + (1-alpha) * image_vector_projected
    alpha : float
        Fusion weight used (0.0–1.0, text dominance).
    has_image : bool
        Whether an image was part of this query.
    """
    text: str
    text_vector: np.ndarray
    fused_vector: np.ndarray
    alpha: float
    has_image: bool
    image_vector: Optional[np.ndarray] = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"EmbeddedQuery(text={self.text[:50]!r}, "
            f"has_image={self.has_image}, alpha={self.alpha}, "
            f"fused_shape={self.fused_vector.shape})"
        )


def embed_records(
    records: List[CodeRecord],
    encoder: CodeBERTEncoder,
    batch_size: int = 32,
    show_progress: bool = True,
) -> List[CodeRecord]:
    """
    Batch-encode a list of CodeRecord using CodeBERT.

    Each record's func_code is encoded and stored in record.embedding.
    Records are mutated in-place AND returned for chaining.

    Parameters
    ----------
    records : list[CodeRecord]
        Output of data_loader.load_processed().
    encoder : CodeBERTEncoder
        Initialised CodeBERT encoder (loaded once, reused).
    batch_size : int
        Codes per forward pass through CodeBERT.
    show_progress : bool
        Show tqdm progress bar.

    Returns
    -------
    list[CodeRecord]
        Same list, each record now has record.embedding set (list[float], len 768).
    """
    if not records:
        logger.warning("embed_records called with empty list.")
        return records

    logger.info("Embedding %d records with CodeBERT (batch_size=%d)...",
                len(records), batch_size)

    codes = [r.func_code for r in records]
    embeddings = encoder.encode_batch(codes, batch_size=batch_size,
                                      show_progress=show_progress)
    # embeddings shape: (N, 768)

    for record, vec in zip(records, embeddings):
        record.embedding = vec.tolist()   # store as list for JSON/Parquet compat

    logger.info("Embedding complete. %d records now have embeddings.", len(records))
    return records


def embed_query(
    text: str,
    text_encoder: CodeBERTEncoder,
    image=None,
    image_encoder: Optional[CLIPImageEncoder] = None,
    alpha: float = DEFAULT_ALPHA,
) -> EmbeddedQuery:
    """
    Encode a bug report (text + optional screenshot) into a fused query vector.

    Parameters
    ----------
    text : str
        Natural language bug report description.
    text_encoder : CodeBERTEncoder
        Loaded CodeBERT encoder.
    image : PIL.Image.Image, optional
        Screenshot of the crash/error. If None, text-only mode.
    image_encoder : CLIPImageEncoder, optional
        Loaded CLIP encoder. Required if image is provided.
    alpha : float
        Fusion weight for text (0.0 = image only, 1.0 = text only).
        Default 0.7 (text-dominant per requirements doc).

    Returns
    -------
    EmbeddedQuery
        Contains text_vector, image_vector (if any), and fused_vector.

    Raises
    ------
    ValueError
        If image is provided but image_encoder is None.
    """
    if not text or not text.strip():
        raise ValueError("Query text must be a non-empty string.")

    if image is not None and image_encoder is None:
        raise ValueError(
            "image_encoder must be provided when an image is supplied."
        )

    # ── Encode text ───────────────────────────────────────────────────────────
    text_vector = text_encoder.encode(text.strip())  # shape: (768,)

    # ── Encode image (if provided) ────────────────────────────────────────────
    image_vector = None
    fused_vector = text_vector.copy()

    if image is not None:
        raw_img_vector = image_encoder.encode(image)  # shape: (512,)
        image_vector = raw_img_vector

        # Dimension alignment: project 512-dim CLIP vector to 768-dim CodeBERT space
        # via zero-padding. This is the simplest alignment strategy; a learned
        # projection matrix (Milestone 6) will replace this.
        projected_img = _pad_to_dim(raw_img_vector, CODE_DIM)  # shape: (768,)

        # Late fusion: weighted addition
        fused_vector = alpha * text_vector + (1.0 - alpha) * projected_img

        # Re-normalise after fusion
        norm = np.linalg.norm(fused_vector)
        if norm > 0:
            fused_vector = fused_vector / norm

    return EmbeddedQuery(
        text=text.strip(),
        text_vector=text_vector,
        image_vector=image_vector,
        fused_vector=fused_vector,
        alpha=alpha if image is not None else 1.0,
        has_image=image is not None,
    )


def _pad_to_dim(vector: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Pad or truncate a 1-D vector to target_dim.

    Used to align CLIP's 512-dim output with CodeBERT's 768-dim space
    before weighted addition. Zero-padding preserves the existing signal
    while filling the remaining dimensions with neutral values.

    Parameters
    ----------
    vector : np.ndarray, shape (D,)
    target_dim : int

    Returns
    -------
    np.ndarray, shape (target_dim,)
        L2-normalised after padding.
    """
    current_dim = vector.shape[0]
    if current_dim == target_dim:
        return vector.copy()
    elif current_dim > target_dim:
        result = vector[:target_dim].copy()
    else:
        result = np.zeros(target_dim, dtype=np.float32)
        result[:current_dim] = vector

    # Re-normalise after padding
    norm = np.linalg.norm(result)
    if norm > 0:
        result = result / norm
    return result