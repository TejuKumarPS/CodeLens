

"""
retriever.retrieve
==================
Top-level convenience function consumed by the reranker (M5) and the
FastAPI backend (M8). Combines embedding + neural ANN search into a
single call.

    results = retrieve(text="app crashes at checkout", index=index,
                       text_encoder=encoder)

This is the function signature the reranker will call:
    candidates = retrieve(text, image, index, text_encoder, image_encoder)
    reranked   = reranker.rerank(text, candidates)

Usage
-----
    from retriever import retrieve
    from indexer import CodeLensIndex
    from embedder import CodeBERTEncoder

    index   = CodeLensIndex("chroma_db/")
    encoder = CodeBERTEncoder()
    results = retrieve("null pointer in payment module", index, encoder,
                       top_k=20)
    for r in results:
        print(r.rank, r.func_name, f"{r.retrieval_score:.4f}")
"""

import logging
from typing import List, Optional

from indexer.chroma_index import CodeLensIndex
from embedder.code_encoder import CodeBERTEncoder
from embedder.image_encoder import CLIPImageEncoder
from embedder.embed_pipeline import embed_query
from .neural import NeuralRetriever
from .models import RetrievalResult

logger = logging.getLogger(__name__)


def retrieve(
    text: str,
    index: CodeLensIndex,
    text_encoder: CodeBERTEncoder,
    image=None,
    image_encoder: Optional[CLIPImageEncoder] = None,
    alpha: float = 0.7,
    top_k: int = 20,
) -> List[RetrievalResult]:
    """
    End-to-end retrieval: encode query → fuse modalities → ANN search.

    Parameters
    ----------
    text : str
        Natural language bug report description.
    index : CodeLensIndex
        Populated ChromaDB index (built by indexer.build).
    text_encoder : CodeBERTEncoder
        Loaded CodeBERT encoder (load once, reuse).
    image : PIL.Image.Image, optional
        Screenshot of the crash/error. If None, text-only mode.
    image_encoder : CLIPImageEncoder, optional
        Loaded CLIP encoder. Required only when image is provided.
    alpha : float
        Fusion weight: 1.0 = text only, 0.0 = image only. Default 0.7.
    top_k : int
        Number of candidates to return. Default 20 (fed to reranker).

    Returns
    -------
    list[RetrievalResult]
        Top-K results sorted by descending cosine similarity.

    Raises
    ------
    ValueError
        If text is empty, or image provided without image_encoder.
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string.")

    # Step 1: Embed the query (text + optional image → fused vector)
    embedded = embed_query(
        text=text,
        text_encoder=text_encoder,
        image=image,
        image_encoder=image_encoder,
        alpha=alpha,
    )

    # Step 2: ANN search
    neural = NeuralRetriever(index=index, top_k=top_k)
    results = neural.search(embedded)

    logger.info(
        "retrieve(): text=%r | has_image=%s | top_k=%d → %d results",
        text[:60], embedded.has_image, top_k, len(results),
    )
    return results