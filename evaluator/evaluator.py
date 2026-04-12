
"""
evaluator.evaluator
===================
Evaluator orchestrates a full evaluation run over a query set:

  For each query:
    1. Embed query text → fused vector
    2. Retrieve top-K candidates from ChromaDB
    3. (Optional) Re-rank with cross-encoder
    4. Compare retrieved IDs vs ground-truth relevant IDs
    5. Record per-query RR, NDCG, Precision

  Aggregate across queries → Metrics dataclass

Ground-truth format used in CodeLens:
  CodeSearchNet provides (docstring, code) pairs. For evaluation we treat
  each docstring as a query and its paired code function as the single
  relevant document. Ground truth = {code_record_id}.

  This is the standard CodeSearchNet evaluation protocol used in the
  original paper (Husain et al., 2019).

Usage
-----
    from evaluator import Evaluator
    from indexer import CodeLensIndex
    from embedder import CodeBERTEncoder

    evaluator = Evaluator(
        index=CodeLensIndex("chroma_db/"),
        text_encoder=CodeBERTEncoder(),
        k=10,
    )
    metrics = evaluator.run(query_records)
    print(metrics)
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from data_loader.models import CodeRecord
from retriever.models import RetrievalResult
from .metrics import Metrics, aggregate_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Runs retrieval evaluation over a set of CodeRecord queries.

    Each CodeRecord acts as both query (via docstring) and ground truth
    (its own ID is the expected top result).

    Parameters
    ----------
    index : CodeLensIndex
        Populated ChromaDB index.
    text_encoder : CodeBERTEncoder
        Loaded CodeBERT encoder.
    k : int
        Evaluation cutoff for NDCG and Precision. Default 10.
    reranker : CrossEncoderReranker, optional
        If provided, re-ranks top-20 before computing metrics.
    top_k_retrieval : int
        Number of candidates to retrieve per query. Default 20.
    """

    def __init__(
        self,
        index,
        text_encoder,
        k: int = 10,
        reranker=None,
        top_k_retrieval: int = 20,
    ) -> None:
        self._index           = index
        self._text_encoder    = text_encoder
        self._k               = k
        self._reranker        = reranker
        self._top_k_retrieval = top_k_retrieval

    def run(
        self,
        query_records: List[CodeRecord],
        method: str = "neural",
        alpha: Optional[float] = None,
        show_progress: bool = True,
        limit: Optional[int] = None,
    ) -> Metrics:
        """
        Run evaluation over a list of CodeRecord queries.

        Parameters
        ----------
        query_records : list[CodeRecord]
            Records from the test split. Docstring = query text.
            Record ID = ground-truth relevant document.
        method : str
            Label for results ("neural", "bm25", "reranked").
        alpha : float, optional
            Fusion alpha value (logged in Metrics, does not affect retrieval
            unless a LateFusion instance was configured externally).
        show_progress : bool
            Log progress every 100 queries.
        limit : int, optional
            Evaluate only the first N queries (for quick sanity checks).

        Returns
        -------
        Metrics
            Aggregate MRR, NDCG@K, P@K.
        """
        from retriever.retrieve import retrieve
        from reranker.rerank import rerank as do_rerank

        records = query_records[:limit] if limit else query_records
        n = len(records)
        logger.info(
            "Evaluator.run(): %d queries | k=%d | method=%s",
            n, self._k, method,
        )

        start = time.time()
        retrieved_lists = []
        relevant_sets   = []

        for i, record in enumerate(records):
            # Query = docstring text
            query_text = record.docstring
            # Ground truth = this record's own ID
            relevant = {record.id}

            # Retrieve
            candidates: List[RetrievalResult] = retrieve(
                text=query_text,
                index=self._index,
                text_encoder=self._text_encoder,
                top_k=self._top_k_retrieval,
            )

            # Optionally rerank
            if self._reranker and candidates:
                candidates = do_rerank(
                    query_text=query_text,
                    candidates=candidates,
                    reranker=self._reranker,
                    top_n=self._k,
                )

            retrieved_ids = [c.id for c in candidates]
            retrieved_lists.append(retrieved_ids)
            relevant_sets.append(relevant)

            if show_progress and (i + 1) % 100 == 0:
                elapsed = time.time() - start
                logger.info(
                    "  Progress: %d / %d queries (%.1fs elapsed)",
                    i + 1, n, elapsed,
                )

        metrics = aggregate_metrics(
            retrieved_lists=retrieved_lists,
            relevant_sets=relevant_sets,
            k=self._k,
            method=method,
            alpha=alpha,
        )

        elapsed = time.time() - start
        logger.info(
            "Evaluation complete: %s | %.1fs", metrics, elapsed
        )
        return metrics

    def run_bm25(
        self,
        query_records: List[CodeRecord],
        corpus_records: List[CodeRecord],
        limit: Optional[int] = None,
    ) -> Metrics:
        """
        Run BM25 baseline evaluation.

        Parameters
        ----------
        query_records : list[CodeRecord]
            Test queries (docstring = query text).
        corpus_records : list[CodeRecord]
            Records to build the BM25 index over.
        limit : int, optional
            Evaluate only the first N queries.

        Returns
        -------
        Metrics
            BM25 aggregate metrics.
        """
        from retriever.bm25 import BM25Retriever

        records = query_records[:limit] if limit else query_records
        bm25    = BM25Retriever(corpus_records)

        retrieved_lists = []
        relevant_sets   = []

        for record in records:
            results = bm25.search(record.docstring, top_k=self._k)
            retrieved_lists.append([r.id for r in results])
            relevant_sets.append({record.id})

        return aggregate_metrics(
            retrieved_lists=retrieved_lists,
            relevant_sets=relevant_sets,
            k=self._k,
            method="bm25",
        )

    def save_report(
        self,
        metrics: Metrics,
        output_path: str,
    ) -> None:
        """
        Save a Metrics object to a JSON file.

        Parameters
        ----------
        metrics : Metrics
        output_path : str
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        logger.info("Evaluation report saved → %s", path)