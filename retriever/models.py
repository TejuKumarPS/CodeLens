"""
retriever.models
================
Defines RetrievalResult — the canonical output type for ALL retrieval
methods (neural, BM25, reranker). Every module downstream of the
retriever (reranker, API, evaluator) works with this type exclusively.

Design rationale:
  - Typed dataclass prevents silent key errors from raw dicts
  - rank field is set by the retriever and updated by the reranker
  - retrieval_score vs rerank_score kept separate so evaluator can
    compare the two stages independently
  - to_dict() provides a JSON-serialisable form for the API layer
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class RetrievalResult:
    """
    A single retrieved code function result.

    Attributes
    ----------
    id : str
        CodeRecord.id — unique identifier.
    func_name : str
        Name of the retrieved function.
    repository : str
        GitHub repository the function comes from.
    url : str
        Direct GitHub link to the function definition.
    language : str
        Programming language (e.g. "python").
    docstring_preview : str
        First 200 chars of the function's docstring.
    code_preview : str
        First 300 chars of the function source code.
    document : str
        Full code snippet stored at index time (up to 500 chars).
    retrieval_score : float
        Cosine similarity score from ANN search (0–1), or BM25 score.
    rank : int
        1-based rank position in the result list.
    rerank_score : float or None
        Score assigned by cross-encoder reranker (set in M5).
        None until reranker has processed this result.
    retrieval_method : str
        "neural" | "bm25" — which retriever produced this result.
    """

    id: str
    func_name: str
    repository: str
    url: str
    language: str
    docstring_preview: str
    code_preview: str
    document: str
    retrieval_score: float
    rank: int
    retrieval_method: str = "neural"
    rerank_score: Optional[float] = field(default=None)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict for the API layer."""
        return asdict(self)

    @classmethod
    def from_chroma_hit(cls, hit: dict, rank: int,
                        method: str = "neural") -> "RetrievalResult":
        """
        Construct a RetrievalResult from a raw ChromaDB search hit dict.

        Parameters
        ----------
        hit : dict
            A dict returned by CodeLensIndex.search(), with keys:
            id, score, func_name, repository, url, language,
            docstring_preview, code_preview, document.
        rank : int
            1-based rank position.
        method : str
            "neural" or "bm25".
        """
        return cls(
            id=hit["id"],
            func_name=hit.get("func_name", ""),
            repository=hit.get("repository", ""),
            url=hit.get("url", ""),
            language=hit.get("language", ""),
            docstring_preview=hit.get("docstring_preview", ""),
            code_preview=hit.get("code_preview", ""),
            document=hit.get("document", ""),
            retrieval_score=float(hit.get("score", 0.0)),
            rank=rank,
            retrieval_method=method,
        )

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(rank={self.rank}, score={self.retrieval_score:.4f}, "
            f"func={self.func_name!r}, repo={self.repository!r})"
        )