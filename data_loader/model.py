"""
data_loader.models
==================
Defines the canonical CodeRecord dataclass used across all CodeLens modules.
Every downstream module (embedder, indexer, retriever, evaluator) consumes
this exact schema — never raw HuggingFace dataset dicts.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


@dataclass
class CodeRecord:
    """
    A single processable code function extracted from CodeSearchNet.

    Attributes
    ----------
    id : str
        Unique identifier. Format: "<repo_name>::<func_name>::<idx>"
    func_name : str
        Name of the function (e.g. "processCheckout")
    func_code : str
        Full source code of the function as a string
    docstring : str
        Natural language documentation / docstring
    language : str
        Programming language (e.g. "python")
    repository : str
        GitHub repository name (e.g. "django/django")
    url : str
        Direct GitHub URL to the function definition
    tokens : List[str]
        Tokenized function code — used for BM25 baseline
    partition : str
        Dataset split: "train" | "valid" | "test"
    """

    id: str
    func_name: str
    func_code: str
    docstring: str
    language: str
    repository: str
    url: str
    tokens: List[str]
    partition: str
    # Optional: filled by embedder module in Milestone 2
    embedding: Optional[List[float]] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Serialize to plain dict (embedding excluded for storage efficiency)."""
        d = asdict(self)
        d.pop("embedding", None)
        return d

    def to_chroma_metadata(self) -> dict:
        """
        Return ChromaDB-compatible metadata dict.
        ChromaDB metadata values must be str/int/float/bool — no lists.
        """
        return {
            "func_name": self.func_name,
            "language": self.language,
            "repository": self.repository,
            "url": self.url,
            "partition": self.partition,
            "docstring_preview": self.docstring[:200],
            "code_preview": self.func_code[:300],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CodeRecord":
        """Reconstruct from serialized dict."""
        return cls(
            id=d["id"],
            func_name=d["func_name"],
            func_code=d["func_code"],
            docstring=d["docstring"],
            language=d["language"],
            repository=d["repository"],
            url=d["url"],
            tokens=d["tokens"],
            partition=d["partition"],
            embedding=d.get("embedding"),
        )

    def __repr__(self) -> str:
        return (
            f"CodeRecord(id={self.id!r}, func={self.func_name!r}, "
            f"lang={self.language!r}, repo={self.repository!r}, "
            f"doc_len={len(self.docstring)}, code_len={len(self.func_code)})"
        )
    