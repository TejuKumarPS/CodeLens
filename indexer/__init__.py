"""
indexer — CodeLens Module 3
============================
Responsible for:
  - Initialising and managing a ChromaDB persistent collection
  - Bulk-indexing CodeRecord embeddings + metadata
  - Incremental upsert (safe to re-run without duplication)
  - Raw ANN vector search (top-K nearest neighbours)
  - Collection inspection utilities (count, peek, delete)

Public API:
  CodeLensIndex          — main class: wraps ChromaDB collection
  build_index(...)       — convenience: embed + index a full record list
"""

from .chroma_index import CodeLensIndex
from .build import build_index

__all__ = [
    "CodeLensIndex",
    "build_index",
]