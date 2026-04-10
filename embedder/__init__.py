"""
embedder — CodeLens Module 2
=============================
Responsible for:
  - Encoding code functions + bug report text → 768-dim vectors (CodeBERT)
  - Encoding screenshot images → 512-dim vectors (CLIP)
  - Batch processing with progress bars
  - Device-aware inference (CPU / CUDA)

Public API:
  CodeBERTEncoder          — encodes text/code strings
  CLIPImageEncoder         — encodes PIL images
  embed_records(...)       — batch-embed a list of CodeRecord objects
  embed_query(...)         — embed a single bug report (text + optional image)
"""

from .code_encoder import CodeBERTEncoder
from .image_encoder import CLIPImageEncoder
from .embed_pipeline import embed_records, embed_query, EmbeddedQuery

__all__ = [
    "CodeBERTEncoder",
    "CLIPImageEncoder",
    "embed_records",
    "embed_query",
    "EmbeddedQuery",
]
