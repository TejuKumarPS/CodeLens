"""
fusion — CodeLens Module 6
===========================
Responsible for:
  - Projecting CLIP image vectors (512-dim) → CodeBERT space (768-dim)
    via a learned linear projection (replaces M2 zero-padding stub)
  - Weighted late fusion: fused = alpha * text_vec + (1-alpha) * img_vec_proj
  - Alpha sweep utility: evaluate multiple alpha values to find optimal weight
  - Producing FusedQuery objects consumed by the retriever (M4)

Public API:
  LinearProjection         — projects 512-dim → 768-dim (learnable weights)
  LateFusion               — fuses text + image with tunable alpha
  fuse()                   — convenience: fuse text_vec + img_vec → fused_vec
  alpha_sweep()            — sweep alpha values, return per-alpha fused vectors
"""

from .projection import LinearProjection
from .late_fusion import LateFusion
from .fuse import fuse, alpha_sweep

__all__ = [
    "LinearProjection",
    "LateFusion",
    "fuse",
    "alpha_sweep",
]