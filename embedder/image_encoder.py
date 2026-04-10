"""
embedder.image_encoder
======================
Wraps openai/clip-vit-base-patch32 to produce 512-dimensional embeddings
for screenshot images.

CLIP (Contrastive Language-Image Pretraining) was trained to align images
and text in a shared vector space. For CodeLens, we use ONLY its vision
encoder — the image tower — to convert crash screenshots into vectors
that can be fused with CodeBERT text vectors.

Pooling strategy: CLIP's built-in pooled_output (CLS + projection head).
This is the standard representation for downstream similarity tasks.

Usage
-----
    from embedder.image_encoder import CLIPImageEncoder
    from PIL import Image

    enc = CLIPImageEncoder()
    img = Image.open("screenshot.png").convert("RGB")
    vec = enc.encode(img)           # np.ndarray (512,)
    vecs = enc.encode_batch([img, img2])  # np.ndarray (2, 512)
"""

import logging
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512


class CLIPImageEncoder:
    """
    Encodes PIL images using the vision tower of openai/clip-vit-base-patch32.

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
            import torch
            from transformers import CLIPProcessor, CLIPModel
        except ImportError as e:
            raise ImportError("pip install transformers torch") from e

        import torch
        self._torch = torch
        self.device = device or self._auto_device()
        logger.info("CLIPImageEncoder: loading %s on %s", MODEL_NAME, self.device)

        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        from transformers import CLIPProcessor, CLIPModel
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME, **kwargs)
        self.model = CLIPModel.from_pretrained(MODEL_NAME, **kwargs)
        self.model.eval()
        self.model.to(self.device)

        logger.info("CLIPImageEncoder ready. Embedding dim: %d", EMBEDDING_DIM)

    # ── Public API ────────────────────────────────────────────────────────────

    def encode(self, image) -> np.ndarray:
        """
        Encode a single PIL Image to a 512-dim vector.

        Parameters
        ----------
        image : PIL.Image.Image
            Screenshot or any RGB image.

        Returns
        -------
        np.ndarray, shape (512,)
            L2-normalised embedding vector.
        """
        return self.encode_batch([image], batch_size=1)[0]

    def encode_batch(
        self,
        images: list,
        batch_size: int = 8,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of PIL Images in batches.

        Parameters
        ----------
        images : list[PIL.Image.Image]
        batch_size : int
            Images per forward pass. CLIP is heavier than BERT; default 8.
        show_progress : bool
            Show tqdm progress bar.

        Returns
        -------
        np.ndarray, shape (N, 512)
            Row-wise L2-normalised embedding matrix.
        """
        if not images:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        # Ensure all images are RGB
        rgb_images = [img.convert("RGB") for img in images]

        all_embeddings: List[np.ndarray] = []
        iterator = range(0, len(rgb_images), batch_size)

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="CLIP encoding", unit="batch")

        with self._torch.no_grad():
            for start in iterator:
                batch = rgb_images[start : start + batch_size]
                embeddings = self._forward(batch)
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings).astype(np.float32)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _forward(self, images: list) -> np.ndarray:
        """Run one preprocess → forward → pool → normalise pass."""
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Use vision model only — we don't need the text tower
        vision_outputs = self.model.vision_model(**inputs)

        # pooler_output is the CLS token passed through a projection layer
        # shape: (B, 512)
        pooled = vision_outputs.pooler_output

        # Apply the visual projection head (maps to shared CLIP embedding space)
        projected = self.model.visual_projection(pooled)

        embeddings = projected.cpu().numpy()

        # L2 normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms

    def _auto_device(self) -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    def __repr__(self) -> str:
        return f"CLIPImageEncoder(model={MODEL_NAME!r}, device={self.device!r})"