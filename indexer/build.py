"""
indexer.build
=============
Convenience pipeline that combines Milestone 2 (embedder) and
Milestone 3 (indexer) into a single CLI-runnable script:

  1. Load processed CodeRecords from Parquet (data_loader.load_processed)
  2. Embed func_code via CodeBERT (embedder.embed_records)
  3. Upsert embedded records into ChromaDB (CodeLensIndex.upsert)

This is the script you run ONCE to build the searchable index before
the retriever and API are started.

Estimated time on CPU:
  - 4,900 records × ~0.05s/record = ~4 min for embedding
  - Upsert: ~10s

CLI Usage
---------
    python -m indexer.build \\
        --parquet data/processed/python_train.parquet \\
        --db      chroma_db/ \\
        --batch   32

    # Rebuild from scratch (deletes existing collection first):
    python -m indexer.build --parquet data/processed/python_train.parquet --rebuild
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def build_index(
    parquet_path: str,
    persist_dir: str = "chroma_db",
    collection_name: str = "codelens_python",
    batch_size: int = 32,
    rebuild: bool = False,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> "CodeLensIndex":
    """
    Full pipeline: Parquet → embed → index.

    Parameters
    ----------
    parquet_path : str
        Path to the Parquet file from data_loader.save_processed().
    persist_dir : str
        ChromaDB storage directory.
    collection_name : str
        Name of the ChromaDB collection.
    batch_size : int
        CodeBERT batch size for embedding. Reduce if OOM.
    rebuild : bool
        If True, delete the existing collection before indexing.
    device : str, optional
        Force CPU/CUDA/MPS. Auto-detected if None.
    cache_dir : str, optional
        HuggingFace cache directory for model weights.

    Returns
    -------
    CodeLensIndex
        The populated index, ready for search.
    """
    from data_loader.pipeline import load_processed
    from embedder.code_encoder import CodeBERTEncoder
    from embedder.embed_pipeline import embed_records
    from indexer.chroma_index import CodeLensIndex

    total_start = time.time()

    # ── Step 1: Load records ──────────────────────────────────────────────────
    logger.info("Step 1/3 — Loading records from %s", parquet_path)
    records = load_processed(parquet_path)
    logger.info("Loaded %d records.", len(records))

    # ── Step 2: Embed with CodeBERT ───────────────────────────────────────────
    logger.info("Step 2/3 — Embedding with CodeBERT (batch_size=%d)...", batch_size)
    encoder = CodeBERTEncoder(device=device, cache_dir=cache_dir)
    records = embed_records(records, encoder, batch_size=batch_size, show_progress=True)
    logger.info("Embedding done.")

    # ── Step 3: Index into ChromaDB ───────────────────────────────────────────
    logger.info("Step 3/3 — Indexing into ChromaDB at %s ...", persist_dir)
    index = CodeLensIndex(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )

    if rebuild:
        logger.warning("--rebuild flag set. Deleting existing collection.")
        index.delete_collection()

    upserted = index.upsert(records, show_progress=True)

    total_elapsed = time.time() - total_start
    logger.info(
        "✅ Index build complete: %d records | %.1f minutes total",
        upserted, total_elapsed / 60,
    )
    logger.info("Collection count: %d", index.count())

    return index


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m indexer.build",
        description="CodeLens — Build ChromaDB vector index from Parquet",
    )
    p.add_argument("--parquet", required=True,
                   help="Path to processed Parquet file (e.g. data/processed/python_train.parquet)")
    p.add_argument("--db", default="chroma_db",
                   help="ChromaDB persist directory (default: chroma_db/)")
    p.add_argument("--collection", default="codelens_python",
                   help="Collection name (default: codelens_python)")
    p.add_argument("--batch", type=int, default=32,
                   help="Embedding batch size (default: 32)")
    p.add_argument("--rebuild", action="store_true",
                   help="Delete existing collection before indexing")
    p.add_argument("--device", default=None,
                   help="Force device: cpu | cuda | mps (default: auto)")
    p.add_argument("--cache-dir", default=None,
                   help="HuggingFace model cache directory")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not Path(args.parquet).exists():
        print(f"❌ Parquet file not found: {args.parquet}")
        raise SystemExit(1)

    index = build_index(
        parquet_path=args.parquet,
        persist_dir=args.db,
        collection_name=args.collection,
        batch_size=args.batch,
        rebuild=args.rebuild,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    # Sanity check: peek at first 3 indexed records
    print("\n📋 Sample indexed records:")
    for rec in index.peek(3):
        print(f"  [{rec['language']}] {rec['func_name']} — {rec['repository']}")

    print(f"\n✅ Index ready. Total vectors: {index.count()}")
    print(f"   Collection : {index.collection_name}")
    print(f"   Storage    : {index.persist_dir}")


if __name__ == "__main__":
    main()