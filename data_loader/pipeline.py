"""
data_loader.pipeline
====================
Orchestrates the full data loading pipeline:
  1. Load raw records from CodeSearchNet (HuggingFace)
  2. Clean and filter each record
  3. Deduplicate by (repository, func_name)
  4. Serialize cleaned records to Parquet
  5. Provide a load_processed() reader for downstream modules

This is the ONLY module downstream code (embedder, indexer, etc.) should
import — not loader.py or cleaner.py directly.

CLI Usage
---------
    python -m data_loader.pipeline \\
        --split train \\
        --language python \\
        --limit 5000 \\
        --output data/processed/

    python -m data_loader.pipeline --split test --limit 500 --output data/processed/
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from .loader import load_raw
from .cleaner import clean_record, compute_cleaning_stats
from .models import CodeRecord

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# Parquet file naming convention: <language>_<split>.parquet
PARQUET_FILENAME_TEMPLATE = "{language}_{split}.parquet"
STATS_FILENAME_TEMPLATE = "{language}_{split}_stats.json"


def process_split(
    split: str = "train",
    language: str = "python",
    limit: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[CodeRecord]:
    """
    Run the full cleaning pipeline for one dataset split.

    Parameters
    ----------
    split : str
        "train" | "validation" | "test"
    language : str
        "python" | "java" | ... (see loader.py)
    limit : int, optional
        Cap on number of raw records to read.
    cache_dir : str, optional
        HuggingFace dataset cache directory.

    Returns
    -------
    list[CodeRecord]
        All records that passed quality filters, deduplicated.
    """
    logger.info("=== Processing split: %s | language: %s | limit: %s ===",
                split, language, limit or "all")
    start = time.time()

    records: List[CodeRecord] = []
    seen_ids: set = set()           # duplicate detection by (repo, func_name)
    raw_count = 0

    raw_iter = load_raw(split=split, language=language, limit=limit, cache_dir=cache_dir)

    for idx, raw in enumerate(tqdm(raw_iter, desc=f"[{split}] Cleaning", unit="rec")):
        raw_count += 1
        record = clean_record(raw, idx=idx, partition=split, language=language)
        if record is None:
            continue

        # Deduplication key: same repo + same function name
        dedup_key = f"{record.repository}::{record.func_name}"
        if dedup_key in seen_ids:
            logger.debug("Duplicate skipped: %s", dedup_key)
            continue
        seen_ids.add(dedup_key)

        records.append(record)

    elapsed = time.time() - start
    stats = compute_cleaning_stats(raw_count, len(records))
    stats["elapsed_seconds"] = round(elapsed, 2)
    stats["split"] = split
    stats["language"] = language
    stats["limit"] = limit

    logger.info(
        "Split '%s' done: raw=%d | kept=%d | dropped=%d | retention=%.1f%% | %.1fs",
        split, stats["raw"], stats["kept"], stats["dropped"],
        stats["retention_pct"], elapsed,
    )
    return records


def save_processed(
    records: List[CodeRecord],
    output_dir: str,
    split: str,
    language: str = "python",
) -> Path:
    """
    Serialize a list of CodeRecord to a Parquet file.

    Parameters
    ----------
    records : list[CodeRecord]
    output_dir : str
        Directory where the Parquet file will be written.
    split : str
        Used in the filename.
    language : str
        Used in the filename.

    Returns
    -------
    Path
        Path to the written Parquet file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = PARQUET_FILENAME_TEMPLATE.format(language=language, split=split)
    out_path = out_dir / filename

    if not records:
        logger.warning("save_processed called with empty records list — writing empty Parquet.")
        df = pd.DataFrame(columns=[
            "id", "func_name", "func_code", "docstring", "language",
            "repository", "url", "tokens", "partition",
        ])
        df.to_parquet(out_path, index=False, engine="pyarrow")
        return out_path

    rows = [r.to_dict() for r in records]
    df = pd.DataFrame(rows)

    # Tokens column is a list — store as JSON string for Parquet compatibility
    df["tokens"] = df["tokens"].apply(json.dumps)

    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Saved %d records → %s (%.2f MB)",
                len(records), out_path,
                out_path.stat().st_size / 1_048_576)
    return out_path


def load_processed(
    parquet_path: str,
) -> List[CodeRecord]:
    """
    Load a previously saved Parquet file back into CodeRecord instances.

    Parameters
    ----------
    parquet_path : str
        Path to the Parquet file written by save_processed().

    Returns
    -------
    list[CodeRecord]
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path, engine="pyarrow")
    # Restore tokens from JSON string
    df["tokens"] = df["tokens"].apply(json.loads)

    records = [CodeRecord.from_dict(row) for row in df.to_dict(orient="records")]
    logger.info("Loaded %d records ← %s", len(records), path)
    return records


def save_stats(stats: dict, output_dir: str, split: str, language: str = "python") -> Path:
    """Save pipeline statistics as JSON for reproducibility."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = STATS_FILENAME_TEMPLATE.format(language=language, split=split)
    out_path = out_dir / filename
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats saved → %s", out_path)
    return out_path


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m data_loader.pipeline",
        description="CodeLens — Data Loading & Preprocessing Pipeline",
    )
    p.add_argument("--split", default="train",
                   choices=["train", "validation", "test"],
                   help="Dataset split to process (default: train)")
    p.add_argument("--language", default="python",
                   choices=["python", "java", "javascript", "go", "php", "ruby"],
                   help="Programming language (default: python)")
    p.add_argument("--limit", type=int, default=None,
                   help="Max raw records to read. Omit for full dataset.")
    p.add_argument("--output", default="data/processed/",
                   help="Output directory for Parquet files (default: data/processed/)")
    p.add_argument("--cache-dir", default=None,
                   help="HuggingFace dataset cache directory")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    records = process_split(
        split=args.split,
        language=args.language,
        limit=args.limit,
        cache_dir=args.cache_dir,
    )

    if not records:
        logger.warning("No records produced. Check filters or dataset availability.")
        return

    parquet_path = save_processed(
        records=records,
        output_dir=args.output,
        split=args.split,
        language=args.language,
    )

    stats = compute_cleaning_stats(
        records_raw=args.limit or len(records),  # approximate if limit not set
        records_kept=len(records),
    )
    stats.update({"split": args.split, "language": args.language,
                  "output_path": str(parquet_path)})
    save_stats(stats, args.output, args.split, args.language)

    print("\n✅ Pipeline complete!")
    print(f"   Records saved : {len(records)}")
    print(f"   Output file   : {parquet_path}")


if __name__ == "__main__":
    main()