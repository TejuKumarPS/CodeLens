"""
data_loader.cleaner
===================
Transforms raw HuggingFace CodeSearchNet records into validated CodeRecord
instances, or returns None if a record is too low-quality to index.

Quality filters applied:
  1. func_code_string must be non-empty and >= MIN_CODE_TOKENS tokens
  2. func_documentation_string must be non-empty and >= MIN_DOC_CHARS chars
  3. func_code_string length must not exceed MAX_CODE_CHARS (truncation risk)
  4. docstring must not be a copy of the function signature (boilerplate)
  5. Duplicate detection by (repo, func_name) is done at pipeline level

Usage
-----
    from data_loader.cleaner import clean_record

    raw = next(load_raw("train", limit=1))
    record = clean_record(raw, idx=0, partition="train")
    if record:
        print(record)
"""

import re
import logging
from typing import Optional

from .models import CodeRecord

logger = logging.getLogger(__name__)

# ── Quality thresholds ──────────────────────────────────────────────────────
MIN_CODE_TOKENS: int = 10       # functions with < 10 tokens are stubs
MAX_CODE_CHARS: int = 8_000     # very long functions may OOM the encoder
MIN_DOC_CHARS: int = 15         # single-word docstrings are useless for IR
MAX_DOC_CHARS: int = 2_000      # overly long docstrings get truncated

# Regex to detect boilerplate docstrings that are just the function signature
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*@\w+"),          # decorator-only
    re.compile(r"^\s*\.\.\.\s*$"),    # ellipsis body
    re.compile(r"^\s*pass\s*$"),      # pass body
]


def clean_record(
    raw: dict,
    idx: int,
    partition: str,
    language: str = "python",
) -> Optional[CodeRecord]:
    """
    Convert a raw HuggingFace CodeSearchNet dict into a CodeRecord.

    Parameters
    ----------
    raw : dict
        Raw record from HuggingFace datasets.
    idx : int
        Global index of this record (used to build unique id).
    partition : str
        Dataset split: "train" | "validation" | "test"
    language : str
        Programming language, default "python"

    Returns
    -------
    CodeRecord or None
        Returns None if the record fails quality filters.
    """
    # ── Extract raw fields ──────────────────────────────────────────────────
    func_code: str = (raw.get("func_code_string") or "").strip()
    docstring: str = (raw.get("func_documentation_string") or "").strip()
    func_name: str = (raw.get("func_name") or "").strip()
    repository: str = (raw.get("repository_name") or "").strip()
    url: str = (raw.get("func_code_url") or "").strip()
    tokens: list = raw.get("func_code_tokens") or []
    detected_lang: str = (raw.get("language") or language).strip().lower()

    # ── Filter 1: Non-empty code ────────────────────────────────────────────
    if not func_code:
        logger.debug("[idx=%d] Skipped: empty func_code", idx)
        return None

    # ── Filter 2: Minimum token count ──────────────────────────────────────
    if len(tokens) < MIN_CODE_TOKENS:
        logger.debug(
            "[idx=%d] Skipped: too few tokens (%d < %d)",
            idx, len(tokens), MIN_CODE_TOKENS,
        )
        return None

    # ── Filter 3: Maximum code length ──────────────────────────────────────
    if len(func_code) > MAX_CODE_CHARS:
        logger.debug(
            "[idx=%d] Truncating code from %d to %d chars",
            idx, len(func_code), MAX_CODE_CHARS,
        )
        func_code = func_code[:MAX_CODE_CHARS]

    # ── Filter 4: Non-empty, non-trivial docstring ──────────────────────────
    if not docstring or len(docstring) < MIN_DOC_CHARS:
        logger.debug(
            "[idx=%d] Skipped: docstring too short (%d chars)",
            idx, len(docstring),
        )
        return None

    # ── Filter 5: Boilerplate docstring detection ───────────────────────────
    for pattern in _BOILERPLATE_PATTERNS:
        if pattern.match(docstring):
            logger.debug("[idx=%d] Skipped: boilerplate docstring", idx)
            return None

    # ── Truncate docstring if needed ────────────────────────────────────────
    if len(docstring) > MAX_DOC_CHARS:
        docstring = docstring[:MAX_DOC_CHARS]

    # ── Normalize func_name ─────────────────────────────────────────────────
    if not func_name:
        # Fall back to extracting from first line of code
        first_line = func_code.split("\n")[0]
        match = re.search(r"def\s+(\w+)", first_line)
        func_name = match.group(1) if match else f"func_{idx}"

    # ── Build unique ID ─────────────────────────────────────────────────────
    # Format: "<repo>::<func_name>::<partition>_<idx>"
    safe_repo = re.sub(r"[^a-zA-Z0-9_\-]", "_", repository) if repository else "unknown"
    safe_func = re.sub(r"[^a-zA-Z0-9_]", "_", func_name)
    record_id = f"{safe_repo}::{safe_func}::{partition}_{idx}"

    return CodeRecord(
        id=record_id,
        func_name=func_name,
        func_code=func_code,
        docstring=docstring,
        language=detected_lang,
        repository=repository,
        url=url,
        tokens=list(tokens),
        partition=partition,
    )


def compute_cleaning_stats(records_raw: int, records_kept: int) -> dict:
    """
    Return a summary dict of cleaning statistics.

    Parameters
    ----------
    records_raw : int
        Total records seen before filtering.
    records_kept : int
        Records that passed all filters.

    Returns
    -------
    dict
        {"raw": N, "kept": N, "dropped": N, "retention_pct": float}
    """
    dropped = records_raw - records_kept
    retention = (records_kept / records_raw * 100) if records_raw else 0.0
    return {
        "raw": records_raw,
        "kept": records_kept,
        "dropped": dropped,
        "retention_pct": round(retention, 2),
    }