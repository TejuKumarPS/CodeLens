"""
data_loader.loader
==================
Fetches the CodeSearchNet dataset from HuggingFace Datasets.

Supported languages: python, java, javascript, go, php, ruby
Default: python (used for this project)

Usage
-----
    from data_loader.loader import load_raw

    train_ds = load_raw(split="train", limit=5000)
    for record in train_ds:
        print(record["func_name"])
"""

import logging
from typing import Optional, Iterator

logger = logging.getLogger(__name__)

# HuggingFace dataset identifier
DATASET_NAME = "code-search-net/code_search_net"

# Valid splits
VALID_SPLITS = {"train", "validation", "test"}

# Valid languages
VALID_LANGUAGES = {"python", "java", "javascript", "go", "php", "ruby"}


def load_raw(
    split: str = "train",
    language: str = "python",
    limit: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Iterator[dict]:
    """
    Load raw records from CodeSearchNet via HuggingFace Datasets.

    Parameters
    ----------
    split : str
        One of "train" | "validation" | "test"
    language : str
        One of "python" | "java" | "javascript" | "go" | "php" | "ruby"
    limit : int, optional
        If set, yield at most this many records. Useful for dev/test runs.
    cache_dir : str, optional
        HuggingFace cache directory. Defaults to ~/.cache/huggingface.

    Yields
    ------
    dict
        Raw HuggingFace dataset record with keys:
        func_name, func_code_string, func_documentation_string,
        repository_name, func_code_url, func_code_tokens, language
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    if language not in VALID_LANGUAGES:
        raise ValueError(f"language must be one of {VALID_LANGUAGES}, got {language!r}")

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "HuggingFace datasets library not installed. "
            "Run: pip install datasets"
        ) from e

    logger.info(
        "Loading CodeSearchNet | language=%s | split=%s | limit=%s",
        language,
        split,
        limit if limit else "all",
    )

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    dataset = load_dataset(DATASET_NAME, language, split=split, **kwargs)

    count = 0
    for record in dataset:
        yield record
        count += 1
        if limit is not None and count >= limit:
            logger.info("Limit of %d records reached. Stopping.", limit)
            break

    logger.info("Loaded %d raw records from split=%s", count, split)


def get_split_sizes(language: str = "python") -> dict:
    """
    Return the number of records in each split without downloading full data.

    Returns
    -------
    dict
        {"train": N, "validation": N, "test": N}
    """
    try:
        from datasets import load_dataset_builder
    except ImportError as e:
        raise ImportError("pip install datasets") from e

    builder = load_dataset_builder(DATASET_NAME, language)
    info = builder.info
    sizes = {}
    for split_name, split_info in info.splits.items():
        sizes[split_name] = split_info.num_examples
    return sizes