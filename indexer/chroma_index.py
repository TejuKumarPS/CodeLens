"""
indexer.chroma_index
=====================
Wraps ChromaDB to provide a clean, typed interface for CodeLens.

Design decisions:
  - Cosine similarity (not L2) because all our vectors are L2-normalised.
    With unit vectors, cosine similarity == dot product, giving scores in [-1, 1].
  - Persistent storage: ChromaDB writes to disk so the index survives restarts.
    One build (~4900 records) takes ~2 min; subsequent runs load in <1 sec.
  - Upsert semantics: adding the same record ID twice updates it, never duplicates.
  - Metadata stored per record: func_name, repository, url, language, docstring_preview.
    This lets the retriever return rich results without touching the Parquet file.

ChromaDB collection naming:
  Default: "codelens_python"  — one collection per language.

Usage
-----
    from indexer import CodeLensIndex

    idx = CodeLensIndex(persist_dir="chroma_db/")
    idx.upsert(records)                        # list[CodeRecord] with embeddings set
    results = idx.search(query_vector, top_k=20)
    print(idx.count())
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from data_loader.models import CodeRecord

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "codelens_python"
DEFAULT_PERSIST_DIR = "chroma_db"


class CodeLensIndex:
    """
    Manages the ChromaDB vector collection for CodeLens.

    Parameters
    ----------
    persist_dir : str
        Directory where ChromaDB writes its SQLite + HNSW files.
        Created automatically if it doesn't exist.
    collection_name : str
        Name of the ChromaDB collection. Default: "codelens_python".
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as e:
            raise ImportError("pip install chromadb") from e

        import chromadb
        from chromadb.config import Settings

        self._persist_dir = str(Path(persist_dir).resolve())
        self._collection_name = collection_name

        logger.info(
            "Initialising ChromaDB | dir=%s | collection=%s",
            self._persist_dir, collection_name,
        )

        # Persistent client — survives process restarts
        self._client = chromadb.PersistentClient(path=self._persist_dir)

        # get_or_create: safe to call multiple times
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine similarity ANN
        )

        logger.info(
            "Collection '%s' ready. Current count: %d",
            collection_name, self._collection.count(),
        )

    # ── Write operations ──────────────────────────────────────────────────────

    def upsert(
        self,
        records: List[CodeRecord],
        batch_size: int = 500,
        show_progress: bool = True,
    ) -> int:
        """
        Upsert a list of CodeRecord into ChromaDB.

        Records without embeddings are silently skipped with a warning.
        Upsert is idempotent: re-indexing the same record updates it in-place.

        Parameters
        ----------
        records : list[CodeRecord]
            Must have record.embedding set (done by embedder.embed_records).
        batch_size : int
            Records per ChromaDB upsert call. 500 is a safe default.
        show_progress : bool
            Print batch progress to logger.

        Returns
        -------
        int
            Number of records successfully upserted.
        """
        valid = [r for r in records if r.embedding is not None]
        skipped = len(records) - len(valid)
        if skipped:
            logger.warning(
                "%d records skipped — embedding is None. "
                "Run embedder.embed_records() first.", skipped,
            )
        if not valid:
            logger.warning("No records to upsert.")
            return 0

        logger.info("Upserting %d records into '%s'...", len(valid), self._collection_name)
        start = time.time()
        upserted = 0

        for batch_start in range(0, len(valid), batch_size):
            batch = valid[batch_start : batch_start + batch_size]

            ids = [r.id for r in batch]
            embeddings = [r.embedding for r in batch]      # list[list[float]]
            metadatas = [r.to_chroma_metadata() for r in batch]
            documents = [r.func_code[:500] for r in batch] # searchable code snippet

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            upserted += len(batch)

            if show_progress:
                logger.info(
                    "  Upserted %d / %d (%.0f%%)",
                    upserted, len(valid), upserted / len(valid) * 100,
                )

        elapsed = time.time() - start
        logger.info(
            "Upsert complete: %d records in %.1fs (%.0f rec/s)",
            upserted, elapsed, upserted / elapsed if elapsed > 0 else 0,
        )
        return upserted

    def delete_collection(self) -> None:
        """
        Delete the entire collection and all its data.
        Useful for re-indexing from scratch.
        """
        self._client.delete_collection(self._collection_name)
        logger.warning("Collection '%s' deleted.", self._collection_name)

        # Re-create empty collection
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Empty collection '%s' re-created.", self._collection_name)

    # ── Read operations ───────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Perform ANN search and return the top-K nearest code functions.

        Parameters
        ----------
        query_vector : np.ndarray, shape (768,)
            L2-normalised query embedding (output of embed_query().fused_vector).
        top_k : int
            Number of results to return. Default 20 (for cross-encoder re-ranking).

        Returns
        -------
        list[dict]
            Each dict has keys:
              id         : str    — CodeRecord.id
              score      : float  — cosine similarity in [0, 1] (ChromaDB convention)
              func_name  : str
              repository : str
              url        : str
              language   : str
              docstring_preview : str
              code_preview      : str
              document   : str   — raw code snippet stored at index time
        """
        if self._collection.count() == 0:
            logger.warning("Collection is empty. Index records first.")
            return []

        query_list = query_vector.tolist()

        results = self._collection.query(
            query_embeddings=[query_list],
            n_results=min(top_k, self._collection.count()),
            include=["metadatas", "distances", "documents"],
        )

        # ChromaDB returns distances, not similarities.
        # For cosine space: similarity = 1 - distance
        hits = []
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]

        for rec_id, dist, meta, doc in zip(ids, distances, metadatas, documents):
            hits.append({
                "id": rec_id,
                "score": round(1.0 - dist, 6),   # cosine similarity
                "func_name": meta.get("func_name", ""),
                "repository": meta.get("repository", ""),
                "url": meta.get("url", ""),
                "language": meta.get("language", ""),
                "docstring_preview": meta.get("docstring_preview", ""),
                "code_preview": meta.get("code_preview", ""),
                "document": doc,
            })

        return hits

    def count(self) -> int:
        """Return number of vectors currently in the collection."""
        return self._collection.count()

    def peek(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Return the first n records stored in the collection.
        Useful for sanity-checking the index.

        Returns
        -------
        list[dict]
            Each dict has: id, func_name, repository, language
        """
        result = self._collection.peek(limit=n)
        output = []
        for i, rec_id in enumerate(result["ids"]):
            meta = result["metadatas"][i] if result["metadatas"] else {}
            output.append({
                "id": rec_id,
                "func_name": meta.get("func_name", ""),
                "repository": meta.get("repository", ""),
                "language": meta.get("language", ""),
            })
        return output

    def get_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single record by its CodeRecord.id.

        Returns
        -------
        dict or None
            Metadata dict if found, None if not in collection.
        """
        result = self._collection.get(
            ids=[record_id],
            include=["metadatas", "documents"],
        )
        if not result["ids"]:
            return None
        meta = result["metadatas"][0]
        return {
            "id": result["ids"][0],
            "func_name": meta.get("func_name", ""),
            "repository": meta.get("repository", ""),
            "url": meta.get("url", ""),
            "language": meta.get("language", ""),
            "docstring_preview": meta.get("docstring_preview", ""),
            "code_preview": meta.get("code_preview", ""),
            "document": result["documents"][0],
        }

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def persist_dir(self) -> str:
        return self._persist_dir

    def __repr__(self) -> str:
        return (
            f"CodeLensIndex(collection={self._collection_name!r}, "
            f"persist_dir={self._persist_dir!r}, "
            f"count={self.count()})"
        )