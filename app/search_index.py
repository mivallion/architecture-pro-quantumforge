"""Search the generated FAISS index and resolve chunk metadata from SQLite."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from .embeddings import QwenEmbedder
from .rag_contracts import RetrievedChunk
from .retrieval import (
    ChunkMetadataNotFoundError,
    FaissRetriever,
    IndexDimensionMismatchError,
    InvalidQueryEmbeddingError,
    QueryEmbedder,
)

DEFAULT_QUERY = "Кто лечит раненых поселенцев?"
DEFAULT_LIMIT = 3

__all__ = [
    "ChunkMetadataNotFoundError",
    "IndexDimensionMismatchError",
    "InvalidQueryEmbeddingError",
    "QueryEmbedder",
    "SearchResult",
    "main",
    "search_index",
]


# Kept as an alias for the Task 3 CLI and its public import path.
SearchResult = RetrievedChunk


def search_index(
    query: str,
    embedder: QueryEmbedder,
    *,
    faiss_path: Path,
    database_path: Path,
    limit: int = DEFAULT_LIMIT,
) -> tuple[RetrievedChunk, ...]:
    """Backward-compatible one-shot wrapper for the Task 3 command-line tool."""
    return FaissRetriever(
        embedder,
        faiss_path=faiss_path,
        database_path=database_path,
    ).retrieve(query, limit=limit)


def main(arguments: Sequence[str] | None = None) -> None:
    raw_arguments = tuple(sys.argv[1:] if arguments is None else arguments)
    query = " ".join(raw_arguments).strip() or DEFAULT_QUERY
    repository_root = Path(__file__).resolve().parent.parent
    index_dir = repository_root / "Task3" / "index"
    results = search_index(
        query,
        QwenEmbedder(),
        faiss_path=index_dir / "faiss.index",
        database_path=index_dir / "metadata.sqlite3",
    )
    print(f"Query: {query}")
    for rank, result in enumerate(results, start=1):
        header = " ".join(
            (
                f"\n{rank}. score={result.score:.3f} | {result.title} |",
                f"{result.source_path}#{result.chunk_number} |",
                f"start={result.start_index}",
            )
        )
        print(f"{header}\n{result.text}")


if __name__ == "__main__":
    main()
