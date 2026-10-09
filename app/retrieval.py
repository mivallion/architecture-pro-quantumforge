"""Long-lived FAISS retrieval over the Task 3 index."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from typing_extensions import override

import numpy as np
from numpy.typing import NDArray

from .rag_contracts import RetrievedChunk


class QueryEmbedder(Protocol):
    """The query-only part of the embedding model used for indexing."""

    @property
    def dimension(self) -> int: ...

    def embed_query(self, query: str) -> NDArray[np.float32]: ...


@dataclass(frozen=True, slots=True)
class IndexDimensionMismatchError(Exception):
    index_dimension: int
    embedder_dimension: int

    @override
    def __str__(self) -> str:
        return (
            f"FAISS index dimension is {self.index_dimension}, "
            f"but embedder dimension is {self.embedder_dimension}"
        )


@dataclass(frozen=True, slots=True)
class InvalidQueryEmbeddingError(Exception):
    expected_shape: tuple[int, ...]
    actual_shape: tuple[int, ...]

    @override
    def __str__(self) -> str:
        return (
            f"expected query embedding shape {self.expected_shape}, "
            f"got {self.actual_shape}"
        )


@dataclass(frozen=True, slots=True)
class ChunkMetadataNotFoundError(Exception):
    chunk_id: int

    @override
    def __str__(self) -> str:
        return f"FAISS chunk {self.chunk_id} has no SQLite metadata"


class FaissRetriever:
    """Keep a FAISS index in memory while resolving metadata per request."""

    def __init__(
        self,
        embedder: QueryEmbedder,
        *,
        faiss_path: Path,
        database_path: Path,
    ) -> None:
        # Importing FAISS only after PyTorch avoids an OpenMP crash on macOS ARM64.
        import faiss

        self._embedder = embedder
        self._database_uri = f"{database_path.resolve().as_uri()}?mode=ro"
        self._faiss = faiss
        self._index = faiss.read_index(str(faiss_path))
        self._lock = threading.Lock()

        index_dimension = int(self._index.d)
        if index_dimension != embedder.dimension:
            raise IndexDimensionMismatchError(index_dimension, embedder.dimension)
        self._dimension = index_dimension

        # Avoid a native OpenMP conflict between PyTorch and faiss-cpu on macOS ARM64.
        faiss.omp_set_num_threads(1)

    def retrieve(self, query: str, *, limit: int) -> tuple[RetrievedChunk, ...]:
        if limit <= 0:
            raise ValueError("search limit must be positive")
        if self._index.ntotal == 0:
            return ()

        with self._lock:
            vector = np.asarray(self._embedder.embed_query(query), dtype=np.float32)
            expected_shape = (self._dimension,)
            if vector.shape != expected_shape or not np.isfinite(vector).all():
                raise InvalidQueryEmbeddingError(expected_shape, vector.shape)
            query_matrix = np.ascontiguousarray(vector.reshape(1, -1))

            # This is process-global in FAISS, so retain the macOS workaround.
            self._faiss.omp_set_num_threads(1)
            scores, identifiers = self._index.search(
                query_matrix,
                min(limit, int(self._index.ntotal)),
            )

        with sqlite3.connect(self._database_uri, uri=True) as connection:
            return tuple(
                _load_chunk(connection, int(chunk_id), float(score))
                for score, chunk_id in zip(scores[0], identifiers[0], strict=True)
                if chunk_id >= 0
            )


def _load_chunk(
    connection: sqlite3.Connection,
    chunk_id: int,
    score: float,
) -> RetrievedChunk:
    row = connection.execute(
        """SELECT d.title, d.source_path, c.chunk_number, c.start_index, c.text
           FROM chunks AS c
           JOIN documents AS d ON d.id = c.document_id
           WHERE c.id = ?""",
        (chunk_id,),
    ).fetchone()
    if row is None:
        raise ChunkMetadataNotFoundError(chunk_id)
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=score,
        title=str(row[0]),
        source_path=str(row[1]),
        chunk_number=int(row[2]),
        start_index=int(row[3]),
        text=str(row[4]),
    )


__all__ = [
    "ChunkMetadataNotFoundError",
    "FaissRetriever",
    "IndexDimensionMismatchError",
    "InvalidQueryEmbeddingError",
    "QueryEmbedder",
]
