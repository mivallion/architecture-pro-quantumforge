"""Validate consistency between the SQLite metadata and FAISS index."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import override

from .embeddings import DEFAULT_EMBEDDING_DIMENSION


@dataclass(frozen=True, slots=True)
class IndexValidation:
    documents: int
    chunks: int
    dimension: int
    index_size_bytes: int


@dataclass(frozen=True, slots=True)
class IndexValidationError(RuntimeError):
    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


def validate_index(database_path: Path, faiss_path: Path) -> IndexValidation:
    # Loading FAISS after PyTorch avoids the macOS ARM OpenMP conflict.
    import faiss

    if not database_path.is_file():
        raise IndexValidationError(f"metadata database not found: {database_path}")
    if not faiss_path.is_file():
        raise IndexValidationError(f"FAISS index not found: {faiss_path}")

    with sqlite3.connect(database_path) as connection:
        integrity_row = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity_row is None or str(integrity_row[0]) != "ok":
            raise IndexValidationError("SQLite integrity check failed")
        document_row = connection.execute("SELECT COUNT(*) FROM documents").fetchone()
        chunk_rows = connection.execute("SELECT id FROM chunks ORDER BY id").fetchall()
    if document_row is None:
        raise IndexValidationError("failed to count documents")

    index = faiss.read_index(str(faiss_path))
    faiss.omp_set_num_threads(1)
    chunk_ids = tuple(int(row[0]) for row in chunk_rows)
    if int(index.ntotal) != len(chunk_ids):
        raise IndexValidationError(
            f"FAISS contains {int(index.ntotal)} vectors, SQLite contains {len(chunk_ids)} chunks"
        )
    if not hasattr(index, "id_map"):
        raise IndexValidationError("FAISS index has no explicit ID map")
    faiss_ids = tuple(
        int(value)
        for value in faiss.vector_to_array(index.id_map)  # pyright: ignore[reportAttributeAccessIssue]
    )
    if faiss_ids != chunk_ids:
        raise IndexValidationError("FAISS identifiers do not match SQLite chunk IDs")
    dimension = int(index.d)
    if dimension != DEFAULT_EMBEDDING_DIMENSION:
        raise IndexValidationError(
            f"FAISS dimension is {dimension}, expected {DEFAULT_EMBEDDING_DIMENSION}"
        )
    return IndexValidation(
        documents=int(document_row[0]),
        chunks=len(chunk_ids),
        dimension=dimension,
        index_size_bytes=faiss_path.stat().st_size,
    )


__all__ = ["IndexValidation", "IndexValidationError", "validate_index"]
