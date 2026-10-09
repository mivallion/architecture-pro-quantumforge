"""SQLite metadata storage and derived FAISS index persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Final

import faiss
import numpy as np
from numpy.typing import NDArray

from .index_contracts import InvalidEmbeddingError, TextChunk

SCHEMA: Final = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    content_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_number INTEGER NOT NULL,
    start_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    UNIQUE(document_id, chunk_number)
);
CREATE TABLE IF NOT EXISTS index_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    config_hash TEXT NOT NULL
);
"""


def initialize_schema(connection: sqlite3.Connection) -> None:
    _ = connection.executescript(SCHEMA)


def load_known_hashes(connection: sqlite3.Connection) -> dict[str, str]:
    rows = connection.execute("SELECT source_path, content_hash FROM documents")
    return {str(row[0]): str(row[1]) for row in rows}


def load_configuration_hash(connection: sqlite3.Connection) -> str | None:
    row = connection.execute(
        "SELECT config_hash FROM index_state WHERE singleton = 1"
    ).fetchone()
    return None if row is None else str(row[0])


def save_configuration_hash(connection: sqlite3.Connection, value: str) -> None:
    _ = connection.execute(
        """INSERT INTO index_state(singleton, config_hash) VALUES (1, ?)
           ON CONFLICT(singleton) DO UPDATE SET config_hash = excluded.config_hash""",
        (value,),
    )


def count_chunks(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
    if row is None:
        raise RuntimeError("failed to count indexed chunks")
    return int(row[0])


def delete_documents(
    connection: sqlite3.Connection,
    source_paths: set[str],
) -> None:
    _ = connection.executemany(
        "DELETE FROM documents WHERE source_path = ?",
        ((source_path,) for source_path in source_paths),
    )


def replace_document(
    connection: sqlite3.Connection,
    *,
    source_path: str,
    title: str,
    content_hash: str,
    chunks: tuple[TextChunk, ...],
    vectors: NDArray[np.float32],
) -> None:
    _ = connection.execute(
        """INSERT INTO documents(source_path, title, content_hash) VALUES (?, ?, ?)
           ON CONFLICT(source_path) DO UPDATE SET
               title = excluded.title, content_hash = excluded.content_hash""",
        (source_path, title, content_hash),
    )
    row = connection.execute(
        "SELECT id FROM documents WHERE source_path = ?",
        (source_path,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"document disappeared after upsert: {source_path}")
    document_id = int(row[0])
    _ = connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    for chunk_number, chunk in enumerate(chunks):
        vector = vectors[chunk_number]
        _ = connection.execute(
            """INSERT INTO chunks(
                   document_id, chunk_number, start_index, text, embedding
               ) VALUES (?, ?, ?, ?, ?)""",
            (
                document_id,
                chunk_number,
                chunk.start_index,
                chunk.text,
                sqlite3.Binary(np.ascontiguousarray(vector).tobytes()),
            ),
        )


def rebuild_faiss(
    connection: sqlite3.Connection,
    path: Path,
    dimension: int,
) -> int:
    rows = connection.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
    if rows:
        identifiers = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
        vectors = np.vstack(
            [np.frombuffer(bytes(row[1]), dtype=np.float32) for row in rows]
        )
        expected_shape = (len(rows), dimension)
        if vectors.shape != expected_shape:
            raise InvalidEmbeddingError(expected_shape, vectors.shape)
        index.add_with_ids(vectors, identifiers)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    faiss.write_index(index, str(temporary_path))
    _ = temporary_path.replace(path)
    return len(rows)
