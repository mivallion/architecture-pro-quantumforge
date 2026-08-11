from __future__ import annotations

import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pytest
from numpy.typing import NDArray

from app.index_store import initialize_schema
from app.search_index import (
    IndexDimensionMismatchError,
    SearchResult,
    search_index,
)


class FakeQueryEmbedder:
    dimension: int

    def __init__(self, *, dimension: int = 2) -> None:
        self.dimension = dimension

    def embed_query(self, query: str) -> NDArray[np.float32]:
        _ = query
        return np.asarray([1.0, 0.0], dtype=np.float32)


def _create_search_data(tmp_path: Path) -> tuple[Path, Path]:
    database_path = tmp_path / "metadata.sqlite3"
    faiss_path = tmp_path / "faiss.index"
    with sqlite3.connect(database_path) as connection:
        initialize_schema(connection)
        _ = connection.execute(
            """INSERT INTO documents(id, source_path, title, content_hash)
               VALUES (1, 'healer.md', 'Healer', 'hash-1'),
                      (2, 'merchant.md', 'Merchant', 'hash-2')"""
        )
        _ = connection.execute(
            """INSERT INTO chunks(
                   id, document_id, chunk_number, start_index, text, embedding
               ) VALUES (?, ?, ?, ?, ?, ?), (?, ?, ?, ?, ?, ?)""",
            (
                10,
                1,
                0,
                15,
                "The healer treats wounded settlers.",
                sqlite3.Binary(np.asarray([1.0, 0.0], dtype=np.float32).tobytes()),
                20,
                2,
                0,
                25,
                "The merchant sells supplies.",
                sqlite3.Binary(np.asarray([0.0, 1.0], dtype=np.float32).tobytes()),
            ),
        )

    index = faiss.IndexIDMap2(faiss.IndexFlatIP(2))
    index.add_with_ids(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([10, 20], dtype=np.int64),
    )
    faiss.write_index(index, str(faiss_path))
    return faiss_path, database_path


def test_search_index_returns_ranked_chunk_with_metadata(tmp_path: Path) -> None:
    # Given
    faiss_path, database_path = _create_search_data(tmp_path)

    # When
    results = search_index(
        "Who heals settlers?",
        FakeQueryEmbedder(),
        faiss_path=faiss_path,
        database_path=database_path,
        limit=2,
    )

    # Then
    assert results[0] == SearchResult(
        chunk_id=10,
        score=1.0,
        title="Healer",
        source_path="healer.md",
        chunk_number=0,
        start_index=15,
        text="The healer treats wounded settlers.",
    )
    assert [result.chunk_id for result in results] == [10, 20]


def test_search_index_rejects_embedding_dimension_mismatch(tmp_path: Path) -> None:
    # Given
    faiss_path, database_path = _create_search_data(tmp_path)
    embedder = FakeQueryEmbedder(dimension=3)

    # When / Then
    with pytest.raises(IndexDimensionMismatchError, match="FAISS index dimension"):
        _ = search_index(
            "Who heals settlers?",
            embedder,
            faiss_path=faiss_path,
            database_path=database_path,
        )
