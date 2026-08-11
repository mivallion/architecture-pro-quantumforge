from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter
from numpy.typing import NDArray
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from app.indexing import (
    IndexConfig,
    KnowledgeBaseNotFoundError,
    RecursiveChunker,
    build_index,
)


class FakeEmbedder:
    model_name: str = "test/embedding-model"
    dimension: int = 4
    normalized: bool = True

    def __init__(self) -> None:
        self.encoded_texts: list[str] = []
        self.calls: int = 0

    def embed_documents(self, texts: tuple[str, ...]) -> NDArray[np.float32]:
        self.calls += 1
        self.encoded_texts.extend(texts)
        vectors: list[NDArray[np.float32]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vector = np.frombuffer(digest[: self.dimension], dtype=np.uint8).astype(
                np.float32
            )
            vectors.append(vector / np.linalg.norm(vector))
        return np.vstack(vectors).astype(np.float32)


def _write_document(path: Path, *, title: str, body: str) -> None:
    _ = path.write_text(
        f"---\nentity: {title}\ncategory: test\n---\n\n# {title}\n\n{body}\n",
        encoding="utf-8",
    )


def _config(tmp_path: Path) -> IndexConfig:
    return IndexConfig(
        knowledge_base_dir=tmp_path / "knowledge_base",
        database_path=tmp_path / "index.sqlite3",
        faiss_path=tmp_path / "faiss.index",
    )


def _chunker(*, chunk_size: int = 80) -> RecursiveChunker:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=10,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )
    return RecursiveChunker(splitter, chunk_size=chunk_size, chunk_overlap=10)


def _word_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")


def test_recursive_chunker_does_not_create_heading_only_chunk() -> None:
    # Given
    chunker = RecursiveChunker.from_huggingface_tokenizer(
        _word_tokenizer(),
        chunk_size=20,
        chunk_overlap=2,
    )
    text = "Lead paragraph.\n\n## Behavior\n\n" + "behavior detail " * 30

    # When
    chunks = chunker.split(text)

    # Then
    assert all(chunk.text != "## Behavior" for chunk in chunks)
    assert any("## Behavior" in chunk.text for chunk in chunks)


def test_build_index_rejects_missing_knowledge_base(tmp_path: Path) -> None:
    # Given
    config = _config(tmp_path)

    # When / Then
    with pytest.raises(KnowledgeBaseNotFoundError, match="knowledge_base"):
        _ = build_index(config, FakeEmbedder(), _chunker())


def test_build_index_persists_chunks_metadata_and_faiss_vectors(
    tmp_path: Path,
) -> None:
    # Given
    config = _config(tmp_path)
    config.knowledge_base_dir.mkdir()
    _write_document(
        config.knowledge_base_dir / "alpha.md",
        title="Alpha",
        body="First paragraph about an ancient gate. " * 8,
    )
    embedder = FakeEmbedder()

    # When
    result = build_index(config, embedder, _chunker())

    # Then
    assert result.scanned_documents == 1
    assert result.indexed_documents == 1
    assert result.indexed_chunks == result.total_chunks
    assert result.index_changed is True
    assert result.unchanged_documents == 0
    assert result.total_chunks > 1
    assert embedder.calls == 1
    assert faiss.read_index(str(config.faiss_path)).ntotal == result.total_chunks

    with sqlite3.connect(config.database_path) as connection:
        rows = connection.execute(
            """SELECT d.source_path, d.title, c.chunk_number, c.start_index, c.text
               FROM chunks AS c JOIN documents AS d ON d.id = c.document_id
               ORDER BY c.chunk_number"""
        ).fetchall()

    assert rows[0][0:4] == ("alpha.md", "Alpha", 0, 0)
    assert all(row[4] for row in rows)


def test_unchanged_document_is_not_embedded_again(tmp_path: Path) -> None:
    # Given
    config = _config(tmp_path)
    config.knowledge_base_dir.mkdir()
    _write_document(config.knowledge_base_dir / "alpha.md", title="Alpha", body="One")
    embedder = FakeEmbedder()
    chunker = _chunker()
    _ = build_index(config, embedder, chunker)
    calls_after_first_build = len(embedder.encoded_texts)
    faiss_modified_at = config.faiss_path.stat().st_mtime_ns

    # When
    result = build_index(config, embedder, chunker)

    # Then
    assert result.indexed_documents == 0
    assert result.indexed_chunks == 0
    assert result.index_changed is False
    assert result.unchanged_documents == 1
    assert len(embedder.encoded_texts) == calls_after_first_build
    assert config.faiss_path.stat().st_mtime_ns == faiss_modified_at


def test_changed_document_replaces_old_chunks(tmp_path: Path) -> None:
    # Given
    config = _config(tmp_path)
    config.knowledge_base_dir.mkdir()
    document_path = config.knowledge_base_dir / "alpha.md"
    _write_document(document_path, title="Alpha", body="Short body")
    embedder = FakeEmbedder()
    chunker = _chunker(chunk_size=60)
    first = build_index(config, embedder, chunker)

    # When
    _write_document(document_path, title="Alpha", body="Expanded body. " * 30)
    second = build_index(config, embedder, chunker)

    # Then
    assert first.total_chunks == 1
    assert second.indexed_documents == 1
    assert second.total_chunks > first.total_chunks
    assert faiss.read_index(str(config.faiss_path)).ntotal == second.total_chunks


def test_deleted_document_is_removed_from_metadata_and_faiss(tmp_path: Path) -> None:
    # Given
    config = _config(tmp_path)
    config.knowledge_base_dir.mkdir()
    alpha = config.knowledge_base_dir / "alpha.md"
    beta = config.knowledge_base_dir / "beta.md"
    _write_document(alpha, title="Alpha", body="Alpha facts")
    _write_document(beta, title="Beta", body="Beta facts")
    embedder = FakeEmbedder()
    chunker = _chunker()
    _ = build_index(config, embedder, chunker)

    # When
    alpha.unlink()
    result = build_index(config, embedder, chunker)

    # Then
    assert result.deleted_documents == 1
    assert result.total_chunks == 1
    assert faiss.read_index(str(config.faiss_path)).ntotal == 1
    with sqlite3.connect(config.database_path) as connection:
        paths = connection.execute("SELECT source_path FROM documents").fetchall()
    assert paths == [("beta.md",)]


def test_chunking_configuration_change_forces_reindex(tmp_path: Path) -> None:
    # Given
    config = _config(tmp_path)
    config.knowledge_base_dir.mkdir()
    _write_document(
        config.knowledge_base_dir / "alpha.md",
        title="Alpha",
        body="A sufficiently long document. " * 20,
    )
    embedder = FakeEmbedder()
    _ = build_index(config, embedder, _chunker(chunk_size=100))
    calls_after_first_build = len(embedder.encoded_texts)

    # When
    result = build_index(config, embedder, _chunker(chunk_size=60))

    # Then
    assert result.indexed_documents == 1
    assert len(embedder.encoded_texts) > calls_after_first_build
