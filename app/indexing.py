"""Incremental knowledge-base indexing orchestration."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import RecursiveChunker
from .index_contracts import (
    BuildResult,
    DocumentEmbedder,
    IndexConfig,
    InvalidEmbeddingError,
    KnowledgeBaseNotFoundError,
    TextChunk,
)
from .index_store import (
    count_chunks,
    delete_documents,
    initialize_schema,
    load_configuration_hash,
    load_known_hashes,
    rebuild_faiss,
    replace_document,
    save_configuration_hash,
)


@dataclass(frozen=True, slots=True)
class PendingDocument:
    source_path: str
    title: str
    content_hash: str
    chunks: tuple[TextChunk, ...]


def build_index(
    config: IndexConfig,
    embedder: DocumentEmbedder,
    chunker: RecursiveChunker,
) -> BuildResult:
    started_at = time.perf_counter()
    if not config.knowledge_base_dir.is_dir():
        raise KnowledgeBaseNotFoundError(config.knowledge_base_dir)
    files = tuple(sorted(config.knowledge_base_dir.glob("*.md")))
    config.database_path.parent.mkdir(parents=True, exist_ok=True)
    config.faiss_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = _configuration_hash(embedder, chunker)

    with sqlite3.connect(config.database_path) as connection:
        initialize_schema(connection)
        known_hashes = load_known_hashes(connection)
        force_reindex = load_configuration_hash(connection) != fingerprint
        current_paths = {path.name for path in files}
        deleted_paths = set(known_hashes) - current_paths
        delete_documents(connection, deleted_paths)

        pending_documents: list[PendingDocument] = []
        unchanged = 0
        for path in files:
            raw = path.read_bytes()
            content_hash = hashlib.sha256(raw).hexdigest()
            if not force_reindex and known_hashes.get(path.name) == content_hash:
                unchanged += 1
                continue
            text = raw.decode("utf-8")
            pending_documents.append(
                PendingDocument(
                    source_path=path.name,
                    title=_extract_title(text, path),
                    content_hash=content_hash,
                    chunks=chunker.split(text),
                )
            )

        _embed_and_store(connection, tuple(pending_documents), embedder)

        save_configuration_hash(connection, fingerprint)
        index_changed = bool(
            force_reindex
            or pending_documents
            or deleted_paths
            or not config.faiss_path.is_file()
        )
        if index_changed:
            total_chunks = rebuild_faiss(
                connection,
                config.faiss_path,
                embedder.dimension,
            )
        else:
            total_chunks = count_chunks(connection)

    indexed_chunks = sum(len(document.chunks) for document in pending_documents)

    return BuildResult(
        scanned_documents=len(files),
        indexed_documents=len(pending_documents),
        indexed_chunks=indexed_chunks,
        unchanged_documents=unchanged,
        deleted_documents=len(deleted_paths),
        total_chunks=total_chunks,
        index_changed=index_changed,
        duration_seconds=time.perf_counter() - started_at,
    )


def _configuration_hash(
    embedder: DocumentEmbedder,
    chunker: RecursiveChunker,
) -> str:
    value = "|".join(
        (
            embedder.model_name,
            str(embedder.dimension),
            str(embedder.normalized),
            chunker.strategy,
            str(chunker.chunk_size),
            str(chunker.chunk_overlap),
        )
    )
    return hashlib.sha256(value.encode()).hexdigest()


def _embed_and_store(
    connection: sqlite3.Connection,
    documents: tuple[PendingDocument, ...],
    embedder: DocumentEmbedder,
) -> None:
    texts = tuple(chunk.text for document in documents for chunk in document.chunks)
    if texts:
        vectors = np.asarray(embedder.embed_documents(texts), dtype=np.float32)
    else:
        vectors = np.empty((0, embedder.dimension), dtype=np.float32)
    expected_shape = (len(texts), embedder.dimension)
    if vectors.shape != expected_shape or not np.isfinite(vectors).all():
        raise InvalidEmbeddingError(expected_shape, vectors.shape)

    offset = 0
    for document in documents:
        next_offset = offset + len(document.chunks)
        replace_document(
            connection,
            source_path=document.source_path,
            title=document.title,
            content_hash=document.content_hash,
            chunks=document.chunks,
            vectors=vectors[offset:next_offset],
        )
        offset = next_offset


def _extract_title(text: str, path: Path) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    return path.stem


__all__ = [
    "IndexConfig",
    "KnowledgeBaseNotFoundError",
    "RecursiveChunker",
    "build_index",
]
